import argparse

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import torch
from shapely.geometry import Polygon
from shapely.geometry import Point

from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from my_utils.encoder_torch import Extractor
from models.experimental import attempt_load
from my_utils.my_dataset import LoadImages
from my_utils import utils
from deep_sort import detection
from my_utils.queuer import Queuer, PotentialQueuer

fps = 60
identity_switch_thres = 30

def run(
    weights='yolov5l.pt',  # model.pt path(s)
    source='frames',  # file/dir/URL/glob, 0 for webcam
    output_dir='out', 
    finish_area=[866,650,1172,473,1281,555,990,776],
    device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    conf_thres=0.5,  # confidence threshold
    iou_thres=0.2,  # NMS IOU threshold
    line=[0, 300, 1000, 200], # boundary crossing line
    queue_polygon=[717, 101, 1453, 515, 1107, 777, 416, 352],   # x y x y x y x y x y
    enqueue_thres=10, 
    dequeue_thres=10,
    finish_thres=1,
    debug_frames=0, # debug mode
    half=False,  # use FP16 half-precision inference
    save_img=False,
    save_video=False,
):

    queue_vertices = []
    for x, y in zip(*[iter(queue_polygon)]*2):   # loop 2 coords at a time
        queue_vertices.append((x,y))
    queue_polygon = Polygon(queue_vertices)
    finish_vertices = []
    for x, y in zip(*[iter(finish_area)]*2):   # loop 2 coords at a time
        finish_vertices.append((x,y))
    finish_polyogn = Polygon(finish_vertices)
    device = utils.select_device(device)
    use_gpu = device == torch.device('cuda:0')
    print(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if half:
        model.half()
    
    encoder = Extractor(str(Path('model') / Path('ckpt.t7')))
    max_cosine_distance = 0.2
    nn_budget = None
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    dataset = LoadImages(source, img_size=640, stride=stride)

    avg_queue_time = 0.0
    queue = {}
    potential_queue = {}
    queue_time = {}

    dir_path = Path(output_dir)
    file_path = Path('output.txt')
    dir_path.mkdir(exist_ok=True)
    if save_video:
        for _, _, im0s, _, frame_idx in dataset:
            height, width, _ = im0s.shape
            break
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video_writer = cv2.VideoWriter(str(dir_path / Path("out.avi")), fourcc, fps, size)
    for _, img, im0s, _, frame_idx in tqdm(dataset):
        if debug_frames > 0 and frame_idx > debug_frames:
            break
        

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        bgr_image = im0s

        
        results = model(img)[0]
        results = utils.non_max_suppression(results, conf_thres, iou_thres)
        if(results[0].shape[0]==0):
            continue

        det = results[0]
        det[:, :4] = utils.scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
        person_ind = [i for i, cls in enumerate(det[:, -1]) if int(cls) == 0]   
        xyxy = det[person_ind, :-2]  # find person only
        xywh_boxes = utils.xyxy2xywh(xyxy)
        tlwh_boxes = utils.xywh2tlwh(xywh_boxes)
        confidence = det[:, -2]
        if use_gpu:
            tlwh_boxes = tlwh_boxes.cpu()
            xyxy = xyxy.cpu()
        xyxy_boxes = np.array(xyxy).astype(int)

        features = encoder.get_features(xyxy_boxes, bgr_image)
        
        detections = [detection.Detection(bbox, confidence, 'person', feature) for bbox, confidence, feature in zip(tlwh_boxes, confidence, features)]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        in_queueing_area = []   

        if save_img or save_video:  
            cv2.putText(bgr_image, "Queue: {}".format(str(list(queue.keys()))[1:-1]), (100, 50),          
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            num_idx = len(queue_time)
            avg_queue_time = sum(queue_time.values()) / num_idx if num_idx > 0 else 0
            cv2.putText(bgr_image, "Time: {}".format(str((queue_time))[1:-1]), (100, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(bgr_image, "Avg time: {}".format(str(round(avg_queue_time, 1))), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            pts=np.array(queue_vertices,np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(bgr_image,[pts],True,(255,0,0), 2) 
            pts=np.array(finish_vertices,np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(bgr_image,[pts],True,(255,255,0), 2) 


        
            
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id = track.track_id
            bbox = track.to_tlbr()
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            cv2.putText(bgr_image, "ID: " + str(track_id), (int(center_x), int(center_y)), 0,
                    1e-3 * bgr_image.shape[0], (255, 0, 0), 1)
            if queue_polygon.intersects(Point(center_x, center_y)): 
                if track_id in list(potential_queue.keys()):
                    start_frame = potential_queue[track_id].start_frame
                    potential_queue[track_id].accumulated_frames += 1
                    accumulated_frames = potential_queue[track_id].accumulated_frames
                    elapse_frames = frame_idx -  start_frame
                    if elapse_frames > enqueue_thres * fps and accumulated_frames > elapse_frames * 0.8:
                        queue[track_id] = Queuer(start_frame, (center_x, center_y), start_frame, False, None)
                        in_queueing_area.append(track_id)
                        del potential_queue[track_id]
                    if save_video:
                        cv2.rectangle(bgr_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                        cv2.putText(bgr_image, str(round((frame_idx -  start_frame)/fps, 1)), (int(center_x), int(bbox[1])), 0,
                                    1e-3 * bgr_image.shape[0], (0, 255, 0), 1)
                elif track_id in queue:
                    if not queue[track_id].enter_finish_area_frame:
                        queue[track_id].last_frame = frame_idx
                    queue[track_id].position = (center_x, center_y)
                    in_queueing_area.append(track_id)
                    if save_video:
                        cv2.rectangle(bgr_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                        cv2.putText(bgr_image, str(round((frame_idx -  queue[track_id].start_frame)/fps, 1)), (int(center_x), int(bbox[1])), 0,
                                    1e-3 * bgr_image.shape[0], (0, 0, 255), 1)
                else:
                    addToPotential = True
                    '''
                    for queue_track_id in queue:
                        (center_x2, center_y2) = queue[queue_track_id].position
                        if abs(center_x-center_x2) + abs(center_y-center_y2) < identity_switch_thres:
                            queue[track_id] = queue[queue_track_id]
                            addToPotential = False
                            del queue[queue_track_id]
                            break
                    '''
                    if addToPotential:
                        potential_queue[track_id] = PotentialQueuer(frame_idx, 0)
                    
        for track_id in list(queue.keys()):
            queueing_time = round((queue[track_id].last_frame - queue[track_id].start_frame) / fps, 1)
            queue_time[track_id] = queueing_time
            if not queue[track_id].enter_finish_area_frame and finish_polyogn.intersects(Point(queue[track_id].position)):
                queue[track_id].enter_finish_area_frame = frame_idx
            if queue[track_id].enter_finish_area_frame and not queue[track_id].finish_queueing:
                inside_finish_area_time = (frame_idx - queue[track_id].enter_finish_area_frame) / fps
                if inside_finish_area_time > finish_thres:
                    queue[track_id].finish_queueing = True
            if track_id not in in_queueing_area and (frame_idx - queue[track_id].last_frame) / fps > dequeue_thres:
                '''
                if queue[track_id].finish_queueing:
                    queueing_time = (queue[track_id].last_frame - queue[track_id].start_frame) / fps
                    queue_time[track_id] = queueing_time
                '''
                del queue_time[track_id]
                del queue[track_id]
        if save_video:
            video_writer.write(bgr_image)


    if save_video:
        video_writer.release()

    num_idx = len(queue_time)
    avg_queue_time = sum(queue_time.values()) / num_idx if num_idx > 0 else 0
    print(queue_time)
    print(avg_queue_time)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--output-dir', type=str, default='out', help='dir for ouput files')
    parser.add_argument('--finish-area', nargs='+', type=int, default=[866,650,1156,410,1358,504,990,776], help='finish area')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold')
    parser.add_argument('--line', nargs='+', type=int, default=[0, 300, 1000, 200], help='boundary crossing line')
    parser.add_argument('--queue-polygon', nargs='+', type=int, default=[717, 101, 1453, 515, 1107, 777, 416, 352], help='queue area')
    parser.add_argument('--enqueue-thres', type=int, default=5, help='not count person with large displacement between frames')
    parser.add_argument('--dequeue-thres', type=int, default=5, help='not count person with large displacement between frames')
    parser.add_argument('--finish-thres', type=int, default=1, help='not count person with large displacement between frames')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--debug-frames', type=int, default=0, help='debug mode, run till frame number x')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference, supported on CUDA only')
    parser.add_argument('--save-img', action='store_true', help='save detection output as images')
    parser.add_argument('--save-video', action='store_true', help='save detection output as an video')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)