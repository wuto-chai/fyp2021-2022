import argparse

from pathlib import Path
import pickle
import cv2
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.geometry import Point

from deep_sort.tracker import Tracker

from deep_sort import nn_matching
from my_utils.my_dataset import LoadImages
from my_utils.queuer import Queuer, PotentialQueuer


fps = 60
identity_switch_thres = 30

def run(
    source='frames',  # file/dir/URL/glob, 0 for webcam
    output_dir='out', 
    finish_area=[866,650,1172,473,1281,555,990,776],
    # line=[584, 823, 1202, 435], # finish line, we assume on the left hand side of the line (point1 -> point2) is the queueing area
    queue_polygon=[717, 101, 1453, 515, 1107, 777, 416, 352],   # x y x y x y x y x y
    enqueue_thres=10, 
    dequeue_thres=10,
    finish_thres=1,
    debug_frames=0, # debug mode
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
    
    max_cosine_distance = 0.2
    nn_budget = None
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    dataset = LoadImages(source, img_size=640, stride=32)

    avg_queue_time = 0.0
    queue = {}
    potential_queue = {}
    queue_time = {}
    with open(str(Path('data') / Path('output_detection.pickle')), 'rb') as handle:
        detection_list = pickle.load(handle)
    
    dir_path = Path(output_dir)
    dir_path.mkdir(exist_ok=True)
    if save_video:
        for _, _, im0s, _, frame_idx in dataset:
            height, width, _ = im0s.shape
            break
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_writer = cv2.VideoWriter(str(dir_path / Path("out.mp4")), fourcc, fps, size)
    for _, _, im0s, _, frame_idx in tqdm(dataset):
        if debug_frames > 0 and frame_idx > debug_frames:
            break
        


        bgr_image = im0s

        detections = detection_list[frame_idx]
        

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        in_queueing_area = []   
        cv2.putText(bgr_image, "Queue length: {}".format(str(list(queue.keys()))[1:-1]), (100, 50),          
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (49, 49, 247), 2)
        cv2.putText(bgr_image, "ID: {}".format(str(list(queue.keys()))[1:-1]), (100, 100),          
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (49, 49, 247), 2)
        num_idx = len(queue_time)
        avg_queue_time = sum(queue_time.values()) / num_idx if num_idx > 0 else 0
        cv2.putText(bgr_image, "Avg time: {}".format(str(round(avg_queue_time, 1))), (100, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (49, 49, 247), 2)
        cv2.putText(bgr_image, "ID, time elapsed: {}".format(str((queue_time))[1:-1]), (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (49, 49, 247), 2)
            
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id = track.track_id
            bbox = track.to_tlbr()
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
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

                elif track_id in queue:
                    if not queue[track_id].enter_finish_area_frame:
                        queue[track_id].last_frame = frame_idx
                    queue[track_id].position = (center_x, center_y)
                    in_queueing_area.append(track_id)

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
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--output-dir', type=str, default='out', help='dir for ouput files')
    parser.add_argument('--finish-area', nargs='+', type=int, default=[866,650,1156,410,1358,504,990,776], help='finish area')
    parser.add_argument('--queue-polygon', nargs='+', type=int, default=[717, 101, 1453, 515, 1107, 777, 416, 352], help='queue area')
    parser.add_argument('--enqueue-thres', type=int, default=5, help='not count person with large displacement between frames')
    parser.add_argument('--dequeue-thres', type=int, default=5, help='not count person with large displacement between frames')
    parser.add_argument('--finish-thres', type=int, default=1, help='not count person with large displacement between frames')
    parser.add_argument('--debug-frames', type=int, default=0, help='debug mode, run till frame number x')
    parser.add_argument('--save-img', action='store_true', help='save detection output as images')
    parser.add_argument('--save-video', action='store_true', help='save detection output as an video')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)