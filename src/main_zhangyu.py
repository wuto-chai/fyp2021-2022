import argparse

from pathlib import Path
import copy
import cv2
import datetime
from tqdm import tqdm
import torch

from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from my_utils.encoder import create_box_encoder
from models.experimental import attempt_load
from my_utils.my_dataset import LoadImages
from my_utils import utils
from deep_sort import detection


def run(
    weights='yolov5l_best.pt',  # model.pt path(s)
    source='frames',  # file/dir/URL/glob, 0 for webcam
    output_dir='out', 
    device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    conf_thres=0.5,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    line=[0, 300, 1000, 200], # boundary crossing line
    debug_frames=0, # debug mode
    half=False,  # use FP16 half-precision inference
):

    line = ((line[0], line[1]),(line[2], line[3]))

    device = utils.select_device(device)
    use_gpu = device == torch.device('cuda:0')
    print(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    model.conf = conf_thres
    model.iou = iou_thres
    if half:
        model.half()
    
    encoder = create_box_encoder('mars-small128.pb', batch_size=32)
    max_cosine_distance = 0.2
    nn_budget = None
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    dataset = LoadImages(source, img_size=640, stride=stride)


    dir_path = Path(output_dir)
    file_path = Path('output.txt')
    dir_path.mkdir(exist_ok=True)
    p = Path(output_dir) / file_path
    lastFrameRes = []
    thisFrameRes = []
    with p.open('w') as f:
        f.write("start_frame,start_time,end_frame,end_time,num,idx\n")
        for _, img, im0s, _, frame_idx in tqdm(dataset):
            if frame_idx < 2900:
              continue
            if debug_frames > 0 and frame_idx > debug_frames:
                break
            
            indexIDs = []
            boxes = []
            ppl_count = 0

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            bgr_image = im0s

            
            results = model(img)[0]
            results = utils.non_max_suppression(results)
            if results[0].shape[0] > 0:
                det = results[0]
                det[:, :4] = utils.scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                person_ind = [i for i, cls in enumerate(det[:, -1]) if int(cls) == 0]   
                xyxy = det[person_ind, :-2]  # find person only
                # xyxy = det[:,:-2]
                xywh_boxes = utils.xyxy2xywh(xyxy)
                tlwh_boxes = utils.xywh2tlwh(xywh_boxes)
                confidence = det[:, -2]
                if use_gpu:
                    tlwh_boxes = tlwh_boxes.cpu()
                features = encoder(bgr_image, tlwh_boxes)
                
                detections = [detection.Detection(bbox, confidence, 'person', feature) for bbox, confidence, feature in zip(tlwh_boxes, confidence, features)]

                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    bbox = track.to_tlbr()
                    boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                    indexIDs.append(track.track_id) # # this frame we have these ppl idxs
                    ppl_count += 1


            thisFrameRes = [frame_idx, ppl_count, indexIDs]
            if not lastFrameRes:
                lastFrameRes = copy.deepcopy(thisFrameRes)
            if thisFrameRes[1] != lastFrameRes[1] or thisFrameRes[2] != lastFrameRes[2]:
                if lastFrameRes[1] > 0:
                    f.write(str(lastFrameRes[0]))
                    f.write(",")
                    f.write(str(datetime.timedelta(seconds=lastFrameRes[0]//25)))
                    f.write(",")
                    f.write(str(frame_idx))
                    f.write(",")
                    f.write(str(datetime.timedelta(seconds=frame_idx//25)))
                    f.write(",")

                    f.write(str(lastFrameRes[1]))
                    f.write(",")
                    for trackid in lastFrameRes[2]:
                        f.write(str(trackid))
                        f.write(" ")
                    f.write("\n")
                lastFrameRes = copy.deepcopy(thisFrameRes)
                
                

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--output-dir', type=str, default='out', help='dir for ouput files')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--line', nargs='+', type=int, default=[0, 300, 1000, 200], help='boundary crossing line')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--debug-frames', type=int, default=0, help='debug mode, run till frame number x')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference, supported on CUDA only')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)