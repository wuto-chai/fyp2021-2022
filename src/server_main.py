import argparse
from pathlib import Path

import cv2
import datetime
from tqdm import tqdm
import torch

from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from utils.encoder import create_box_encoder
from utils.datasets import LoadImages
from utils import utils
from deep_sort import detection


def run(
    weights='yolov5l_best.pt',  # model.pt path(s)
    source='frames',  # file/dir/URL/glob, 0 for webcam
    imgsz=640,  # inference size (pixels)
    output_dir='out', 
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    conf_thres=0.5,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    half=False,  # use FP16 half-precision inference

):

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights) 
    stride = int(model.stride.max())  # model stride
    model.conf = conf_thres
    model.iou = iou_thres
    
    device = utils.select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    encoder = create_box_encoder('mars-small128.pb', batch_size=32)
    max_cosine_distance = 0.2
    nn_budget = None
    use_gpu = True
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
    p = Path(output_dir) / 'output.txt'
    with p.open('a') as f:
        for _, img, im0s, _, frame_idx in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            bgr_image = im0s
            
            results = model(img)
            results = utils.non_max_suppression(results)
            if(results.xywh[0].shape[0] == 0):
                continue
            xywh_boxes = results.xywh[0][:,:-2]
            tlwh_boxes = utils.xywh2tlwh(xywh_boxes)
            confidence = results.pred[0][:, -2]
            if use_gpu:
                tlwh_boxes = tlwh_boxes.cpu()
            features = encoder(bgr_image, tlwh_boxes)
            detections = [detection.Detection(bbox, confidence, 'person', feature) for bbox, confidence, feature in zip(tlwh_boxes, confidence, features)]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)


            ppl_count = 0   
            frame_res = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue


            '''
            bbox = track.to_tlbr()
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            cv2.rectangle(bgr_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(bgr_image, "ID: " + str(track.track_id), (int(center_x), int(center_y)), 0,
                                1e-3 * bgr_image.shape[0], (0, 255, 0), 1)
            '''


            frame_res.append(track.track_id)
            ppl_count += 1
            
            if ppl_count > 0:
                '''
                print(frame_idx)
                cv2_imshow(bgr_image)
                '''
                f.write(str(frame_idx))
                f.write(",")
                f.write(str(datetime.timedelta(seconds=frame_idx//25)))
                f.write(",")
                f.write(str(ppl_count))
                for trackid in frame_res:
                    f.write(",")
                    f.write(str(trackid))
                f.write("\n")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference, supported on CUDA only')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)