import argparse
from models.experimental import attempt_load
import torch

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

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride


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