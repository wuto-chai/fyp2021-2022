import argparse

from pathlib import Path
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
    weights='../model/yolov5l_best.pt',  # model.pt path(s)
    source='frames',  # file/dir/URL/glob, 0 for webcam
    output_dir='out', 
    device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    conf_thres=0.5,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    line=[0, 300, 1000, 200], # boundary crossing line
    debug_frames=0, # debug mode
    half=False,  # use FP16 half-precision inference
    save_img=False,
    save_video=False,
):

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
    
    dataset = LoadImages(source, img_size=640, stride=stride)

    dir_path = Path(output_dir)
    file_path = Path('output.txt')
    dir_path.mkdir(exist_ok=True)
    p = Path(output_dir) / file_path
    for _, img, im0s, _, frame_idx in dataset:
        height, width, _ = im0s.shape
        break
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(str(dir_path / Path("out.mp4")), fourcc, 25, size)
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
        results = utils.non_max_suppression(results)
        if(results[0].shape[0]>0):
            video_writer.write(bgr_image)
    video_writer.release()
            

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../model/yolov5l_best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--output-dir', type=str, default='out', help='dir for ouput files')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--line', nargs='+', type=int, default=[0, 300, 1000, 200], help='boundary crossing line')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--debug-frames', type=int, default=0, help='debug mode, run till frame number x')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference, supported on CUDA only')
    parser.add_argument('--save-img', action='store_true', help='save detection output as image')
    parser.add_argument('--save-video', action='store_true', help='save detection output as an video')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)