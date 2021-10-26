import argparse

from pathlib import Path
import pickle
import numpy as np
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

    dataset = LoadImages(source, img_size=640, stride=32)

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
        for detection in detections:
            bbox = detection.tlwh
            bbox[2:] = bbox[:2] + bbox[2:]

            cv2.rectangle(bgr_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        video_writer.write(bgr_image)


    video_writer.release()

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