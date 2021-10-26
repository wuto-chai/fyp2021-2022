import argparse

from pathlib import Path
from tqdm import tqdm
import torch
from models.experimental import attempt_load
from deep_sort import detection
from my_utils.encoder import create_box_encoder
from my_utils.my_dataset import LoadImages
from my_utils import utils
import pickle

fps = 60
identity_switch_thres = 30

def run(
    weights='yolov5l.pt',  # model.pt path(s)
    source='frames',  # file/dir/URL/glob, 0 for webcam
    output_dir='out', 
    device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    conf_thres=0.5,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    debug_frames=0, # debug mode
    imgsz=640, # inference size (pixels)
    half=False,  # use FP16 half-precision inference
):

    device = utils.select_device(device)
    use_gpu = device == torch.device('cuda:0')
    print(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if half:
        model.half()

    encoder = create_box_encoder(str(Path('model') / Path('mars-small128.pb')), batch_size=32)

    dataset = LoadImages(source, img_size=imgsz, stride=stride)


    dir_path = Path(output_dir)
    file_path = Path('output.pt')
    dir_path.mkdir(exist_ok=True)
    p = Path(output_dir) / file_path
    output_tensor_list = []
    if use_gpu:
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    for _, img, im0s, _, frame_idx in tqdm(dataset):
        if debug_frames > 0 and frame_idx > debug_frames:
            break
        

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        
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
        features = encoder(im0s, tlwh_boxes)
        
        detections = [detection.Detection(bbox, confidence, 'person', feature) for bbox, confidence, feature in zip(tlwh_boxes, confidence, features)]

        output_tensor_list.append(detections)
    with open('output_tensor.pickle', 'wb') as handle:
        pickle.dump(output_tensor_list, handle)
        

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--output-dir', type=str, default='out', help='dir for ouput files')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold')
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