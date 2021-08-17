import torch
weights='yolov5l_best2.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)