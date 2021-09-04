# This is our FYP homepage

# Dataset
1. `!gdown --id 1LyFDze-n8BeUe1DJsCTL42c69CsJPqAp` (500 iamges in the train set, 100 in the test set)

# Pretrained model
1. mars-small128.pb: `!gdown --id 1oYaVZQ3Cvp8kcMLvpthKMgfF8P30E0Hr` (for deep sort)
2. yolov5l_best2.pt: `!gdown --id 1zQcGG3jRixytrpPtlDhK71fj0QdMj-Kx` (yolov5l model trained on overlooking viewport, detects person only)
3. yolo_dead_detection.pt: `!gdown --id 1UULc2Wphdo5A-J-AIQvREoP119t1cj77` (for queue detection, detect heads only)

# Sample usage
1. `!python -Wi src/queue_detect.py --source /content/fyp2021-2022/VIP_queue_video_1.mp4 --save-video --device="0" --debug-frames=$((30*60)) --queue-polygon 211 606 976 319 1034 651 282 893`
2. `!python -Wi src/server_main.py --weights yolov5l_best.pt --source /content/fyp2021-2022/2_20210723_20210723130119_20210723221501_131036.mp4 --save-img --device="0"`