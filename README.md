# This is our FYP homepage

# Dataset
1. `!gdown --id 1LyFDze-n8BeUe1DJsCTL42c69CsJPqAp` (500 iamges in the train set, 100 in the test set)

# Sample usage
1. `!python -Wi src/queue_detect.py --source /content/fyp2021-2022/VIP_queue_video_1.mp4 --save-video --device="0" --debug-frames=$((30*60)) --queue-polygon 211 606 976 319 1034 651 282 893`
2. `!python -Wi src/server_main.py --weights yolov5l_best.pt --source /content/fyp2021-2022/2_20210723_20210723130119_20210723221501_131036.mp4 --save-img --device="0"`