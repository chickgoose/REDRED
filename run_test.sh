#!/bin/bash
# skip=1: 모든 프레임 (정확도 최대, RTF 높음)
# skip=2: 2프레임마다 1번 추론 (RTF 절반)
SKIP=${1:-1}
OUT=~/REDRED/output/submission_skip${SKIP}.csv

PYTHONPATH=~/yolov7 python ~/REDRED/pipeline/run_pipeline.py \
    --videos \
        ~/Dataset/4.TestVideo_Sample/cam0/Sample_1.mp4 \
        ~/Dataset/4.TestVideo_Sample/cam1/Sample_1.mp4 \
        ~/Dataset/4.TestVideo_Sample/cam2/Sample_1.mp4 \
        ~/Dataset/4.TestVideo_Sample/cam3/Sample_1.mp4 \
        ~/Dataset/4.TestVideo_Sample/cam4/Sample_1.mp4 \
    --weights ~/Dataset/yolov7_custom.pt \
    --names  ~/Dataset/names.txt \
    --prices ~/REDRED/data/prices_template.csv \
    --out    $OUT \
    --device 0 \
    --skip   $SKIP
