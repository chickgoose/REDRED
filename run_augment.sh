#!/bin/bash
# Cut & Paste 증강 실행 스크립트
# Usage: bash run_augment.sh [num_images]
NUM=${1:-5000}

python ~/REDRED/augment/cut_paste_aug.py \
    --seg_dir  ~/Dataset/3.background_substracted_white \
    --bg_dir   ~/Dataset/3.Background_Images \
    --out_dir  ~/Dataset/augmented \
    --num_images $NUM \
    --max_objects 4 \
    --no_erasing

echo "Done: $NUM augmented images -> ~/Dataset/augmented"
