# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash scripts/AFLW/OK-REG.sh 0
echo script name: $0
echo $# arguments
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for gpu devices" 
  exit 1
fi
gpus=$1
det=default
use_gray=1
batch_size=64

save_path=./snapshots/WELL-AFLW-${det}-REG-112x112-G${use_gray}
rm -rf ${save_path}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/BASE-main.py \
    --train_lists ./cache_data/lists/AFLW/train.pth \
    --eval_ilists ./cache_data/lists/AFLW/test.pth \
                  ./cache_data/lists/AFLW/test.front.pth \
                  ./cache_data/lists/AFLW/test-less-30-yaw.pth \
                  ./cache_data/lists/AFLW/test-more-60-yaw.pth \
    --mean_point  ./cache_data/lists/AFLW/train-mean.pth \
    --boxindicator ${det} --normalizeL ${det} --use_gray ${use_gray} \
    --num_pts 19 --data_indicator face19-AFLW \
    --model_config ./configs/face/REG/REG.AFLW.config \
    --opt_config   ./configs/face/REG/ADAM.L1.AFLW-REG.config \
    --save_path    ${save_path} \
    --procedure regression --cutout_length -1 \
    --pre_crop_expand 0.2 \
    --scale_prob 1.0 --scale_min 0.85 --scale_max 1.15 \
    --offset_max 0.2 --rotate_max 30 \
    --height 112 --width 112 --sigma 3 --batch_size ${batch_size} \
    --print_freq 50 --print_freq_eval 100 --eval_freq 2 --workers 8 \
    --heatmap_type gaussian 
