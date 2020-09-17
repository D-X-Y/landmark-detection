# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash ./scripts/AFLW/HEAT-AFLW-V2.sh 0 default ADAM 2
# bash ./scripts/AFLW/HEAT-AFLW-V2.sh 0     GTB ADAM 2
echo script name: $0
echo $# arguments
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for gpu devices, and the face-box, and the optimizer and the sigma"
  exit 1
fi
gpus=$1
det=$2
opt=$3
sigma=$4
use_gray=1
batch_size=8

save_dir=./snapshots/AFLW/HG-DET-${det}-${opt}-128x128-S${sigma}-B${batch_size}
rm -rf ${save_dir}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/BASE-main.py \
    --train_lists ./cache_data/lists/AFLW/train.pth \
    --eval_ilists ./cache_data/lists/AFLW/test.pth \
                  ./cache_data/lists/AFLW/test.front.pth \
    --mean_point  ./cache_data/lists/AFLW/train-mean.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 19 --data_indicator face19-AFLW \
    --procedure heatmap \
    --model_config ./configs/AFLW/HG.config \
    --opt_config   ./configs/AFLW/${opt}.config \
    --save_path    ${save_dir} \
    --pre_crop_expand 0.2 \
    --scale_prob 1 --scale_min 1 --scale_max 1 \
    --offset_max 0.2 --rotate_max 20 --rotate_prob 0.5 \
    --normalizeL default \
    --height 128 --width 128 --sigma ${sigma} --batch_size ${batch_size} \
    --print_freq 300 --print_freq_eval 1500 --eval_freq 10 --workers 8 \
    --heatmap_type gaussian
