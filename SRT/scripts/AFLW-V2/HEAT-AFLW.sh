# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash ./scripts/AFLW-V2/HEAT-AFLW.sh 0 HG/HRNet 1.5
set -e
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for gpu devices and the model and the sigma"
  exit 1
fi
gpus=$1
model=$2
det=default
sigma=$3
use_gray=1
batch_size=16

save_dir=./snapshots/AFLW/${model}-DET-256x256-S${sigma}-B${batch_size}
rm -rf ${save_dir}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/BASE-main.py \
    --train_lists ./cache_data/lists/AFLW/train.pth \
    --eval_ilists ./cache_data/lists/AFLW/test.pth \
                  ./cache_data/lists/AFLW/test.front.pth \
    --mean_point  ./cache_data/lists/AFLW/train-mean.pth \
    --boxindicator ${det} --normalizeL ${det} --use_gray ${use_gray} \
    --num_pts 19 --data_indicator face19-AFLW \
    --procedure heatmap \
    --model_config ./configs/AFLW-V2/${model}.config \
    --opt_config   ./configs/AFLW-V2/${model}.opt.config \
    --save_path    ${save_dir} \
    --pre_crop_expand 0.2 \
    --scale_prob 0.5 --scale_min 0.9 --scale_max 1.1 --color_disturb 0.4 \
    --offset_max 0.1 --rotate_max 30 \
    --height 256 --width 256 --sigma ${sigma} --batch_size ${batch_size} \
    --print_freq 1000 --eval_freq 10 --workers 10 \
    --heatmap_type gaussian


CUDA_VISIBLE_DEVICES=${gpus} python ./exps/ROBUST-eval.py \
    --eval_lists  ./cache_data/lists/AFLW/test.pth \
                  ./cache_data/lists/AFLW/test.front.pth \
                  ./cache_data/lists/AFLW/test-less-30-yaw.pth \
                  ./cache_data/lists/AFLW/test-more-60-yaw.pth \
    --mean_point  ./cache_data/lists/AFLW/train-mean.pth \
    --save_path   ${save_dir}/ \
    --init_model  ${save_dir}/last-info.pth \
    --robust_scale 0.2 --robust_offset 0.1 --robust_rotate 30 \
    --rand_seed 1111 \
    --print_freq 200 --workers 10
