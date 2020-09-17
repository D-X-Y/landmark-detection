# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash scripts/WFLW/SBR-HEAT-300VW.sh 0 HG
set -e
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for gpu devices and the model"
  exit 1
fi
gpus=$1
model=$2
sigma=3
det=default
use_gray=1
i_batch_size=16
v_batch_size=8

save_path=./snapshots/WFLW/${model}-SBR-WFLW-300VW-${det}-256x256-${i_batch_size}.${v_batch_size}-S${sigma}
rm -rf ${save_path}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/SBR-main.py \
    --train_lists ./cache_data/lists/WFLW/train.pth \
                  ./cache_data/lists/300VW/300VW.train-no-points.pth \
    --eval_ilists ./cache_data/lists/WFLW/test.pth \
    --mean_point  ./cache_data/lists/WFLW/train-mean.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 98 --data_indicator face98-WFLW \
    --model_config ./configs/WFLW/${model}.config \
    --opt_config   ./configs/WFLW/ADAM.${model}-WFLW.config \
    --sbr_config   ./configs/face/HEAT/SBR.HEAT.WFLW-300VW.W10.config \
    --init_model   ./snapshots/WFLW/${model}-DET-WFLW-${det}-256x256-S${sigma}/last-info.pth \
    --save_path    ${save_path} \
    --procedure heatmap --cutout_length -1 \
    --pre_crop_expand 0.2 \
    --scale_prob 0.5 --scale_min 0.9 --scale_max 1.1 --color_disturb 0.4 \
    --offset_max 0.2 --rotate_max 30 --rotate_prob 0.5 \
    --height 256 --width 256 --sigma ${sigma} \
    --i_batch_size ${i_batch_size} --v_batch_size ${v_batch_size} \
    --print_freq 50 --print_freq_eval 1000 --eval_freq 5 --workers 18 \
    --heatmap_type gaussian 

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/ROBUST-eval.py \
    --eval_lists  ./cache_data/lists/WFLW/train.pth \
                  ./cache_data/lists/WFLW/test.pth \
    --mean_point  ./cache_data/lists/WFLW/train-mean.pth \
    --save_path   ${save_path} \
    --init_model  ${save_path}/last-info.pth \
    --robust_scale 0.2 --robust_offset 0.1 --robust_rotate 30 \
    --rand_seed 1111 \
    --print_freq 200 --workers 10
