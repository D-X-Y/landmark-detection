# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash ./scripts/300W-V2/draw-300WVW-49.sh 0
echo script name: $0
echo $# arguments
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters :" $#
  echo "Need 1 parameters for the gpu device and model-version"
  exit 1
fi
gpus=$1
VER=V2
opt=ADAM
det=default
sigma=3
use_gray=1
sbr_loss=HEAT
sbr_weight=W05
i_batch_size=0
v_batch_size=8
point=L49

save_path=./snapshots/DEBUG-DXY/HEAT-SBR-${sbr_loss}-300WVW-FINAL

rm -rf ${save_path}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/draw-300W.py \
    --train_lists ./cache_data/lists/300VW/300VW.train.pth \
    --eval_ilists ./cache_data/lists/300W/300w.train.pth \
    --mean_point  ./cache_data/lists/300W/300w.train-mean.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 68 --data_indicator face49-300W-300VW --x68to49 \
    --procedure heatmap \
    --model_config ./configs/300W/HG.300W.config \
    --opt_config   ./configs/300W-V2/${opt}.HG-300WVW.config \
    --sbr_config   ./configs/300W-V2/DEBUG-SBR-HEAT.config \
    --init_model   ./snapshots/300W/HEAT-SBR-HEAT-300WVW-ADAM.W05-default-256x256-16.8-L49/checkpoint/checkpoint-${VER}.pth \
    --save_path ${save_path} \
    --pre_crop_expand 0.2 \
    --scale_prob 0.5 --scale_min 0.9 --scale_max 1.1 --color_disturb 0.4 \
    --offset_max 0.1 --rotate_max 30 --rotate_prob 0.5 \
    --height 256 --width 256 --sigma ${sigma} \
    --i_batch_size ${i_batch_size} --v_batch_size ${v_batch_size} \
    --print_freq 300 --print_freq_eval 2000 --eval_freq 5 --workers 16 \
    --heatmap_type gaussian
