# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash scripts/300W/OK-CPM.sh 0 DET 4
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for gpu devices, and the box, and the sigma"
  exit 1
fi
gpus=$1
det=$2
sigma=$3
use_gray=0
batch_size=8

save_path=./snapshots/WELL-300W-${det}-CPM-S${sigma}-96x96
rm -rf ${save_path}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/BASE-main.py \
    --train_lists ./cache_data/lists/300W/300w.train.pth \
    --eval_ilists ./cache_data/lists/300W/300w.test-common.pth \
                  ./cache_data/lists/300W/300w.test-challenge.pth \
                  ./cache_data/lists/300W/300w.test-full.pth \
    --mean_point  ./cache_data/lists/300W/300w.train-mean.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 68 --data_indicator face68-300W \
    --model_config ./configs/face/WELL/CPM.300W.config \
    --opt_config   ./configs/face/WELL/SGD.300W.config \
    --save_path    ${save_path} \
    --procedure heatmap \
    --pre_crop_expand 0.2 \
    --scale_prob 1.0 --scale_min 0.9 --scale_max 1.1 \
    --offset_max 0.3 --rotate_max 30 \
    --robust_iter 2 \
    --height 96 --width 96 --sigma ${sigma} --batch_size ${batch_size} \
    --print_freq 150 --print_freq_eval 100 --eval_freq 1 --workers 6 \
    --heatmap_type gaussian 
