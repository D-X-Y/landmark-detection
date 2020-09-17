# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash scripts/300W/OK-HG-RMSP.sh 0 DET 3 1
echo script name: $0
echo $# arguments
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for gpu devices, the box, and the sigma, and the gray"
  exit 1
fi
gpus=$1
det=$2
sigma=$3
use_gray=$4
batch_size=8
save_dir=./snapshots/WELL-300W-HG-${det}-S${sigma}-120x96-${use_gray}

rm -rf ${save_dir}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/BASE-main.py \
    --train_lists ./cache_data/lists/300W/300w.train.pth \
    --eval_ilists ./cache_data/lists/300W/300w.test-common.pth \
                  ./cache_data/lists/300W/300w.test-challenge.pth \
                  ./cache_data/lists/300W/300w.test-full.pth \
    --eval_vlists ./cache_data/lists/300VW/300VW.test-3.pth \
    --mean_point  ./cache_data/lists/300W/300w.train-mean.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 68 --data_indicator face68-300W \
    --procedure heatmap \
    --model_config ./configs/face/WELL/HG.300W.config \
    --opt_config   ./configs/face/R128/RMSP.300W.config \
    --save_path    ${save_dir} \
    --pre_crop_expand 0.2 \
    --scale_prob 1.0 --scale_min 0.9 --scale_max 1.1 --color_disturb 0.4 \
    --offset_max 0.2 --rotate_max 20 \
    --robust_iter 2 \
    --height 120 --width 96 --sigma ${sigma} --batch_size ${batch_size} \
    --print_freq 100 --print_freq_eval 100 --eval_freq 5 --workers 8 \
    --heatmap_type gaussian

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/BASE-main.py \
    --eval_ilists ./cache_data/lists/300W/300w.test-common.pth \
                  ./cache_data/lists/300W/300w.test-challenge.pth \
                  ./cache_data/lists/300W/300w.test-full.pth \
    --mean_point  ./cache_data/lists/300W/300w.train-mean.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 68 --data_indicator face68_pupil-300W \
    --procedure heatmap \
    --model_config ./configs/face/WELL/HG.300W.config \
    --opt_config   ./configs/face/WELL/RMSP.300W.config \
    --init_model   ${save_dir} \
    --save_path    ${save_dir} \
    --eval_once 300W-pupil \
    --pre_crop_expand 0.2 \
    --scale_prob 1.0 --scale_min 0.9 --scale_max 1.1 \
    --offset_max 0.2 --rotate_max 20 \
    --height 120 --width 96 --sigma ${sigma} --batch_size ${batch_size} \
    --print_freq 100 --print_freq_eval 2000 --eval_freq 5 --workers 8 \
    --heatmap_type gaussian
