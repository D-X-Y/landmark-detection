# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash ./scripts/300W-V2/DET-HRNET-68.sh 0 RD01
set -e
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for gpu devices and the optimizer"
  exit 1
fi
gpus=$1
opt=$2
det=default
sigma=3
use_gray=1
batch_size=16
save_dir=./snapshots/300W/HRNET-DET-300W-${opt}-${det}-256x256-S${sigma}-L68

rm -rf ${save_dir}

# --eval_vlists ./cache_data/lists/300VW/300VW.test-1.pth \
CUDA_VISIBLE_DEVICES=${gpus} python ./exps/BASE-main.py \
    --train_lists ./cache_data/lists/300W/300w.train.pth \
    --eval_ilists ./cache_data/lists/300W/300w.test-common.pth \
                  ./cache_data/lists/300W/300w.test-challenge.pth \
                  ./cache_data/lists/300W/300w.test-full.pth \
    --mean_point  ./cache_data/lists/300W/300w.train-mean.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 68 --data_indicator face68-300W \
    --procedure heatmap \
    --model_config ./configs/300W/HRNet.300W.config \
    --opt_config   ./configs/300W-V2/ADAM.HRNet-${opt}.config \
    --save_path    ${save_dir} \
    --pre_crop_expand 0.2 \
    --scale_prob 0.5 --scale_min 0.9 --scale_max 1.1 --color_disturb 0.4 \
    --offset_max 0.1 --rotate_max 30 --rotate_prob 0.5 \
    --height 256 --width 256 --sigma ${sigma} --batch_size ${batch_size} \
    --print_freq 100 --print_freq_eval 600 --eval_freq 10 --workers 8 \
    --heatmap_type gaussian


CUDA_VISIBLE_DEVICES=${gpus} python ./exps/ROBUST-eval.py \
    --eval_lists  ./cache_data/lists/300W/300w.train.pth \
                  ./cache_data/lists/300W/300w.test-common.pth \
                  ./cache_data/lists/300W/300w.test-challenge.pth \
                  ./cache_data/lists/300W/300w.test-full.pth \
    --mean_point  ./cache_data/lists/300W/300w.train-mean.pth \
    --save_path   ${save_dir} \
    --init_model  ${save_dir}/last-info.pth \
    --robust_scale 0.2 --robust_offset 0.1 --robust_rotate 30 \
    --rand_seed 1111 \
    --print_freq 200 --workers 10
