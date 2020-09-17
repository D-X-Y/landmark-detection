# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash scripts/X-Mugsy/REG-DET-VALID.sh 0 RD01
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for gpu devices, and the optimization schedular"
  exit 1
fi
gpus=$1
opt=$2
sigma=4
use_gray=1
batch_size=32

save_path=./snapshots/X-Mugsy/VALID-REG-DET.${opt}-Mugsy-96x96-${batch_size}
rm -rf ${save_path}
#    --opt_config   ./configs/Mugsy-V2/ADAM.REG-${opt}.config \

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/BASE-main.py \
    --train_lists ./cache_data/lists/Mugsy/split-xtrain-pure-annotated-image.pth \
    --eval_ilists ./cache_data/lists/Mugsy/split-xtrain-pure-annotated-image.pth \
                  ./cache_data/lists/Mugsy/split-xvalid-pure-annotated-image.pth \
    --mean_point  ./cache_data/lists/Mugsy/Mugsy-train-mean.pth \
    --boxindicator default --use_gray ${use_gray} \
    --num_pts 18 --data_indicator Mugsy-18 \
    --model_config ./configs/Mugsy/REG-detector.config \
    --opt_config   ./configs/300W-V2/ADAM.REG.${opt}.config \
    --save_path    ${save_path} \
    --procedure regression --cutout_length -1 \
    --pre_crop_expand 0 \
    --scale_prob 0.5 --scale_min 0.9 --scale_max 1.1 --color_disturb 0.4 \
    --offset_max 0.2 --rotate_max 30 --rotate_prob 0.9 \
    --height 96 --width 96 --sigma ${sigma} --batch_size ${batch_size} \
    --print_freq 100 --print_freq_eval 200 --eval_freq 10 --workers 10 \
    --heatmap_type gaussian 

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/ROBUST-eval.py \
    --eval_lists  ./cache_data/lists/Mugsy/split-xtrain-pure-annotated-image.pth \
                  ./cache_data/lists/Mugsy/split-xvalid-pure-annotated-image.pth \
    --mean_point  ./cache_data/lists/Mugsy/Mugsy-train-mean.pth \
    --save_path   ${save_path} \
    --init_model  ${save_path}/last-info.pth \
    --robust_scale 0.2 --robust_offset 0.1 --robust_rotate 30 \
    --rand_seed 1111 \
    --print_freq 200 --workers 10
