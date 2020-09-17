# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash scripts/AFLW/REG-AFLW.sh 0 32
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for gpu devices, and the batch size"
  exit 1
fi
gpus=$1
opt=L1
det=GTL
use_gray=1
batch_size=$2

save_path=./snapshots/REG-AFLW-${opt}-96x96-${batch_size}
rm -rf ${save_path}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/BASE-main.py \
    --train_lists ./cache_data/lists/AFLW/train.pth \
    --eval_ilists ./cache_data/lists/AFLW/test.front.pth \
                  ./cache_data/lists/AFLW/test.pth \
    --mean_point  ./cache_data/lists/AFLW/train-mean.pth \
    --boxindicator ${det} --normalizeL ${det} --use_gray ${use_gray} \
    --num_pts 19 --data_indicator face19-AFLW \
    --model_config ./configs/face/REG/REG.AFLW.config \
    --opt_config   ./configs/face/REG/ADAM.${opt}.AFLW-REG.config \
    --save_path    ${save_path} \
    --procedure regression --cutout_length -1 \
    --pre_crop_expand 0.2 \
    --scale_prob 0.5 --scale_min 0.9 --scale_max 1.1 --color_disturb 0.4 \
    --offset_max 0.2 --rotate_max 30 --rotate_prob 0.9 \
    --height 96 --width 96 --sigma 3 --batch_size ${batch_size} \
    --print_freq 400 --print_freq_eval 1000 --eval_freq 5 --workers 20 \
    --heatmap_type gaussian 
