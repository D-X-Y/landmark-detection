# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash ./scripts/MPII/DET-RMSP.sh 0 MSPN 3
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for gpu devices, the detecotr, and the sigma"
  exit 1
fi
gpus=$1
detector=$2
det=default
sigma=$3
if [ $detector == 'CPM' ] ;then
  use_gray=0
  opt_config=SGD.MPII.config
else
  use_gray=1
  opt_config=RMSP.MPII.config
fi
batch_size=16

save_path=./snapshots/MPII/BASE-${detector}-S${sigma}-B${batch_size}-${use_gray}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/BASE-main.py \
    --train_lists ./cache_data/lists/MPII/train.pth \
    --eval_ilists ./cache_data/lists/MPII/valid.pth \
    --mean_point  ./cache_data/lists/MPII/MPII-trainval-mean.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 16 --data_indicator pose16-MPII \
    --normalizeL head --procedure heatmap \
    --model_config ./configs/pose/${detector}.MPII.config \
    --opt_config   ./configs/pose/${opt_config} \
    --save_path    ${save_path} \
    --pre_crop_expand 0.2 \
    --scale_prob 1.0 --scale_min 0.8 --scale_max 1.2 \
    --offset_max 0.2 --rotate_max 30 \
    --robust_iter 0 \
    --height 288 --width 272 --sigma ${sigma} --batch_size ${batch_size} \
    --print_freq 300 --eval_freq 10 --workers 10 \
    --heatmap_type gaussian
