# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash scripts/AFLW/PRO-HG-RMSP.sh 2 V1 V1 3 0
echo script name: $0
echo $# arguments
if [ "$#" -ne 5 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 5 parameters for gpu devices, the model version, and the optimization version, and the box, and the sigma, and the gray"
  exit 1
fi
gpus=$1
HGV=$2
OPTV=$3
det=GTL
sigma=$4
use_gray=$5
batch_size=12

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/BASE-main.py \
    --train_lists ./cache_data/lists/AFLW/train.pth \
    --eval_ilists ./cache_data/lists/AFLW/test.pth \
                  ./cache_data/lists/AFLW/test.front.pth \
    --mean_point  ./cache_data/lists/AFLW/train-mean.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 19 --data_indicator face19-AFLW \
    --procedure heatmap \
    --model_config ./configs/face/HG.${HGV}.config \
    --opt_config   ./configs/face/RMSP.${OPTV}.config \
    --save_path    ./snapshots/PRO-AFLW-HG-${HGV}-RMSP-${OPTV}-S${sigma}-120x96-${use_gray} \
    --pre_crop_expand 0.2 \
    --scale_prob 1.0 --scale_min 0.8 --scale_max 1.2 \
    --offset_max 0.2 --rotate_max 30 \
    --normalizeL GTL \
    --height 120 --width 96 --sigma ${sigma} --batch_size ${batch_size} \
    --print_freq 1000 --eval_freq 10 --workers 6 \
    --heatmap_type gaussian
