# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# sh scripts/300W/SBR_HG_RMSP.sh 2 V1 V1 V1 DET 3
echo script name: $0
echo $# arguments
if [ "$#" -ne 6 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 6 parameters for gpu devices, the model version, and the optimization version, and the sbr version, and the box, and the sigma"
  exit 1
fi
gpus=$1
HGV=$2
OPTV=$3
SBRV=$4
det=$5
sigma=$6
use_gray=0
i_batch_size=8
v_batch_size=8

#                  ./cache_data/lists/300VW/300VW.train.pth \
CUDA_VISIBLE_DEVICES=${gpus} python ./exps/SBR-main.py \
    --train_lists ./cache_data/lists/300W/300w.train.pth \
                  ./cache_data/lists/300VW/300VW.test-1.pth \
    --eval_ilists ./cache_data/lists/300W/300w.test-common.pth \
                  ./cache_data/lists/300W/300w.test-challenge.pth \
                  ./cache_data/lists/300W/300w.test-full.pth \
    --mean_point  ./cache_data/lists/300W/300w.train-mean.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 68 --data_indicator face68-300VW \
    --model_config ./configs/face/HG.${HGV}.config \
    --opt_config   ./configs/face/RMSP.${OPTV}.config \
    --sbr_config   ./configs/face/SBR.${SBRV}.config \
    --save_path    ./snapshots/SBR-300W-${det}-HG-${HGV}-RMSP-${OPTV}-S${sigma}-120x96-${use_gray} \
    --init_model   ./snapshots/PRO-300W-${det}-HG-${HGV}-RMSP-${OPTV}-S${sigma}-120x96-${use_gray}/last-info.pth \
    --pre_crop_expand 0.2 \
    --scale_prob 1.0 --scale_min 0.9 --scale_max 1.1 \
    --offset_max 0.2 --rotate_max 20 \
    --robust_iter 2 \
    --height 120 --width 96 --sigma ${sigma} \
    --i_batch_size ${i_batch_size} --v_batch_size ${v_batch_size} \
    --print_freq 150 --eval_freq 10 --workers 12 \
    --heatmap_type gaussian
