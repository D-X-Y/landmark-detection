# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# sh scripts/300W/SBR-DEMO.sh 2 GTL 2
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for gpu devices, and the box, and the sigma"
  exit 1
fi
gpus=$1
HGV=V1
SBRV=V1
det=$2
sigma=$3
use_gray=1
i_batch_size=8
v_batch_size=2

#                  ./cache_data/lists/300VW/300VW.train.pth \
CUDA_VISIBLE_DEVICES=${gpus} python ./exps/SBR-main.py \
    --train_lists ./cache_data/lists/300W/300w.train.pth \
                  ./cache_data/lists/demo/demo-sbr.pth \
    --eval_ilists ./cache_data/lists/300W/300w.test-common.pth \
                  ./cache_data/lists/300W/300w.test-challenge.pth \
                  ./cache_data/lists/300W/300w.test-full.pth \
    --eval_vlists ./cache_data/lists/demo/demo-sbr.pth \
    --mean_point  ./cache_data/lists/300W/300w.train-mean.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 68 --data_indicator face68-DEMO \
    --model_config ./configs/face/HG.${HGV}.config \
    --opt_config   ./configs/face/SGD.sbr.config \
    --sbr_config   ./configs/face/SBR.${SBRV}.config \
    --save_path    ./snapshots/SBR-DEMO-${det}-HG-${HGV}-SGD.sbr-S${sigma}-120x96-${use_gray} \
    --init_model   ./snapshots/PRO-300W-${det}-HG-${HGV}-ADAM-V2-S${sigma}-120x96-${use_gray}/last-info.pth \
    --pre_crop_expand 0.2 \
    --scale_prob 1.0 --scale_min 0.9 --scale_max 1.1 \
    --offset_max 0.2 --rotate_max 20 \
    --robust_iter 2 \
    --height 120 --width 96 --sigma ${sigma} \
    --i_batch_size ${i_batch_size} --v_batch_size ${v_batch_size} \
    --print_freq 30 --eval_freq 2 --workers 12 \
    --heatmap_type gaussian
