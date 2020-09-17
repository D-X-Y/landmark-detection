# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash ./scripts/300W/SBR/OK-SBR-REG-DEMO.sh 3
echo script name: $0
echo $# arguments
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for gpu devices"
  exit 1
fi
gpus=$1
det=default
sigma=3
use_gray=1
i_batch_size=8
v_batch_size=2

#   --init_model   ./snapshots/PRO-300W-${det}-HG-${HGV}-ADAM-${OPTV}-S${sigma}-112x112-${use_gray}/last-info.pth \
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
    --procedure regression \
    --model_config ./configs/face/WELL/REG.300W.config \
    --opt_config   ./configs/face/WELL/ADAM.300W-REG.config \
    --sbr_config   ./configs/face/SBR.DEMO.config \
    --save_path    ./snapshots/SBR-DEMO-${det}-REG-112x112-${use_gray} \
    --pre_crop_expand 0.2 \
    --scale_prob 1.0 --scale_min 0.9 --scale_max 1.1 \
    --offset_max 0.2 --rotate_max 20 \
    --height 112 --width 112 --sigma ${sigma} \
    --i_batch_size ${i_batch_size} --v_batch_size ${v_batch_size} \
    --print_freq 30 --eval_freq 2 --workers 12 \
    --heatmap_type gaussian
