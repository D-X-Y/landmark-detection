# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash ./scripts/300W/HEAT/OK-SBT-HG-300VW-49.sh 0 L1 W01 16 8
echo script name: $0
echo $# arguments
if [ "$#" -ne 5 ] ;then
  echo "Input illegal number of parameters :" $#
  echo "Need 5 parameters for gpu devices, sbt-loss, weights, image-batch-size, and multi-view-batch-size"
  exit 1
fi
gpus=$1
opt=ADAM
det=default
sigma=3
use_gray=1
sbt_loss=$2
sbt_weight=$3
i_batch_size=$4
v_batch_size=0
m_batch_size=$5
point=L49

save_path=./snapshots/SBT-${sbt_loss}-HEAT-300W.PFaceS-HG-${opt}.${sbt_weight}-${det}-256x256-${i_batch_size}.${v_batch_size}.${m_batch_size}-${point}

rm -rf ${save_path}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/STM-main.py \
    --train_lists ./cache_data/lists/300W/300w.train.pth \
     		  ./cache_data/lists/Panoptic-FACE/simple-face-2000-nopts-novid.pth \
    --eval_ilists ./cache_data/lists/300W/300w.test-common.pth \
                  ./cache_data/lists/300W/300w.test-challenge.pth \
                  ./cache_data/lists/300W/300w.test-full.pth \
    --eval_vlists ./cache_data/lists/300VW/300VW.test-1.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 68 --data_indicator face49-300W-PFaceS --x68to49 \
    --procedure heatmap \
    --model_config ./configs/face/HEAT/HG.300W.config \
    --opt_config   ./configs/face/HEAT/${opt}.HG-300W-PFaceS.config \
    --stm_config   ./configs/face/HEAT/SBT-${sbt_loss}.300W-PFaceS.${sbt_weight}.config \
    --init_model   ./snapshots/HEAT-300W-HG-${opt}-${det}-256x256-S${sigma}-${point}/last-info.pth \
    --save_path ${save_path} \
    --pre_crop_expand 0.2 \
    --scale_prob 0.5 --scale_min 0.9 --scale_max 1.1 --color_disturb 0.4 \
    --offset_max 0.1 --rotate_max 30 --rotate_prob 0.5 \
    --height 256 --width 256 --sigma ${sigma} \
    --i_batch_size ${i_batch_size} --v_batch_size ${v_batch_size} --m_batch_size ${m_batch_size} \
    --print_freq 300 --print_freq_eval 2000 --eval_freq 2 --workers 16 \
    --heatmap_type gaussian
