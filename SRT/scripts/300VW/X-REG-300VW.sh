# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash scripts/300VW/X-REG-300VW.sh 0 W01
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for gpu devices, and the optimizer"
  exit 1
fi
gpus=$1
opt=L1
det=default
use_gray=1
sbr_weight=$2
i_batch_size=32
v_batch_size=16

init_save_path=./snapshots/300W/REG-DET-300W-RD02-${det}-96x96-32
save_path=./snapshots/300VW/X-SBR-DET-300W-${opt}-${det}-96x96-${i_batch_size}.${v_batch_size}-P49
rm -rf ${save_path}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/X-SBR-main.py \
    --train_lists ./cache_data/lists/300W/300w.train.pth \
                  ./cache_data/lists/300VW/300VW.train-no-points.pth \
    --eval_ilists ./cache_data/lists/300W/300w.test-common.pth \
                  ./cache_data/lists/300W/300w.test-challenge.pth \
                  ./cache_data/lists/300W/300w.test-full.pth \
    --mean_point  ./cache_data/lists/300W/300w.train-mean.pth \
    --shared_img_cache cache_data/lists/300W/path2tensor-G${use_gray}.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 68 --data_indicator face49-300W --x68to49 \
    --model_config ./configs/face/REG/REG.300W.config \
    --opt_config   ./configs/face/REG/ADAM.${opt}.300WVW-REG.config \
    --sbr_config   ./configs/face/REG/SBR.REG.300VW.${sbr_weight}.config \
    --init_model   ${init_save_path}/last-info.pth \
    --save_path    ${save_path} \
    --procedure regression --cutout_length -1 \
    --pre_crop_expand 0.2 \
    --scale_prob 0.5 --scale_min 0.9 --scale_max 1.1 --color_disturb 0.4 \
    --offset_max 0.2 --rotate_max 30 --rotate_prob 0.9 \
    --height 96 --width 96 --sigma 3 \
    --i_batch_size ${i_batch_size} --v_batch_size ${v_batch_size} \
    --print_freq 50 --print_freq_eval 1000 --eval_freq 5 --workers 40 \
    --heatmap_type gaussian 
