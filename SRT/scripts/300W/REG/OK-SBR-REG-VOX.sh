# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash ./scripts/300W/REG/OK-SBR-REG-VOX.sh 0
echo script name: $0
echo $# arguments
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for gpu devices"
  exit 1
fi
gpus=$1
opt=L1
det=default
sigma=3
use_gray=1
i_batch_size=32
v_batch_size=32

save_path=./snapshots/SBR-DEMO-300W.VOX-${det}-REG-96x96-${i_batch_size}.${v_batch_size}

rm -rf ${save_path}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/SBR-main.py \
    --train_lists ./cache_data/lists/300W/300w.train.pth \
     		  ./cache_data/lists/VoxCeleb2/Unlabeled-VoxCeleb2-Videos.pth \
    --eval_ilists ./cache_data/lists/300W/300w.test-common.pth \
                  ./cache_data/lists/300W/300w.test-challenge.pth \
                  ./cache_data/lists/300W/300w.test-full.pth \
    --eval_vlists ./cache_data/lists/300VW/300VW.test-1.pth \
    --mean_point  ./cache_data/lists/300W/300w.train-mean.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 68 --data_indicator face49-300W-300VW --x68to49 \
    --procedure regression \
    --model_config ./configs/face/REG/REG.300W.config \
    --opt_config   ./configs/face/REG/ADAM.L1.300W-REG-VOX.config \
    --sbr_config   ./configs/face/REG/SBR.REG.300W-VOX.config \
    --init_model   ./snapshots/REG-DEMO-300W-${opt}-${det}-96x96-32/last-info.pth \
    --skip_first_eval \
    --save_path    ${save_path} \
    --pre_crop_expand 0.2 \
    --scale_prob 0.5 --scale_min 0.9 --scale_max 1.1 --color_disturb 0.4 \
    --offset_max 0.2 --rotate_max 30 --rotate_prob 0.9 \
    --height 96 --width 96 --sigma ${sigma} \
    --i_batch_size ${i_batch_size} --v_batch_size ${v_batch_size} \
    --print_freq 300 --print_freq_eval 2000 --eval_freq 20 --workers 40 \
    --heatmap_type gaussian
