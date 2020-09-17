# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash scripts/AFLW-V2/REG-SRT-PFACE.sh 0
set -e
echo script name: $0
echo $# arguments
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for gpu devices, and the batch size"
  exit 1
fi
gpus=$1
det=default
use_gray=1
i_batch_size=32
v_batch_size=32
m_batch_size=16

save_path=./snapshots/AFLW/SRT-AFLW.PFACE-96x96-${i_batch_size}.${v_batch_size}.${m_batch_size}
rm -rf ${save_path}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/STM-main.py \
    --train_lists ./cache_data/lists/AFLW/train.pth \
    		  ./cache_data/lists/Panoptic-FACE/all-face-2000-nopts.pth \
    --eval_ilists ./cache_data/lists/AFLW/test.front.pth \
                  ./cache_data/lists/AFLW/test.pth \
		  ./cache_data/lists/AFLW/test-less-30-yaw.pth \
		  ./cache_data/lists/AFLW/test-more-60-yaw.pth \
    --boxindicator ${det} --normalizeL ${det} --use_gray ${use_gray} \
    --num_pts 19 --data_indicator face19-AFLW \
    --model_config ./configs/face/REG/REG.AFLW.config \
    --opt_config   ./configs/AFLW-V2/ADAM.REG.PFACE.config \
    --stm_config   ./configs/face/REG/STM.REG.AFLW-PFace.W10.W10.config \
    --init_model   ./snapshots/AFLW/REG-AFLW-RD02-96x96-32/last-info.pth \
    --save_path    ${save_path} \
    --eval_first \
    --procedure regression --cutout_length -1 \
    --pre_crop_expand 0.2 \
    --scale_prob 0.5 --scale_min 0.9 --scale_max 1.1 --color_disturb 0.4 \
    --offset_max 0.2 --rotate_max 30 --rotate_prob 0.9 \
    --height 96 --width 96 --sigma 3 \
    --i_batch_size ${i_batch_size} --v_batch_size ${v_batch_size} --m_batch_size ${m_batch_size} \
    --print_freq 400 --print_freq_eval 1000 --eval_freq 5 --workers 16 \
    --heatmap_type gaussian 

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/ROBUST-eval.py \
    --eval_lists  ./cache_data/lists/AFLW/test.pth \
                  ./cache_data/lists/AFLW/test.front.pth \
                  ./cache_data/lists/AFLW/test-less-30-yaw.pth \
                  ./cache_data/lists/AFLW/test-more-60-yaw.pth \
    --mean_point  ./cache_data/lists/AFLW/train-mean.pth \
    --save_path   ${save_path}/ \
    --init_model  ${save_path}/last-info.pth \
    --robust_scale 0.2 --robust_offset 0.1 --robust_rotate 30 \
    --rand_seed 1111 \
    --print_freq 200 --workers 10
