# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash scripts/X-300W/REG-300W-P49.sh 0 RD01
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for gpu devices, and the optimizer"
  exit 1
fi
gpus=$1
opt=$2
det=default
use_gray=1
batch_size=32

save_path=./snapshots/X-300W/REG-DET-300W-${opt}-${det}-96x96-${batch_size}-P49
rm -rf ${save_path}


CUDA_VISIBLE_DEVICES=${gpus} python ./exps/BASE-main.py \
    --train_lists ./cache_data/lists/300W/300w.train.pth \
    --eval_ilists ./cache_data/lists/300W/300w.test-common.pth \
                  ./cache_data/lists/300W/300w.test-challenge.pth \
                  ./cache_data/lists/300W/300w.test-full.pth \
    --eval_vlists ./cache_data/lists/300VW/300VW.test-1.pth \
                  ./cache_data/lists/300VW/300VW.test-2.pth \
		  ./cache_data/lists/300VW/300VW.test-3.pth \
    --mean_point  ./cache_data/lists/300W/300w.train-mean.pth \
    --shared_img_cache cache_data/lists/300W/path2tensor-G${use_gray}.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 68 --data_indicator face49-300W --x68to49 \
    --model_config ./configs/face/REG/REG.300W.config \
    --opt_config   ./configs/300W-V2/ADAM.REG.${opt}.config \
    --save_path    ${save_path} \
    --procedure regression --cutout_length -1 \
    --pre_crop_expand 0.2 \
    --scale_prob 0.5 --scale_min 0.9 --scale_max 1.1 --color_disturb 0.4 \
    --offset_max 0.2 --rotate_max 30 --rotate_prob 0.9 \
    --height 96 --width 96 --sigma 3 --batch_size ${batch_size} \
    --print_freq 50 --print_freq_eval 1000 --eval_freq 10 --workers 40 \
    --heatmap_type gaussian 


CUDA_VISIBLE_DEVICES=${gpus} python ./exps/ROBUST-eval.py \
    --eval_lists  ./cache_data/lists/300W/300w.train.pth \
                  ./cache_data/lists/300W/300w.test-common.pth \
                  ./cache_data/lists/300W/300w.test-challenge.pth \
                  ./cache_data/lists/300W/300w.test-full.pth \
                  ./cache_data/lists/300VW/300VW.test-1.pth \
                  ./cache_data/lists/300VW/300VW.test-2.pth \
		  ./cache_data/lists/300VW/300VW.test-3.pth \
    --mean_point  ./cache_data/lists/300W/300w.train-mean.pth \
    --save_path   ${save_path}/ \
    --init_model  ${save_path}/last-info.pth \
    --robust_scale 0.2 --robust_offset 0.1 --robust_rotate 30 \
    --rand_seed 1111 \
    --print_freq 200 --workers 10
