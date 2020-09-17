# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash scripts/X-Mugsy/REG-SBR-TEST.sh 0 W01
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
v_batch_size=32

save_path=./snapshots/X-Mugsy/TEST-REG-SBR-${opt}-Mugsy-.${sbr_weight}-${det}-96x96-${i_batch_size}.${v_batch_size}-P49
#rm -rf ${save_path}


CUDA_VISIBLE_DEVICES=${gpus} python ./exps/SBR-main.py \
    --train_lists ./cache_data/lists/Mugsy/train-pure-annotated-video.pth \
    --eval_ilists ./cache_data/lists/Mugsy/train-pure-annotated-image.pth \
                  ./cache_data/lists/Mugsy/test-pure-annotated-image.pth \
    --mean_point  ./cache_data/lists/Mugsy/Mugsy-train-mean.pth \
    --boxindicator ${det} --use_gray ${use_gray} \
    --num_pts 18 --data_indicator Mugsy-18 \
    --procedure regression \
    --model_config ./configs/Mugsy/REG-detector.config \
    --opt_config   ./configs/300W-V2/ADAM.REG.${opt}.config \
    --sbr_config   ./configs/X-300W/SBR.REG.300VW.${sbr_weight}.config \
    --init_model   ./snapshots/X-300W/REG-DET-300W-RD09-${det}-96x96-32-P49/last-info.pth \
    --save_path    ${save_path} \
    --pre_crop_expand 0.2 \
    --scale_prob 0.5 --scale_min 0.9 --scale_max 1.1 --color_disturb 0.4 \
    --offset_max 0.2 --rotate_max 30 --rotate_prob 0.9 \
    --height 96 --width 96 --sigma 3 \
    --i_batch_size ${i_batch_size} --v_batch_size ${v_batch_size} \
    --print_freq 300 --print_freq_eval 2000 --eval_freq 10 --workers 20 \
    --heatmap_type gaussian


CUDA_VISIBLE_DEVICES=${gpus} python ./exps/ROBUST-eval.py \
    --eval_lists  ./cache_data/lists/Mugsy/train-pure-annotated-image.pth \
		  ./cache_data/lists/Mugsy/test-pure-annotated-image.pth \
    --mean_point  ./cache_data/lists/Mugsy/Mugsy-train-mean.pth \
    --save_path   ${save_path}/ \
    --init_model  ${save_path}/last-info.pth \
    --robust_scale 0.2 --robust_offset 0.1 --robust_rotate 30 \
    --rand_seed 1111 \
    --print_freq 200 --workers 10

rm ${save_path}/last-info.pth
