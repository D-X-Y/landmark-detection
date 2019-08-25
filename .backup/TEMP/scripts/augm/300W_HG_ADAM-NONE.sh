#!/usr/bin/env sh
# sh scripts/augm/300W_HG_ADAM.sh 0
echo script name: $0
echo $# arguments
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for gpu devices"
  exit 1
fi
gpus=$1
HGV=V1
OPTV=V3
det=DET
batch_size=8
sigma=4
height=96
width=96

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/basic_main.py \
    --train_lists ./cache_data/lists/300W/300w.train.${det} \
    --eval_ilists ./cache_data/lists/300W/300w.test.common.${det} \
                  ./cache_data/lists/300W/300w.test.challenge.${det} \
                  ./cache_data/lists/300W/300w.test.full.${det} \
    --num_pts 68 --data_indicator 300W-68 \
    --model_config ./configs/face/HG.${HGV}.config \
    --opt_config   ./configs/face/ADAM.${OPTV}.config \
    --save_path    ./snapshots/300W-HG-${HGV}-ADAM-${OPTV}-${det}-NONE \
    --pre_crop_expand 0.2 \
    --sigma ${sigma} --batch_size ${batch_size} \
    --crop_height ${height} --crop_width ${width} --crop_perturb_max 0 --rotate_max 1 \
    --scale_prob 1.0 --scale_min 1 --scale_max 1 --scale_eval 1 \
    --print_freq 100 --workers 12 \
    --heatmap_type gaussian
