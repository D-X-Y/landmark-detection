#!/usr/bin/env sh
echo script name: $0
echo $# arguments
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for gpu devices, the normalization layer, the amsgrad, and the face detector"
  exit 1
fi
TIME=$(date +"%Y-%h-%d--%T")
TIME="${TIME//:/-}"
gpus=$1
det=$4
batch_size=64
sigma=4
height=96
width=96
epochs=500
norm=$2
amsgrad=$3

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/style_image_LSGAN.py \
    --train_lists ./cache_data/lists/300W/300w.train.${det} \
    --eval_lists  ./cache_data/lists/300W/300w.test.full.${det} \
    --num_pts 68 --data_indicator 300W-68 \
    --epochs ${epochs} --critic_iters 1 --gan_norm ${norm} \
    --LR_D 0.0002 --LR_G 0.0002 --wgan_clip 0.01 --debug \
    --amsgrad ${amsgrad} \
    --eval_freq 100 \
    --save_path    ./snapshots/300W-STYLE-LSGAN-${norm}-AMS${amsgrad}-${det}-${TIME} \
    --pre_crop_expand 0.2 \
    --sigma ${sigma} --batch_size ${batch_size} \
    --crop_height ${height} --crop_width ${width} --crop_perturb_max 5 --rotate_max 10 \
    --scale_prob 1.0 --scale_min 0.9 --scale_max 1.1 --scale_eval 1 \
    --print_freq 50 --workers 12 \
    --heatmap_type gaussian
