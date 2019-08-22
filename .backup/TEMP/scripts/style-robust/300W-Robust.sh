#!/usr/bin/env sh
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for gpu devices, the normalization layer"
  exit 1
fi
gpus=$1
HGV=V1
det=DET
GAN=LSGAN
batch_size=16
sigma=4
height=96
width=96
epochs=300
norm=$2
amsgrad=0

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/style_robust_main.py \
    --train_lists ./cache_data/lists/300W/300w.train.${det} \
    --eval_lists  ./cache_data/lists/300W/300w.test.common.${det} \
                  ./cache_data/lists/300W/300w.test.challenge.${det} \
                  ./cache_data/lists/300W/300w.test.full.${det} \
    --num_pts 68 --data_indicator 300W-68 \
    --model_config ./configs/face/HG.${HGV}.config \
    --gan_norm ${norm} \
    --GenNetPath   ./snapshots/300W-STYLE-${GAN}-${norm}-AMS${amsgrad}-${det}/checkpoint/style-epoch-499-500.pth \
    --save_path    ./snapshots/300W-DETECTION-${GAN}-${norm}-AMS${amsgrad}-${det} \
    --epochs ${epochs} --critic_iters 5 \
    --LR_D 0.0002 --LR_N 0.01 --debug \
    --eval_freq 10 --use_tf \
    --pre_crop_expand 0.2 \
    --sigma ${sigma} --batch_size ${batch_size} \
    --crop_height ${height} --crop_width ${width} --crop_perturb_max 30 --rotate_max 20 \
    --scale_prob 1.0 --scale_min 0.9 --scale_max 1.1 --scale_eval 1 \
    --print_freq 50 --workers 12 \
    --heatmap_type gaussian
