#!/usr/bin/env sh
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for gpu devices and detector and the cluster"
  exit 1
fi
gpus=$1
model=vgg16_base
cluster=$3
batch_size=64
height=224
width=224
dataset_name=AFLW_$2

CUDA_VISIBLE_DEVICES=${gpus} python cluster.py \
    --style_train_root ./cache_data/cache/AFLW \
    --style_eval_root ./cache_data/cache/300W \
    --train_list ./cache_data/lists/AFLW/Original/train.$2 \
        	 ./cache_data/lists/AFLW/Original/test.$2 \
    --learning_rate 0.01 --epochs 2 \
    --save_path ./snapshots/CLUSTER-${dataset_name}-${cluster} \
    --num_pts 68 --pre_crop_expand 0.2 \
    --arch ${model} --cpm_stage 3 \
    --dataset_name ${dataset_name} \
    --scale_min 1 --scale_max 1 --scale_eval 1 --eval_batch ${batch_size} --batch_size ${batch_size} \
    --crop_height ${height} --crop_width ${width} --crop_perturb_max 30 \
    --sigma 3 --print_freq 100 --print_freq_eval 100 --pretrain --gpu_ids 0 \
    --evaluation --heatmap_type gaussian --argmax_size 3 --n_clusters ${cluster}
