#!/usr/bin/env sh
# bash ./scripts/300W/300W_Cluster_Simple.sh 0,1 GTB 3
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for gpu devices and detector and the number of cluster"
  exit 1
fi
gpus=$1
cluster=$3
batch_size=64
height=224
width=224
dataset_name=300W_$2

CUDA_VISIBLE_DEVICES=${gpus} python cluster.py \
    --style_train_root ./cache_data/cache/300W \
    --train_list ./cache_data/lists/300W/Original/300w.train.$2 \
        	 ./cache_data/lists/300W/Original/300w.test.full.$2 \
    --learning_rate 0.01 --epochs 2 \
    --save_path ./snapshots/CLUSTER-${dataset_name}-${cluster}-TEST \
    --num_pts 68 --pre_crop_expand 0.2 \
    --dataset_name ${dataset_name} \
    --scale_min 1 --scale_max 1 --scale_eval 1 --eval_batch ${batch_size} --batch_size ${batch_size} \
    --crop_height ${height} --crop_width ${width} --crop_perturb_max 30 \
    --sigma 3 --print_freq 50 --print_freq_eval 100 --pretrain \
    --evaluation --heatmap_type gaussian --argmax_size 3 --n_clusters ${cluster}
