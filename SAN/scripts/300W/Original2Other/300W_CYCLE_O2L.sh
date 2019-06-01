#!/usr/bin/env sh
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for gpu devices and detector and sigma"
  exit 1
fi
gpus=$1
model=itn_cpm
epochs=55
stages=3
batch_size=8
GPUS=2
sigma=4
height=128
width=128
PREFIX=O2L_ITN
dataset_name=300W_$2

CUDA_VISIBLE_DEVICES=${gpus} python san_main.py \
    --train_list ./cache_data/lists/300W/Original/300w.train.$2 \
    --eval_lists ./cache_data/lists/300W/Light/300w.test.common.$2 \
        	 ./cache_data/lists/300W/Light/300w.test.challenge.$2 \
        	 ./cache_data/lists/300W/Light/300w.test.full.$2 \
        	 ./cache_data/lists/300W/Original/300w.test.full.$2 \
    --cycle_a_lists ./cache_data/lists/300W/Original/300w.train.$2 \
    --cycle_b_lists ./cache_data/lists/300W/Light/300w.test.full.$2 \
    --num_pts 68 --pre_crop_expand 0.2 \
    --arch ${model} --cpm_stage ${stages} \
    --save_path ./snapshots/${PREFIX}_${dataset_name}_${model}_${stages}_${epochs}_sigma${sigma}_${height}x${width}x8 \
    --learning_rate 0.00005 --decay 0.0005 --batch_size ${batch_size} --workers 20 --gpu_ids 0,1 \
    --epochs ${epochs} --schedule 30 35 40 45 50 --gammas 0.5 0.5 0.5 0.5 0.5 \
    --dataset_name ${dataset_name} \
    --scale_min 1 --scale_max 1 --scale_eval 1 --eval_batch ${batch_size} \
    --crop_height ${height} --crop_width ${width} --crop_perturb_max 30 \
    --sigma ${sigma} --print_freq 50 --print_freq_eval 100 --pretrain \
    --evaluation --heatmap_type gaussian --argmax_size 3 \
    --epoch_count 1 --niter 100 --niter_decay 100 --identity 0.1 \
    --cycle_batchSize 32 
