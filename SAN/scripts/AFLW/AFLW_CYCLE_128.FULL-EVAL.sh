#!/usr/bin/env sh
echo script name: $0
echo $# arguments
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for gpu devices"
  exit 1
fi
gpus=$1
model=itn_cpm
epochs=50
stages=3
batch_size=8
GPUS=2
sigma=4
height=128
width=128
dataset_name=AFLW_FULL_GTB
save_path=./snapshots/SAN_${dataset_name}_${model}_${stages}_${epochs}_sigma${sigma}_${height}x${width}x8 \

python san_main.py \
    --train_list ./cache_data/lists/AFLW/Original/train.GTB \
    --eval_lists ./cache_data/lists/AFLW/Original/test.GTB \
    --cycle_a_lists ./snapshots/CLUSTER-AFLW_GTB-3/cluster-00-03.lst \
    --cycle_b_lists ./snapshots/CLUSTER-AFLW_GTB-3/cluster-02-03.lst \
    --num_pts 19 --pre_crop_expand 0.2 \
    --arch ${model} --cpm_stage ${stages} \
    --save_path ./snapshots/SAN_${dataset_name}_${model}_${stages}_${epochs}_sigma${sigma}_${height}x${width}x8-EVAL \
    --cycle_model_path ${save_path}/cycle-gan/itn-epoch-200-201 \
    --resume ${save_path}/checkpoint_49.pth.tar --eval_once \
    --learning_rate 0.00005 --decay 0.0005 --batch_size ${batch_size} --workers 20 --gpu_ids ${gpus} \
    --epochs ${epochs} --schedule 30 35 40 45 --gammas 0.5 0.5 0.5 0.5 \
    --dataset_name ${dataset_name} \
    --scale_min 1 --scale_max 1 --scale_eval 1 --eval_batch ${batch_size} \
    --crop_height ${height} --crop_width ${width} --crop_perturb_max 30 \
    --sigma ${sigma} --print_freq 200 --print_freq_eval 100 --pretrain \
    --evaluation --heatmap_type gaussian --argmax_size 3 \
    --epoch_count 1 --niter 100 --niter_decay 100 --identity 0.1 \
    --cycle_batchSize 32
