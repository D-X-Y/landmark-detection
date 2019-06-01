# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
CUDA_VISIBLE_DEVICES=0,1 python ./exps/basic_main.py \
	--train_lists ./cache_data/lists/AFLW/train.GTB \
	--eval_ilists ./cache_data/lists/AFLW/test.GTB \
	--num_pts 19 \
	--model_config ./configs/Detector.config \
	--opt_config ./configs/SGD.config \
	--save_path ./snapshots/AFLW-CPM-DET \
	--pre_crop_expand 0.2 --sigma 4 --batch_size 8 \
	--crop_perturb_max 30 --rotate_max 20 \
	--scale_prob 1.0 --scale_min 0.9 --scale_max 1.1 --scale_eval 1 \
	--heatmap_type gaussian
