# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
CUDA_VISIBLE_DEVICES=2,3 python ./exps/lk_main.py \
	--train_lists ./cache_data/lists/demo/demo-sbr.lst \
	              ./cache_data/lists/300W/300w.train.DET \
	--eval_ilists ./cache_data/lists/demo/demo-sbr.lst \
	--num_pts 68 \
	--model_config ./configs/Detector.config \
	--opt_config   ./configs/LK.SGD.config \
	--lk_config    ./configs/mix.lk.config \
	--video_parser x-1-1 --save_path ./snapshots/CPM-SBR \
	--init_model ./snapshots/300W-CPM-DET/checkpoint/cpm_vgg16-epoch-049-050.pth  \
	--pre_crop_expand 0.2 --sigma 4 \
	--batch_size 8 --crop_perturb_max 5 --scale_prob 1 --scale_min 1 --scale_max 1 --scale_eval 1 --heatmap_type gaussian \
	--print_freq 10
