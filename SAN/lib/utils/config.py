# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Fast R-CNN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()

__C.DEBUG = False
__C.VIS   = False

#
# Testing options
#
__C.num_levels_scale = 1
__C.scale_starting = 0.8
__C.scale_ending   = 1.2

__C.TEST = edict()
__C.TEST.rescale_height = 1280
__C.TEST.rescale_width  = 960
# percentage of max value in the heat-map for smoothing
__C.TEST.heatmap_smooth_percentage = 0.4
# visualize the detected landmark only when scores are larger than this threshold
__C.TEST.visualization_threshold = 0.2
__C.TEST.minimum_width = 512
# use average heat-map from multi-scale
__C.TEST.heatmap_merge = 'avg'
# choose best points location from multi-scale
__C.TEST.pts_merge_scale = 'max'
# choose best points location from multiple peaks found (max or clustering)
__C.TEST.pts_merge_peak = 'max'
# tolerance of distance during clustering peak points found
__C.TEST.cluster_bandwidth = 10
__C.TEST.padValue = 128
__C.TEST.mean_value = 0.5

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

#
# Path Settings
__C.DATA_300VW = osp.abspath(osp.join(os.environ['HOME'], 'datasets', '300VW'))
__C.DATA_300W = osp.abspath(osp.join(os.environ['HOME'], 'datasets', '300VW', '300W-ALL'))
__C.DATA_LOCAL_300W = osp.abspath(osp.join(__C.ROOT_DIR, 'datasets', '300-W'))

__C.MUGSY = edict()
__C.MUGSY.FullFace = osp.abspath(osp.join(os.environ['HOME'], 'datasets', 'MUGSY', 'full_face', 'face'))
__C.LOCAL_MUGSY = edict()
__C.LOCAL_MUGSY.FullFace = osp.abspath(osp.join(__C.ROOT_DIR, 'datasets', 'MUGSY'))
__C.caffe_path = osp.abspath(osp.join(__C.ROOT_DIR, 'caffe'))

#
# Models
__C.model_pretrained = osp.abspath(osp.join(os.environ['HOME'], 'datasets', 'model_pretrained'))
__C.TRAIN.model_save_dir   = osp.abspath(osp.join(os.environ['HOME'], 'datasets', 'model_saved'))

#
# For temporal visual path
__C.INT_MAX_CAFFE = 11184000

#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1./16.
