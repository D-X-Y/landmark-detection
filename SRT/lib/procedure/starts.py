# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, sys, time, torch, random, PIL, copy, numpy as np
from log_utils import time_for_file, Logger


def prepare_seed(rand_seed):
  np.random.seed(rand_seed)
  random.seed(rand_seed)
  torch.manual_seed(rand_seed)
  torch.cuda.manual_seed_all(rand_seed)


def prepare_logger(xargs):
  args = copy.deepcopy( xargs )
  logstr = 'seed-{:}-time-{:}'.format(args.rand_seed, time_for_file())
  logger = Logger(args.save_path, logstr)
  logger.log('Main Function with logger : {:}'.format(logger))
  logger.log('Arguments : -------------------------------')
  for name, value in args._get_kwargs():
    logger.log('{:16} : {:}'.format(name, value))
  logger.log("Python  Version : {:}".format(sys.version.replace('\n', ' ')))
  logger.log("Pillow  Version : {:}".format(PIL.__version__))
  logger.log("PyTorch Version : {:}".format(torch.__version__))
  logger.log("cuDNN   Version : {:}".format(torch.backends.cudnn.version()))
  logger.log("CUDA available  : {:}".format(torch.cuda.is_available()))
  logger.log("CUDA GPU numbers: {:}".format(torch.cuda.device_count()))
  return logger


def prepare_data_augmentation(transforms, xargs):
  args = copy.deepcopy( xargs )
  # General Data Augmentation
  if args.use_gray == False:
    mean_fill = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])
  else:
    mean_fill = (0.5,)
    normalize = transforms.Normalize(mean=[mean_fill[0]], std=[0.5])

  train_transform = [transforms.ToTensor()]
  if args.color_disturb is not None: train_transform += [transforms.ColorDisturb(args.color_disturb)]
  train_transform += [normalize, transforms.PreCrop(args.pre_crop_expand), \
                     transforms.RandomOffset(args.offset_max), \
                     transforms.AugScale(args.scale_prob, args.scale_min, args.scale_max)]
  if args.crop_max      is not None: train_transform += [transforms.AugCrop(args.crop_max)]
  if args.rotate_prob   is not None: train_transform += [transforms.AugRotate(args.rotate_max, args.rotate_prob)]
  if args.arg_flip:
    train_transform.append( transforms.AugHorizontalFlip() )
  train_transform = transforms.Compose2V( train_transform )

  eval_transform  = transforms.Compose2V([transforms.ToTensor(), normalize, \
                                          transforms.PreCrop(args.pre_crop_expand), \
                                          transforms.CenterCrop(args.crop_max)])
  if args.robust_iter is None or args.robust_iter <= 0:
    robust_transform = None
  else:
    robust_transform = transforms.Compose2V([transforms.ToTensor(), normalize, \
                                             transforms.PreCrop(args.pre_crop_expand), \
                                             transforms.RandomTransf((0.9,1.1), 0.9, 30, args.robust_iter)])

  return normalize, train_transform, eval_transform, robust_transform


def get_path2image(cache_lists):
  if cache_lists is None: return {}
  cache = {}
  for flist in cache_lists:
    xdata = torch.load( flist )
    cache = {**cache, **xdata}
  return cache
