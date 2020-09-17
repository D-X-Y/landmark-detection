# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division

import sys, time, torch, random, argparse, PIL
import os.path as osp
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
import numbers, numpy as np
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from config_utils import obtain_robust_args
from procedure    import prepare_seed, prepare_logger, prepare_data_augmentation
from procedure    import eval_robust_heatmap, eval_robust_regression
from log_utils    import AverageMeter, time_for_file, convert_secs2time, time_string
from config_utils import load_configure
from datasets     import RobustDataset, convert68to49
from models       import obtain_pro_model, remove_module_dict, count_parameters_in_MB, load_checkpoint
from xvision      import transforms3v as transforms
import xvision


procedures = {'heatmap'    : eval_robust_heatmap,
              'regression' : eval_robust_regression}


def main(args):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = True
  torch.set_num_threads( args.workers )
  print ('Evaluate the Robustness of a Detector : prepare_seed : {:}'.format(args.rand_seed))
  prepare_seed(args.rand_seed)

  assert args.init_model is not None and Path(args.init_model).exists(), 'invalid initial model path : {:}'.format(args.init_model)
  
  checkpoint = load_checkpoint( args.init_model )
  xargs      = checkpoint['args']
  eval_func  = procedures[xargs.procedure]

  logger     = prepare_logger(args)

  if xargs.use_gray == False:
    mean_fill = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])
  else:
    mean_fill = (0.5,)
    normalize = transforms.Normalize(mean=[mean_fill[0]], std=[0.5])

  robust_component =  [transforms.ToTensor(), normalize, transforms.PreCrop(xargs.pre_crop_expand)]
  robust_component += [transforms.RandomTrans(args.robust_scale, args.robust_offset, args.robust_rotate, args.robust_iters, args.robust_cache_dir, True)]
  robust_transform = transforms.Compose3V( robust_component )
  logger.log('--- arguments --- : {:}'.format( args ))
  logger.log('robust_transform  : {:}'.format( robust_transform ))

  recover      = xvision.transforms2v.ToPILImage(normalize)
  model_config = load_configure(xargs.model_config, logger)
  shape        = (xargs.height, xargs.width)
  logger.log('Model : {:} $$$$ Shape : {:}'.format(model_config, shape))

  # Evaluation Dataloader
  assert args.eval_lists is not None and len(args.eval_lists) > 0, 'invalid args.eval_lists : {:}'.format(args.eval_lists)
  eval_loaders = []
  for eval_list in args.eval_lists:
    eval_data = RobustDataset(robust_transform, xargs.sigma, model_config.downsample, xargs.heatmap_type, shape, xargs.use_gray, xargs.data_indicator)
    if xargs.x68to49:
      eval_data.load_list(eval_list, 68, xargs.boxindicator, True)
      convert68to49( eval_data )
    else:
      eval_data.load_list(eval_list, xargs.num_pts, xargs.boxindicator, True)
    eval_data.get_normalization_distance(None, True)
    if hasattr(xargs, 'batch_size'):
      batch_size = xargs.batch_size
    elif hasattr(xargs, 'i_batch_size') and xargs.i_batch_size > 0:
      batch_size = xargs.i_batch_size
    elif hasattr(xargs, 'v_batch_size') and xargs.v_batch_size > 0:
      batch_size = xargs.v_batch_size
    else:
      raise ValueError('can not find batch size information in xargs : {:}'.format( xargs ))
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    eval_loaders.append( eval_loader )

  # define the detection network
  detector = obtain_pro_model(model_config, xargs.num_pts, xargs.sigma, xargs.use_gray)
  assert model_config.downsample == detector.downsample, 'downsample is not correct : {:} vs {:}'.format(model_config.downsample, detector.downsample)
  logger.log("=> detector :\n {:}".format(detector))
  logger.log("=> Net-Parameters : {:} MB".format(count_parameters_in_MB(detector)))

  for i, eval_loader in enumerate(eval_loaders):
    logger.log('The [{:2d}/{:2d}]-th testing-data = {:}'.format(i, len(eval_loaders), eval_loader.dataset))

  logger.log('basic-arguments : {:}\n'.format(xargs))
  logger.log('xoxox-arguments : {:}\n'.format( args))


  detector.load_state_dict( remove_module_dict(checkpoint['detector']) )
  detector = detector.cuda()

  for ieval, loader in enumerate(eval_loaders):
    errors, valids, meta = eval_func(detector, loader, args.print_freq, logger)
    logger.log('[{:2d}/{:02d}] eval-data : error : mean={:.3f}, std={:.3f}'.format(ieval, len(eval_loaders), np.mean(errors), np.std(errors)))
    logger.log('[{:2d}/{:02d}] eval-data : valid : mean={:.3f}, std={:.3f}'.format(ieval, len(eval_loaders), np.mean(valids), np.std(valids)))
    nme, auc, pck_curves = meta.compute_mse(loader.dataset.dataset_name, logger)
  logger.close()


if __name__ == '__main__':
  args = obtain_robust_args()
  main(args)
