# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Basic Training

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
from config_utils import basic_basic_args_v2 as obtain_basic_args

from procedure import get_path2image, save_checkpoint
from procedure import prepare_seed, prepare_logger, prepare_data_augmentation
from procedure import basic_main_heatmap    , basic_main_regression
from procedure import basic_eval_all_heatmap, basic_eval_all_regression
from datasets import GeneralDatasetV2 as Dataset, SpecialBatchSampler, convert68to49
#from xvision import transformsImage as transforms
from xvision import transforms2v as transforms
from log_utils import AverageMeter, time_for_file, convert_secs2time, time_string
from config_utils import load_configure
from models import obtain_pro_model, remove_module_dict, count_parameters_in_MB, load_checkpoint
from optimizer import obtain_optimizer


procedures = {'default-train'   : basic_main_heatmap,
              'default-test'    : basic_eval_all_heatmap,
              'heatmap-train'   : basic_main_heatmap,
              'heatmap-test'    : basic_eval_all_heatmap,
              'regression-train': basic_main_regression,
              'regression-test' : basic_eval_all_regression}


def main(args):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = True
  torch.set_num_threads( args.workers )
  print ('Training Base Detector : prepare_seed : {:}'.format(args.rand_seed))
  prepare_seed(args.rand_seed)

  basic_main, eval_all = procedures['{:}-train'.format(args.procedure)], procedures['{:}-test'.format(args.procedure)]

  logger = prepare_logger(args)

  # General Data Augmentation
  normalize, train_transform, eval_transform, robust_transform = prepare_data_augmentation(transforms, args)
  #data_cache = get_path2image( args.shared_img_cache )
  data_cache = None

  recover = transforms.ToPILImage(normalize)
  args.tensor2imageF = recover
  assert (args.scale_min+args.scale_max) / 2 == 1, 'The scale is not ok : {:} ~ {:}'.format(args.scale_min, args.scale_max)
  logger.log('robust_transform : {:}'.format( robust_transform ))
  
  # Model Configure Load
  model_config = load_configure(args.model_config, logger)
  shape = (args.height, args.width)
  logger.log('--> {:}\n--> Sigma : {:}, Shape : {:}'.format(model_config, args.sigma, shape))

  # Training Dataset
  if args.train_lists:
    train_data   = Dataset(train_transform, args.sigma, model_config.downsample, args.heatmap_type, shape, args.use_gray, args.mean_point, args.data_indicator, data_cache)
    safex_data   = Dataset( eval_transform, args.sigma, model_config.downsample, args.heatmap_type, shape, args.use_gray, args.mean_point, args.data_indicator, data_cache)
    train_data.set_cutout( args.cutout_length )
    safex_data.set_cutout( args.cutout_length )
    train_data.load_list(args.train_lists, args.num_pts, args.boxindicator, args.normalizeL, True)
    safex_data.load_list(args.train_lists, args.num_pts, args.boxindicator, args.normalizeL, True)
    if args.sampler is None:
      train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
      safex_loader = torch.utils.data.DataLoader(safex_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    else:
      train_sampler = SpecialBatchSampler(train_data, args.batch_size, args.sampler)
      safex_sampler = SpecialBatchSampler(safex_data, args.batch_size, args.sampler)
      logger.log('Training-sampler : {:}'.format(train_sampler))
      train_loader  = torch.utils.data.DataLoader(train_data, batch_sampler=train_sampler, num_workers=args.workers, pin_memory=True)
      safex_loader  = torch.utils.data.DataLoader(safex_data, batch_sampler=safex_sampler, num_workers=args.workers, pin_memory=True)
    logger.log('Training-data : {:}'.format(train_data))
  else:
    train_data, safex_loader = None, None

  #train_data[0]
  # Evaluation Dataloader
  eval_loaders = []
  if args.eval_ilists is not None:
    for eval_ilist in args.eval_ilists:
      eval_idata = Dataset(eval_transform, args.sigma, model_config.downsample, args.heatmap_type, shape, args.use_gray, args.mean_point, args.data_indicator, data_cache)
      eval_idata.load_list(eval_ilist, args.num_pts, args.boxindicator, args.normalizeL, True)
      eval_iloader = torch.utils.data.DataLoader(eval_idata, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
      eval_loaders.append((eval_iloader, False))
  if args.eval_vlists is not None:
    for eval_vlist in args.eval_vlists:
      eval_vdata = Dataset(eval_transform, args.sigma, model_config.downsample, args.heatmap_type, shape, args.use_gray, args.mean_point, args.data_indicator, data_cache)
      eval_vdata.load_list(eval_vlist, args.num_pts, args.boxindicator, args.normalizeL, True)
      eval_vloader = torch.utils.data.DataLoader(eval_vdata, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
      eval_loaders.append((eval_vloader, True))
  # from 68 points to 49 points, removing the face contour
  if args.x68to49:
    assert args.num_pts == 68, 'args.num_pts is not 68 vs. {:}'.format(args.num_pts)
    if train_data is not None: train_data = convert68to49( train_data )
    for eval_loader, is_video in eval_loaders:
      convert68to49( eval_loader.dataset )
    args.num_pts = 49

  # define the detector
  detector = obtain_pro_model(model_config, args.num_pts, args.sigma, args.use_gray)
  assert model_config.downsample == detector.downsample, 'downsample is not correct : {:} vs {:}'.format(model_config.downsample, detector.downsample)
  logger.log("=> detector :\n {:}".format(detector))
  logger.log("=> Net-Parameters : {:} MB".format(count_parameters_in_MB(detector)))

  for i, eval_loader in enumerate(eval_loaders):
    eval_loader, is_video = eval_loader
    logger.log('The [{:2d}/{:2d}]-th testing-data [{:}] = {:}'.format(i, len(eval_loaders), 'video' if is_video else 'image', eval_loader.dataset))

  logger.log('arguments : {:}\n'.format(args))
  logger.log('train_transform : {:}'.format(train_transform))
  logger.log('eval_transform  : {:}'.format(eval_transform))
  opt_config = load_configure(args.opt_config, logger)

  if hasattr(detector, 'specify_parameter'):
    net_param_dict = detector.specify_parameter(opt_config.LR, opt_config.weight_decay)
  else:
    net_param_dict = detector.parameters()

  optimizer, scheduler, criterion = obtain_optimizer(net_param_dict, opt_config, logger)
  logger.log('criterion : {:}'.format(criterion))
  detector, criterion = detector.cuda(), criterion.cuda()
  net = torch.nn.DataParallel(detector)

  last_info = logger.last_info()
  if last_info.exists():
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info = torch.load(last_info)
    start_epoch = last_info['epoch'] + 1
    checkpoint  = torch.load(last_info['last_checkpoint'])
    assert last_info['epoch'] == checkpoint['epoch'], 'Last-Info is not right {:} vs {:}'.format(last_info, checkpoint['epoch'])
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    logger.log("=> load-ok checkpoint '{:}' (epoch {:}) done" .format(logger.last_info(), checkpoint['epoch']))
  elif args.init_model is not None:
    last_checkpoint = load_checkpoint( args.init_model )
    net.load_state_dict( last_checkpoint['detector'] )
    logger.log("=> initialize the detector : {:}".format(args.init_model))
    start_epoch = 0
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch = 0


  if args.eval_once is not None:
    logger.log("=> only evaluate the model once")
    #if safex_loader is not None:
    #  safe_results, safe_metas = eval_all(args, [(safex_loader, False)], net, criterion, 'eval-once-train', logger, opt_config, robust_transform)
    #  logger.log('-'*50 + ' evaluate the training set')
    #import pdb; pdb.set_trace()
    eval_results, eval_metas = eval_all(args, eval_loaders, net, criterion, 'eval-once', logger, opt_config, robust_transform)
    all_predictions = [eval_meta.predictions for eval_meta in eval_metas]
    torch.save(all_predictions, osp.join(args.save_path, '{:}-predictions.pth'.format(args.eval_once)))
    logger.log('==>> evaluation results : {:}'.format(eval_results))
    logger.log('==>> configuration : {:}'.format(model_config))
    logger.close() ; return


  # Main Training and Evaluation Loop
  start_time = time.time()
  epoch_time = AverageMeter()
  for epoch in range(start_epoch, opt_config.epochs):

    need_time = convert_secs2time(epoch_time.avg * (opt_config.epochs-epoch), True)
    epoch_str = 'epoch-{:03d}-{:03d}'.format(epoch, opt_config.epochs)
    LRs       = scheduler.get_lr()
    logger.log('\n==>>{:s} [{:s}], [{:s}], LR : [{:.5f} ~ {:.5f}], Config : {:}'.format(time_string(), epoch_str, need_time, min(LRs), max(LRs), opt_config))

    # train for one epoch
    train_loss, train_meta, train_nme = basic_main(args, train_loader, net, criterion, optimizer, epoch_str, logger, opt_config, 'train')
    scheduler.step()
    # log the results    
    logger.log('==>>{:s} Train [{:}] Average Loss = {:.6f}, NME = {:.2f}'.format(time_string(), epoch_str, train_loss, train_nme*100))

    save_path = save_checkpoint({
          'epoch': epoch,
          'args' : deepcopy(args),
          'arch' : model_config.arch,
          'detector'  : net.state_dict(),
          'state_dict': net.state_dict(),
          'scheduler' : scheduler.state_dict(),
          'optimizer' : optimizer.state_dict(),
          }, logger.path('model') / 'seed-{:}-{:}.pth'.format(args.rand_seed, model_config.arch), logger)

    last_info = save_checkpoint({
          'epoch': epoch,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
          }, logger.last_info(), logger)

    if (args.eval_freq is None) or (epoch+1 == opt_config.epochs) or (epoch%args.eval_freq == 0):
      if epoch+1 == opt_config.epochs: _robust_transform = robust_transform
      else                           : _robust_transform = None
      logger.log('')
      eval_results, eval_metas = eval_all(args, eval_loaders, net, criterion, epoch_str, logger, opt_config, _robust_transform)
      #save_path = save_checkpoint(eval_metas, logger.path('meta') / '{:}-{:}.pth'.format(model_config.arch, epoch_str), logger)
      save_path = save_checkpoint(eval_metas, logger.path('meta') / 'seed-{:}-{:}.pth'.format(args.rand_seed, model_config.arch), logger)
      logger.log('==>> evaluation results : {:}\n==>> save evaluation results into {:}.'.format(eval_results, save_path))
    
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  logger.log('Final checkpoint into {:}'.format(logger.last_info()))
  logger.close()


if __name__ == '__main__':
  args = obtain_basic_args()
  main(args)
