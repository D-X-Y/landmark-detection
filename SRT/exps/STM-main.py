# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Spatial-Temporal-Multiview

import sys, time, torch, random, argparse, PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
import numbers, numpy as np
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from config_utils import obtain_stm_args_v2 as obtain_args

from procedure import prepare_seed, prepare_logger, save_checkpoint, prepare_data_augmentation
from procedure import stm_main_regression, stm_main_heatmap
from procedure import basic_eval_all_regression, basic_eval_all_heatmap
from datasets import GeneralDatasetV2 as IDataset, convert68to49
from datasets import STMDataset, StmBatchSampler
from xvision import transforms2v as transforms
from log_utils import AverageMeter, time_for_file, convert_secs2time, time_string
from config_utils import load_configure
from models import obtain_pro_stm, remove_module_dict, load_checkpoint
from optimizer import obtain_optimizer


procedures = {'default-train'   : stm_main_regression,
              'default-test'    : basic_eval_all_regression,
              'regression-train': stm_main_regression,
              'regression-test' : basic_eval_all_regression,
              'heatmap-train'   : stm_main_heatmap,
              'heatmap-test'    : basic_eval_all_heatmap}


def main(args):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = True
  torch.set_num_threads( args.workers )
  print ('Training Base Detector : prepare_seed : {:}'.format(args.rand_seed))
  prepare_seed(args.rand_seed)

  stm_main, eval_all = procedures['{:}-train'.format(args.procedure)], procedures['{:}-test'.format(args.procedure)]

  logger = prepare_logger(args)

  # General Data Augmentation
  normalize, train_transform, eval_transform, robust_transform = prepare_data_augmentation(transforms, args)
  recover = transforms.ToPILImage(normalize)
  args.tensor2imageF = recover
  assert (args.scale_min+args.scale_max) / 2 == 1, 'The scale is not ok : {:} ~ {:}'.format(args.scale_min, args.scale_max)
  
  # Model Configure Load
  model_config = load_configure(args.model_config, logger)
  stm_config   = load_configure(args.stm_config, logger)
  shape = (args.height, args.width)
  logger.log('--> {:}\n--> Sigma : {:}, Shape : {:}'.format(model_config, args.sigma, shape))
  logger.log('--> STM Configuration : {:}\n'.format(stm_config))

  # Training Dataset
  train_data    = STMDataset(train_transform, args.sigma, model_config.downsample, args.heatmap_type, shape, args.use_gray, \
                             args.data_indicator, stm_config, transforms.ToPILImage(normalize, 'cv2gray'))
  train_data.load_list(args.train_lists, args.num_pts, args.boxindicator, args.normalizeL, True)
  train_data.prepare_multiview()
  batch_sampler = StmBatchSampler(train_data, args.i_batch_size, args.v_batch_size, args.m_batch_size)
  train_loader  = torch.utils.data.DataLoader(train_data, batch_sampler=batch_sampler, num_workers=args.workers, pin_memory=True)

  # Evaluation Dataloader
  eval_loaders = []
  if args.eval_ilists is not None:
    for eval_ilist in args.eval_ilists:
      eval_idata = IDataset(eval_transform, args.sigma, model_config.downsample, args.heatmap_type, shape, args.use_gray, None, args.data_indicator)
      eval_idata.load_list(eval_ilist, args.num_pts, args.boxindicator, args.normalizeL, True)
      eval_iloader = torch.utils.data.DataLoader(eval_idata, batch_size=args.i_batch_size+args.v_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
      eval_loaders.append((eval_iloader, False))
  if args.eval_vlists is not None:
    for eval_vlist in args.eval_vlists:
      eval_vdata = IDataset(eval_transform, args.sigma, model_config.downsample, args.heatmap_type, shape, args.use_gray, None, args.data_indicator)
      eval_vdata.load_list(eval_vlist, args.num_pts, args.boxindicator, args.normalizeL, True)
      eval_vloader = torch.utils.data.DataLoader(eval_vdata, batch_size=args.i_batch_size+args.v_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
      eval_loaders.append((eval_vloader, True))
  # from 68 points to 49 points, removing the face contour
  if args.x68to49:
    assert args.num_pts == 68, 'args.num_pts is not 68 vs. {:}'.format(args.num_pts)
    if train_data is not None: train_data = convert68to49( train_data )
    for eval_loader, is_video in eval_loaders:
      convert68to49( eval_loader.dataset )
    args.num_pts = 49

  # define the STM model (accelerated STM)
  net = obtain_pro_stm(model_config, stm_config, args.num_pts, args.sigma, args.use_gray)
  assert model_config.downsample == net.downsample, 'downsample is not correct : {:} vs {:}'.format(model_config.downsample, net.downsample)
  logger.log("=> network :\n {}".format(net))

  logger.log('Training-data : {:}'.format(train_data))
  for i, eval_loader in enumerate(eval_loaders):
    eval_loader, is_video = eval_loader
    logger.log('The [{:2d}/{:2d}]-th testing-data [{:}] = {:}'.format(i, len(eval_loaders), 'video' if is_video else 'image', eval_loader.dataset))

  logger.log('arguments : {:}'.format(args))
  opt_config = load_configure(args.opt_config, logger)

  if hasattr(net, 'specify_parameter'): net_param_dict = net.specify_parameter(opt_config.LR, opt_config.weight_decay)
  else                                : net_param_dict = net.parameters()

  optimizer, scheduler, criterion = obtain_optimizer(net_param_dict, opt_config, logger)
  logger.log('criterion : {:}'.format(criterion))
  net, criterion = net.cuda(), criterion.cuda()
  net = torch.nn.DataParallel(net)

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
    last_checkpoint = load_checkpoint(args.init_model)
    checkpoint = remove_module_dict(last_checkpoint['state_dict'], False)
    net.module.detector.load_state_dict( checkpoint )
    logger.log("=> initialize the detector : {:}".format(args.init_model))
    start_epoch = 0
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch = 0

  detector = torch.nn.DataParallel(net.module.detector)

  if args.eval_first:
    logger.log('===>>> First Time Evaluation')
    eval_results, eval_metas = eval_all(args, eval_loaders, detector, criterion, 'Before-Training', logger, opt_config, None)
    save_path = save_checkpoint(eval_metas, logger.path('meta') / '{:}-first.pth'.format(model_config.arch), logger)
    logger.log('===>>> Before Training : {:}'.format(eval_results))

  # Main Training and Evaluation Loop
  start_time = time.time()
  epoch_time = AverageMeter()
  for epoch in range(start_epoch, opt_config.epochs):

    need_time = convert_secs2time(epoch_time.avg * (opt_config.epochs-epoch), True)
    epoch_str = 'epoch-{:03d}-{:03d}'.format(epoch, opt_config.epochs)
    LRs       = scheduler.get_lr()
    logger.log('\n==>>{:s} [{:s}], [{:s}], LR : [{:.5f} ~ {:.5f}], Config : {:}'.format(time_string(), epoch_str, need_time, min(LRs), max(LRs), opt_config))

    # train for one epoch
    train_loss, train_nme = stm_main(args, train_loader, net, criterion, optimizer, epoch_str, logger, opt_config, stm_config, (epoch>=stm_config.start_sbr,epoch>=stm_config.start_stm), 'train')
    scheduler.step()
    # log the results    
    logger.log('==>>{:s} Train [{:}] Average Loss = {:.6f}, NME = {:.2f}'.format(time_string(), epoch_str, train_loss, train_nme*100))

    save_path = save_checkpoint({
          'epoch': epoch,
          'args' : deepcopy(args),
          'arch' : model_config.arch,
          'detector'  : detector.state_dict(),
          'state_dict': net.state_dict(),
          'scheduler' : scheduler.state_dict(),
          'optimizer' : optimizer.state_dict(),
          }, logger.path('model') / 'ckp-seed-{:}-last-{:}.pth'.format(args.rand_seed, model_config.arch), logger)

    last_info = save_checkpoint({
          'epoch': epoch,
          'last_checkpoint': save_path,
          }, logger.last_info(), logger)
    if (args.eval_freq is None) or (epoch+1 == opt_config.epochs) or (epoch%args.eval_freq == 0):

      if epoch+1 == opt_config.epochs: _robust_transform = robust_transform
      else                           : _robust_transform = None
      logger.log('')
      eval_results, eval_metas = eval_all(args, eval_loaders, detector, criterion, epoch_str, logger, opt_config, _robust_transform)
      save_path = save_checkpoint(eval_metas, logger.path('meta') / '{:}-{:}.pth'.format(model_config.arch, epoch_str), logger)
      logger.log('==>> evaluation results : {:}'.format(eval_results))
    
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  logger.log('Final checkpoint into {:}'.format(logger.last_info()))

  logger.close()


if __name__ == '__main__':
  args = obtain_args()
  main(args)
