# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division

import os, sys, time, torch, random, argparse, PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
import numbers, numpy as np
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from config_utils import obtain_StyleDet_args
from procedure import prepare_seed, save_checkpoint
from procedure import basic_eval_all as eval_all
from procedure import compute_stage_loss, show_stage_loss
from procedure import generate_noise
from datasets import GeneralDataset as Dataset
from xvision import transforms, style_trans
from log_utils import Logger, AverageMeter, time_for_file, convert_secs2time, time_string
from config_utils import load_configure
from models import obtain_stlye

def mse_loss(inputs, targets):
  return torch.mean((inputs - targets) ** 2)

def main(args):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = True
  print ('In basic_main : prepare_seed : {:}'.format(args.rand_seed))
  prepare_seed(args.rand_seed)

  logstr = 'seed-{:}-time-{:}'.format(args.rand_seed, time_for_file())
  logger = Logger(args.save_path, logstr, args.use_tf)
  logger.log('Main Function with logger : {:}'.format(logger))
  logger.log('Arguments : -------------------------------')
  for name, value in args._get_kwargs():
    logger.log('{:16} : {:}'.format(name, value))
  logger.log("Python  version : {}".format(sys.version.replace('\n', ' ')))
  logger.log("Pillow  version : {}".format(PIL.__version__))
  logger.log("PyTorch version : {}".format(torch.__version__))
  logger.log("cuDNN   version : {}".format(torch.backends.cudnn.version()))

  # General Data Argumentation
  mean_fill   = tuple( [int(x*255) for x in [0.5, 0.5, 0.5] ] )
  normalize   = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])
  assert args.arg_flip == False, 'The flip is : {}, rotate is {}'.format(args.arg_flip, args.rotate_max)
  train_transform  = [transforms.PreCrop(args.pre_crop_expand)]
  train_transform += [transforms.TrainScale2WH((args.crop_width, args.crop_height))]
  train_transform += [style_trans.AugStyle()]
  train_transform += [transforms.AugScale(args.scale_prob, args.scale_min, args.scale_max)]
  #if args.arg_flip:
  #  train_transform += [transforms.AugHorizontalFlip()]
  if args.rotate_max:
    train_transform += [transforms.AugRotate(args.rotate_max)]
  train_transform += [transforms.AugCrop(args.crop_width, args.crop_height, args.crop_perturb_max, mean_fill)]
  train_transform += [transforms.ToTensor(), normalize]
  train_transform  = transforms.Compose( train_transform )

  eval_transform  = transforms.Compose([transforms.PreCrop(args.pre_crop_expand), \
                                        transforms.TrainScale2WH((args.crop_width, args.crop_height)), \
                                        transforms.ToTensor(), normalize])
  assert (args.scale_min+args.scale_max) / 2 == args.scale_eval, 'The scale is not ok : {},{} vs {}'.format(args.scale_min, args.scale_max, args.scale_eval)
  recover = transforms.ToPILImageWithNorm(normalize)
  
  # Model Configure Load
  model_config = load_configure(args.model_config, logger)
  args.sigma   = args.sigma * args.scale_eval
  logger.log('Real Sigma : {:}'.format(args.sigma))

  # Training Dataset
  train_data   = Dataset(train_transform, args.sigma, model_config.downsample, args.heatmap_type, args.data_indicator)
  train_data.load_list(args.train_lists, args.num_pts, True)
  logger.log('Training Data : {:}'.format(train_data))
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

  eval_loaders = []
  if args.eval_lists is not None:
    for idx, eval_list in enumerate(args.eval_lists):
      eval_data = Dataset(eval_transform, args.sigma, model_config.downsample, args.heatmap_type, args.data_indicator)
      eval_data.load_list(eval_list, args.num_pts, True)
      logger.log('Evaluation {:}-th Data : {:}'.format(idx, eval_data))
      eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
      eval_loaders.append( (eval_loader, True) )

  # Evaluation Dataloader
  DetNet, GenNet, DisNet = obtain_stlye(model_config, args.num_pts + 1, 4, 256, args.gan_norm)

  logger.log('Face Generator : \n{:}'.format(GenNet))
  logger.log('Face Discriminator : \n{:}'.format(DisNet))
  logger.log('Face Landmark Prediction : \n{:}'.format(DetNet))
  logger.log('Train : {:}\n Test : {:}\nArguments : {:}'.format(train_loader, eval_loader, args))

  optimizerD = torch.optim.Adam(DisNet.parameters(), lr=args.LR_D, betas=(0.5, 0.9))
  optimizerN = torch.optim.Adam(DetNet.parameters(), lr=args.LR_N, betas=(0.5, 0.9))

  DetNet, GenNet, DisNet = torch.nn.DataParallel(DetNet).cuda(), torch.nn.DataParallel(GenNet).cuda(), torch.nn.DataParallel(DisNet).cuda()
  criterion = torch.nn.MSELoss(True).cuda()
  MSE_loss  = torch.nn.MSELoss().cuda()

  # Load Generator
  if os.path.isfile(args.GenNetPath):
    tempinfo = torch.load(args.GenNetPath)
    GenNet.load_state_dict(tempinfo['state_dict_G'])
  else:
    raise ValueError('The generator path is not avaliable : {:}'.format(args.GenNetPath))

  last_info = logger.last_info()
  if last_info.exists():
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info = torch.load(last_info)
    start_epoch = last_info['epoch'] + 1
    checkpoint  = torch.load(last_info['last_checkpoint'])
    assert last_info['epoch'] == checkpoint['epoch'], 'Last-Info is not right {:} vs {:}'.format(last_info, checkpoint['epoch'])
    DetNet.load_state_dict(checkpoint['state_dict_DET'])
    GenNet.load_state_dict(checkpoint['state_dict_Gen'])
    DisNet.load_state_dict(checkpoint['state_dict_D'])
    optimizerN.load_state_dict(checkpoint['optimizerN'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])
    logger.log("=> load-ok checkpoint '{:}' (epoch {:}) done" .format(logger.last_info(), checkpoint['epoch']))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch = 0

  # Main Training and Evaluation Loop
  start_time, epoch_time, stepD, stepG = time.time(), AverageMeter(), 0, 0


  for epoch in range(start_epoch, args.epochs):

    need_time = convert_secs2time(epoch_time.val * (args.epochs-epoch), True)
    epoch_str = 'epoch-{:03d}-{:03d}'.format(epoch, args.epochs)
    logger.log('==>>{:s} [{:s}], [{:s}]'.format(time_string(), epoch_str, need_time))

    # log the results
    #logger.log('==>>{:s} Train [{:}] Average Loss = [G={:.6f} D={:.6f}]'.format(time_string(), epoch_str, Gloss, Dloss))
    data_time, iter_time, end = AverageMeter(), AverageMeter(), time.time()
    Glosses, Dlosses, visible_points = AverageMeter(), AverageMeter(), AverageMeter()
    PDlosses, SDlosses, Flosses, Tlosses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    for ibatch, ((PlainImages, StyleImages), target, mask, points, image_index, nopoints, cropped_size) in enumerate(train_loader):
      if ibatch == len(train_loader.dataset) // args.batch_size: break
      optimizerN.zero_grad()
      DetNet.train()
      # Preparation
      image_index = image_index.numpy().squeeze(1).tolist()
      target = target.cuda(non_blocking=True)
      batch_size, num_pts = PlainImages.size(0), args.num_pts
      visible_point_num   = float(np.sum(mask.numpy()[:,:-1,:,:])) / batch_size
      visible_points.update(visible_point_num, batch_size)
      nopoints    = nopoints.numpy().squeeze(1).tolist()
      annotated_num = batch_size - sum(nopoints)
      # measure data loading time
      mask = mask.cuda(non_blocking=True)
      data_time.update(time.time() - end)

      # batch_heatmaps is a list for stage-predictions, each element should be [Batch, C, H, W]
      batch_P_features, batch_P_heatmaps, batch_P_locs, batch_P_scos = DetNet(PlainImages)
      PDetLoss, P_stage_loss_value = compute_stage_loss(criterion, target, batch_P_heatmaps, mask)
      # Feature Loss
      #use_GAN_generate = random.random() < 0.5
      #if use_GAN_generate:
      #  noise_inputs = generate_noise(PlainImages)
      #  StyleImages = GenNet(noise_inputs).detach()
      #  SDetWeight  = 0.1
      #else:
      SDetWeight  = 1
      batch_S_features, batch_S_heatmaps, batch_S_locs, batch_S_scos = DetNet(StyleImages)
      SDetLoss, S_stage_loss_value = compute_stage_loss(criterion, target, batch_S_heatmaps, mask)
      INDEX  =  len(batch_P_features) // 2
      batch_P_features, batch_S_features = batch_P_features[INDEX], batch_S_features[INDEX]
      Feature_Similarity_Loss = mse_loss(batch_P_features, batch_S_features) * 0.05

      # G Loss
      y_real_, y_fake_ = torch.ones(batch_size, 1).cuda(), torch.zeros(batch_size, 1).cuda()
      D_fake = DisNet(batch_S_features).mean(dim=1).mean(dim=1)
      G_loss = MSE_loss(D_fake, y_real_) * 0.1

      totalLoss = PDetLoss + SDetLoss * SDetWeight + Feature_Similarity_Loss + G_loss
      totalLoss.backward()
      optimizerN.step()

      PDlosses.update(PDetLoss.item(), batch_size)
      SDlosses.update(SDetLoss.item(), batch_size)
      Glosses.update(G_loss.item(), batch_size)
      Flosses.update(Feature_Similarity_Loss.item(), batch_size)
      Tlosses.update(totalLoss.item(), batch_size)
    
      # D Loss
      optimizerD.zero_grad()
      #y_real_, y_fake_ = torch.ones(batch_size, 1).cuda(), torch.zeros(batch_size, 1).cuda()
      D_real = DisNet(batch_P_features.detach()).mean(dim=1).mean(dim=1)
      D_real_loss = MSE_loss(D_real, y_real_)
      D_fake = DisNet(batch_S_features.detach()).mean(dim=1).mean(dim=1)
      D_fake_loss = MSE_loss(D_fake, y_fake_)
      D_loss = (D_real_loss + D_fake_loss)
      D_loss.backward()
      optimizerD.step()
    
      Dlosses.update(D_loss.item(), batch_size)
      
      end = time.time()
      iter_time.update(time.time() - end)
      if ibatch % args.print_freq == 0 or (ibatch + 1) == len(train_loader):
        Tstr = time_string() + ' [ROBUST]: [{:}][{:03d}/{:03d}]'.format(epoch_str, ibatch, len(train_loader))
        Lstr = ' PD {PD.val:7.4f} ({PD.avg:7.4f}) SD {SD.val:7.4f} ({SD.avg:7.4f}) F:{FD.val:6.4f} ({FD.avg:6.4f})'.format(PD=PDlosses, SD=SDlosses, FD=Flosses)
        Lstr = Lstr + ' G {Gloss.val:7.4f} ({Gloss.avg:7.4f}) D {Dloss.val:7.4f} ({Dloss.avg:7.4f})'.format(Gloss=Glosses, Dloss=Dlosses)
        Lstr = Lstr + ' Loss : {:6.4f} ({:7.5f})'.format(Tlosses.val, Tlosses.avg)
        Estr = ' P{:} S{:}'.format(show_stage_loss(P_stage_loss_value), show_stage_loss(S_stage_loss_value))
        logger.log(Tstr + Lstr + Estr)

    if epoch % args.eval_freq == 0 or (epoch+1) == args.epochs:
      eval_results = eval_all(args, eval_loaders, DetNet, criterion, epoch_str, logger, None)

      # remember best prec@1 and save checkpoint
      save_path = save_checkpoint({
          'epoch': epoch,
          'args' : deepcopy(args),
          'state_dict_DET': DetNet.state_dict(),
          'optimizerN'    : optimizerN.state_dict(),
          'state_dict_Gen': GenNet.state_dict(),
          'state_dict_D'  : DisNet.state_dict(),
          'optimizerD'    : optimizerD.state_dict(),
          }, logger.path('model') / 'robust-{:}.pth'.format(epoch_str), logger)

      last_info = save_checkpoint({
          'epoch': epoch,
          'last_checkpoint': save_path,
          }, logger.last_info(), logger)

      logger.log('\n')

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  logger.close()

if __name__ == '__main__':
  args = obtain_StyleDet_args()
  main(args)
