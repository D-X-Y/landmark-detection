# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division

import sys, time, torch, random, argparse, PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from config_utils import obtain_style_args
from procedure import prepare_seed, save_checkpoint, generate_noise, style_eval_plain
from datasets import GeneralDataset as Dataset
from xvision import transforms, style_trans
from log_utils import Logger, AverageMeter, time_for_file, convert_secs2time, time_string
from config_utils import load_configure
from models import obtain_GAN

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
                                        transforms.TrainScale2WH((args.crop_width, args.crop_height)), style_trans.AugStyle(), \
                                        transforms.ToTensor(), normalize])
  assert (args.scale_min+args.scale_max) / 2 == args.scale_eval, 'The scale is not ok : {},{} vs {}'.format(args.scale_min, args.scale_max, args.scale_eval)
  recover = transforms.ToPILImageWithNorm(normalize)
  
  # Model Configure Load
  args.sigma   = args.sigma * args.scale_eval
  logger.log('Real Sigma : {:}'.format(args.sigma))

  # Training Dataset
  train_data   = Dataset(train_transform, args.sigma, 8, args.heatmap_type, args.data_indicator)
  train_data.load_list(args.train_lists, args.num_pts, True)
  logger.log('Training Data : {:}'.format(train_data))
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

  if args.eval_lists is not None:
    eval_data = Dataset(eval_transform, args.sigma, 8, args.heatmap_type, args.data_indicator)
    for idx, eval_list in enumerate(args.eval_lists):
      if idx == 0: eval_data.load_list(eval_list, args.num_pts, True)
      else:        eval_data.load_list(eval_list, args.num_pts, False)
    logger.log('Evaluation Data : {:}'.format(eval_data))
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
  else: eval_loader = None

  # Evaluation Dataloader
  netG, netD = obtain_GAN(4, args.gan_norm)
  logger.log('Face Generator : \n{:}'.format(netG))
  logger.log('Face Discriminator : \n{:}'.format(netD))
  logger.log('Train : {:}\n Test : {:}\nArguments : {:}'.format(train_loader, eval_loader, args))

  MSE_loss   = torch.nn.MSELoss().cuda()
  optimizerD = torch.optim.Adam(netD.parameters(), lr=args.LR_D, betas=(0.5, 0.9), amsgrad=args.amsgrad>0)
  optimizerG = torch.optim.Adam(netG.parameters(), lr=args.LR_G, betas=(0.5, 0.9), amsgrad=args.amsgrad>0)

  netG, netD = torch.nn.DataParallel(netG).cuda(), torch.nn.DataParallel(netD).cuda()

  last_info = logger.last_info()
  if last_info.exists():
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info = torch.load(last_info)
    start_epoch = last_info['epoch'] + 1
    checkpoint  = torch.load(last_info['last_checkpoint'])
    assert last_info['epoch'] == checkpoint['epoch'], 'Last-Info is not right {:} vs {:}'.format(last_info, checkpoint['epoch'])
    netG.load_state_dict(checkpoint['state_dict_G'])
    netD.load_state_dict(checkpoint['state_dict_D'])
    optimizerG.load_state_dict(checkpoint['optimizerG'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])
    logger.log("=> load-ok checkpoint '{:}' (epoch {:}) done" .format(logger.last_info(), checkpoint['epoch']))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch = 0

  # Main Training and Evaluation Loop
  start_time, epoch_time, stepD, stepG = time.time(), AverageMeter(), 0, 0
  Glosses, Dlosses = AverageMeter(), AverageMeter()
  for epoch in range(start_epoch, args.epochs):

    need_time = convert_secs2time(epoch_time.val * (args.epochs-epoch), True)
    epoch_str = 'epoch-{:03d}-{:03d}'.format(epoch, args.epochs)
    logger.log('==>>{:s} [{:s}], [{:s}], critic-iters={:}, amsgrad={:}'.format(time_string(), epoch_str, need_time, args.critic_iters, args.amsgrad))

    y_real_, y_fake_ = torch.ones(args.batch_size, 1).cuda(), torch.zeros(args.batch_size, 1).cuda()

    netD.train() ; netG.train()
    for _iter, ((PlainImages, StyleImages), target, mask, points, image_index, nopoints, cropped_size) in enumerate(train_loader):
      if _iter == len(train_loader.dataset) // args.batch_size: break
      PlainImages, StyleImages = PlainImages.cuda(), StyleImages.cuda()

      Isize = list(PlainImages.size())

      # Update D Network
      optimizerD.zero_grad()
      D_real = netD(StyleImages).mean(dim=1).mean(dim=1)
      D_real_loss = MSE_loss(D_real, y_real_)
      
      noise_inputs = generate_noise(PlainImages)
      G_ = netG(noise_inputs)
      D_fake = netD(G_).mean(dim=1).mean(dim=1)
      D_fake_loss = MSE_loss(D_fake, y_fake_)

      D_loss = D_real_loss + D_fake_loss
      D_loss.backward()
      optimizerD.step()

      logger.scalar_summary('D-loss', D_loss.item(), stepD)
      stepD = stepD + 1
      Dlosses.update(D_loss.item(), Isize[0])

      if (stepD+1) % args.critic_iters == 0:
        optimizerG.zero_grad()
        noise_inputs = generate_noise(PlainImages)
        G_ = netG(noise_inputs)
        D_fake = netD(G_).mean(dim=1).mean(dim=1)
        G_loss = MSE_loss(D_fake, y_real_)
        G_loss.backward()
        optimizerG.step()

        logger.scalar_summary('G-loss', G_loss.item(), stepG)
        stepG = stepG + 1
        Glosses.update(G_loss.item(), Isize[0])

      if _iter % args.print_freq == 0:
        logger.log(time_string() + ' [Train-LSGAN]: [{:}][{:03d}/{:03d}] Gloss {Gloss.val:7.4f} ({Gloss.avg:7.4f}) Dloss {Dloss.val:7.4f} ({Dloss.avg:7.4f})'.format(
                     epoch_str, _iter, len(train_loader), Gloss=Glosses, Dloss=Dlosses)
                   + ' I={:}'.format(Isize) + ' G-Step={:}, D-Step={:}'.format(stepG, stepD))
    # log the results    
    #logger.log('==>>{:s} Train [{:}] Average Loss = [G={:.6f} D={:.6f}]'.format(time_string(), epoch_str, Gloss, Dloss))

    if epoch % args.eval_freq == 0 or (epoch+1) == args.epochs:
      style_eval_plain(args, train_loader, netG, recover, logger.path('meta') / (epoch_str+'-train'), logger)
      if eval_loader is not None:
        style_eval_plain(args, eval_loader, netG, recover, logger.path('meta') / (epoch_str+'-eval'), logger)

      # remember best prec@1 and save checkpoint
      save_path = save_checkpoint({
          'epoch': epoch,
          'args' : deepcopy(args),
          'state_dict_G': netG.state_dict(),
          'optimizerG'  : optimizerG.state_dict(),
          'state_dict_D': netD.state_dict(),
          'optimizerD'  : optimizerD.state_dict(),
          }, logger.path('model') / 'style-{:}.pth'.format(epoch_str), logger)

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
  args = obtain_style_args()
  main(args)
