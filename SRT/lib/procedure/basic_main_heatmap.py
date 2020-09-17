# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time, os, sys, torch, copy
from pathlib import Path
import numpy as np
# private lib
from xvision import Eval_Meta
from xvision import normalize_points, denormalize_points
from log_utils import AverageMeter, time_for_file, time_string, convert_secs2time
from .losses import compute_stage_loss, show_stage_loss
from .debug_utils import pro_debug_save


def basic_eval_all_heatmap(args, loaders, net, criterion, epoch_str, logger, opt_config, robust_transform):
  args = copy.deepcopy(args)
  logger.log('Basic-Eval-All evaluates {:} datasets, with {:}'.format(len(loaders), robust_transform))
  evalstrs, eval_metas = [], []
  for i, (loader, is_video) in enumerate(loaders):
    logger.log('==>>{:}, [{:}], evaluate the {:}/{:}-th dataset [{:}] : {:}'.format(time_string(), epoch_str, i, len(loaders), 'video' if is_video else 'image', loader.dataset))
    with torch.no_grad():
      xstr = epoch_str + ".{:}/{:}".format(i,len(loaders))
      eval_loss, eval_meta, eval_nme = basic_main_heatmap(args, loader, net, criterion, None, xstr, logger, opt_config, 'test')
      eval_metas.append( eval_meta )
      # test the robustness
      if robust_transform is not None:
        temp_transform = loader.dataset.transform
        loader.dataset.transform = copy.deepcopy(robust_transform)
        robust_error, robust_count = basic_eval_robust(args, loader, net, criterion, xstr, logger)
        loader.dataset.transform = temp_transform
        robust_str = 'robust error = {:7.2f}‰, with {:4.1f} points'.format(robust_error.avg*1000, robust_count.avg)
      else:
        robust_str = ''
    evalstrs.append('NME = {:6.3f} ,  '.format(eval_nme*100) + robust_str)
  evalstrs = '\n  ->'.join(['']+evalstrs)
  return evalstrs, eval_metas


# train function 
def basic_main_heatmap(args, loader, net, criterion, optimizer, epoch_str, logger, opt_config, mode):
  assert mode == 'train' or mode == 'test', 'invalid mode : {:}'.format(mode)
  args = copy.deepcopy(args)
  batch_time, data_time, forward_time, eval_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  visible_points, losses = AverageMeter(), AverageMeter()
  eval_meta = Eval_Meta()
  cpu = torch.device('cpu')

  if args.debug: save_dir = Path(args.save_path) / 'DEBUG' / ('{:}-'.format(mode) + epoch_str)
  else         : save_dir = None

  # switch to train mode
  if mode == 'train':
    logger.log('basic-main-V2 : training mode.')
    print_freq = args.print_freq
    net.train() ; criterion.train()
  else:
    logger.log('basic-main-V2 : evaluation mode.')
    print_freq = args.print_freq_eval
    net.eval()  ; criterion.eval()

  end = time.time()
  for i, (inputs, targets, masks, normpoints, transthetas, meanthetas, image_index, nopoints, shapes) in enumerate(loader):
    # inputs : Batch, Channel, Height, Width

    # information
    image_index = image_index.squeeze(1).tolist()
    (batch_size, C, H, W), num_pts = inputs.size(), args.num_pts
    visible_point_num   = float(np.sum(masks.numpy()[:,:-1,:,:])) / batch_size
    visible_points.update(visible_point_num, batch_size)
    annotated_num = batch_size - sum(nopoints)

    det_masks     = (1-nopoints).view(batch_size, 1, 1, 1) * masks
    det_masks     = det_masks.cuda(non_blocking=True)
    nopoints      = nopoints.squeeze(1).tolist()
    targets       = targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)

    # batch_heatmaps is a list for stage-predictions, each element should be [Batch, C, H, W]
    batch_features, batch_heatmaps, batch_locs, batch_scos = net(inputs)
    forward_time.update(time.time() - end)

    loss, each_stage_loss_value = compute_stage_loss(criterion, targets, batch_heatmaps, det_masks)

    # measure accuracy and record loss
    losses.update(loss.item(), batch_size)

    # compute gradient and do SGD step
    if mode == 'train': # training mode
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    eval_time.update(time.time() - end)

    with torch.no_grad():
      batch_locs, batch_scos = batch_locs.detach().to(cpu), batch_scos.detach().to(cpu)
      # evaluate the training data
      for ibatch, (imgidx, nopoint) in enumerate(zip(image_index, nopoints)):
        locations = batch_locs[ibatch,:-1,:]
        norm_locs = normalize_points((H,W), locations.transpose(1,0))
        norm_locs = torch.cat((norm_locs, torch.ones(1, num_pts)), dim=0)
        transtheta = transthetas[ibatch][:2,:]
        norm_locs = torch.mm(transtheta, norm_locs)
        real_locs = denormalize_points(shapes[ibatch].tolist(), norm_locs)
        real_locs = torch.cat((real_locs, batch_scos[ibatch, :-1].view(1, -1)), dim=0)
        #real_locs = torch.cat((real_locs, torch.ones(1, num_pts)), dim=0)
        image_path = loader.dataset.datas[imgidx]
        normDistce = loader.dataset.NormDistances[imgidx]

        if nopoint == 1: xpoints = None
        else           : xpoints = loader.dataset.labels[imgidx].get_points().numpy()
        eval_meta.append(real_locs.numpy(), xpoints, image_path, normDistce)
        if save_dir:
          pro_debug_save(save_dir, Path(image_path).name, inputs[ibatch], targets[ibatch], normpoints[ibatch], meanthetas[ibatch], batch_heatmaps[-1][ibatch], args.tensor2imageF)

    # measure elapsed time
    batch_time.update(time.time() - end)
    last_time = convert_secs2time(batch_time.avg * (len(loader)-i-1), True)
    end = time.time()

    if i % print_freq == 0 or i+1 == len(loader):
      logger.log(' -->>[{:}]: [{:}][{:03d}/{:03d}] '
                'Time {batch_time.val:4.2f} ({batch_time.avg:4.2f}) '
                'Data {data_time.val:4.2f} ({data_time.avg:4.2f}) '
                'Forward {forward_time.val:4.2f} ({forward_time.avg:4.2f}) '
                'Loss {loss.val:7.4f} ({loss.avg:7.4f})  '.format(
                    mode, epoch_str, i, len(loader), batch_time=batch_time,
                    data_time=data_time, forward_time=forward_time, loss=losses)
                  + last_time + show_stage_loss(each_stage_loss_value) \
                  + ' In={:} Tar={:}'.format(list(inputs.size()), list(targets.size())) \
                  + ' Vis-PTS : {:2d} ({:.1f})'.format(int(visible_points.val), visible_points.avg))
  nme, _, _ = eval_meta.compute_mse(loader.dataset.dataset_name, logger)
  return losses.avg, eval_meta, nme
