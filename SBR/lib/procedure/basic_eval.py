# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time, os, sys, numpy as np
import torch
from copy import deepcopy
from pathlib import Path
from xvision import Eval_Meta
from log_utils import AverageMeter, time_for_file, time_string, convert_secs2time
from .losses import compute_stage_loss, show_stage_loss

def basic_eval_all(args, loaders, net, criterion, epoch_str, logger, opt_config):
  args = deepcopy(args)
  logger.log('Basic-Eval-All evaluates {:} dataset'.format(len(loaders)))
  nmes = []
  for i, (loader, is_video) in enumerate(loaders):
    logger.log('==>>{:}, [{:}], evaluate the {:}/{:}-th dataset [{:}] : {:}'.format(time_string(), epoch_str, i, len(loaders), 'video' if is_video else 'image', loader.dataset))
    with torch.no_grad():
      eval_loss, eval_meta = basic_eval(args, loader, net, criterion, epoch_str+"::{:}/{:}".format(i,len(loaders)), logger, opt_config)
    nme, _, _ = eval_meta.compute_mse(logger)
    meta_path = logger.path('meta') / 'eval-{:}-{:02d}-{:02d}.pth'.format(epoch_str, i, len(loaders))
    eval_meta.save(meta_path)
    nmes.append(nme*100)
  return ', '.join(['{:.2f}'.format(x) for x in nmes])
  

def basic_eval(args, loader, net, criterion, epoch_str, logger, opt_config):
  batch_time, data_time, forward_time, eval_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  visible_points, losses = AverageMeter(), AverageMeter()
  eval_meta = Eval_Meta()
  cpu = torch.device('cpu')

  # switch to train mode
  net.eval()
  criterion.eval()

  end = time.time()
  for i, (inputs, target, mask, points, image_index, nopoints, cropped_size) in enumerate(loader):
    # inputs : Batch, Channel, Height, Width

    target = target.cuda(non_blocking=True)

    image_index = image_index.numpy().squeeze(1).tolist()
    batch_size, num_pts = inputs.size(0), args.num_pts
    visible_point_num   = float(np.sum(mask.numpy()[:,:-1,:,:])) / batch_size
    visible_points.update(visible_point_num, batch_size)
    nopoints    = nopoints.numpy().squeeze(1).tolist()
    annotated_num = batch_size - sum(nopoints)

    # measure data loading time
    mask = mask.cuda(non_blocking=True)
    data_time.update(time.time() - end)

    # batch_heatmaps is a list for stage-predictions, each element should be [Batch, C, H, W]
    batch_heatmaps, batch_locs, batch_scos = net(inputs)
    forward_time.update(time.time() - end)

    if annotated_num > 0:
      loss, each_stage_loss_value = compute_stage_loss(criterion, target, batch_heatmaps, mask)
      if opt_config.lossnorm:
        loss, each_stage_loss_value = loss / annotated_num, [x/annotated_num for x in each_stage_loss_value]
      each_stage_loss_value = show_stage_loss(each_stage_loss_value)
      # measure accuracy and record loss
      losses.update(loss.item(), batch_size)
    else:
      loss, each_stage_loss_value = 0, 'no-det-loss'

    eval_time.update(time.time() - end)

    np_batch_locs, np_batch_scos = batch_locs.to(cpu).numpy(), batch_scos.to(cpu).numpy()
    cropped_size = cropped_size.numpy()
    # evaluate the training data
    for ibatch, (imgidx, nopoint) in enumerate(zip(image_index, nopoints)):
      #if nopoint == 1: continue
      locations, scores = np_batch_locs[ibatch,:-1,:], np.expand_dims(np_batch_scos[ibatch,:-1], -1)
      xpoints = loader.dataset.labels[imgidx].get_points()
      assert cropped_size[ibatch,0] > 0 and cropped_size[ibatch,1] > 0, 'The ibatch={:}, imgidx={:} is not right.'.format(ibatch, imgidx, cropped_size[ibatch])
      scale_h, scale_w = cropped_size[ibatch,0] * 1. / inputs.size(-2) , cropped_size[ibatch,1] * 1. / inputs.size(-1)
      locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[ibatch,2], locations[:, 1] * scale_h + cropped_size[ibatch,3]
      assert xpoints.shape[1] == num_pts and locations.shape[0] == num_pts and scores.shape[0] == num_pts, 'The number of points is {} vs {} vs {} vs {}'.format(num_pts, xpoints.shape, locations.shape, scores.shape)
      # recover the original resolution
      prediction = np.concatenate((locations, scores), axis=1).transpose(1,0)
      image_path = loader.dataset.datas[imgidx]
      face_size  = loader.dataset.face_sizes[imgidx]
      if nopoint == 1:
        eval_meta.append(prediction, None, image_path, face_size)
      else:
        eval_meta.append(prediction, xpoints, image_path, face_size)

    # measure elapsed time
    batch_time.update(time.time() - end)
    last_time = convert_secs2time(batch_time.avg * (len(loader)-i-1), True)
    end = time.time()

    if i % (args.print_freq) == 0 or i+1 == len(loader):
      logger.log(' -->>[Eval]: [{:}][{:03d}/{:03d}] '
                'Time {batch_time.val:4.2f} ({batch_time.avg:4.2f}) '
                'Data {data_time.val:4.2f} ({data_time.avg:4.2f}) '
                'Forward {forward_time.val:4.2f} ({forward_time.avg:4.2f}) '
                'Loss {loss.val:7.4f} ({loss.avg:7.4f})  '.format(
                    epoch_str, i, len(loader), batch_time=batch_time,
                    data_time=data_time, forward_time=forward_time, loss=losses)
                  + last_time + each_stage_loss_value \
                  + ' In={:} Tar={:}'.format(list(inputs.size()), list(target.size())) \
                  + ' Vis-PTS : {:2d} ({:.1f})'.format(int(visible_points.val), visible_points.avg))
  return losses.avg, eval_meta
