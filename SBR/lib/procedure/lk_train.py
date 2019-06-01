# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time, os, numpy as np
import torch
import numbers, warnings
from copy import deepcopy
from pathlib import Path
from log_utils import AverageMeter, time_for_file, convert_secs2time
from .losses import compute_stage_loss, show_stage_loss
from .lk_loss import lk_target_loss

# train function (forward, backward, update)
def lk_train(args, loader, net, criterion, optimizer, epoch_str, logger, opt_config, lk_config, use_lk):
  args = deepcopy(args)
  batch_time, data_time, forward_time, eval_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  visible_points, detlosses, lklosses = AverageMeter(), AverageMeter(), AverageMeter()
  alk_points, losses = AverageMeter(), AverageMeter()
  cpu = torch.device('cpu')
  
  annotate_index = loader.dataset.center_idx

  # switch to train mode
  net.train()
  criterion.train()

  end = time.time()
  for i, (inputs, target, mask, points, image_index, nopoints, video_or_not, cropped_size) in enumerate(loader):
    # inputs : Batch, Sequence Channel, Height, Width

    target = target.cuda(non_blocking=True)

    image_index = image_index.numpy().squeeze(1).tolist()
    batch_size, sequence, num_pts = inputs.size(0), inputs.size(1), args.num_pts
    mask_np = mask.numpy().squeeze(-1).squeeze(-1)
    visible_point_num   = float(np.sum(mask.numpy()[:,:-1,:,:])) / batch_size
    visible_points.update(visible_point_num, batch_size)
    nopoints    = nopoints.numpy().squeeze(1).tolist()
    video_or_not= video_or_not.numpy().squeeze(1).tolist()
    annotated_num = batch_size - sum(nopoints)

    # measure data loading time
    mask = mask.cuda(non_blocking=True)
    data_time.update(time.time() - end)

    # batch_heatmaps is a list for stage-predictions, each element should be [Batch, Sequence, PTS, H/Down, W/Down]
    batch_heatmaps, batch_locs, batch_scos, batch_next, batch_fback, batch_back = net(inputs)
    annot_heatmaps = [x[:, annotate_index] for x in batch_heatmaps]
    forward_time.update(time.time() - end)

    if annotated_num > 0:
      # have the detection loss
      detloss, each_stage_loss_value = compute_stage_loss(criterion, target, annot_heatmaps, mask)
      if opt_config.lossnorm:
        detloss, each_stage_loss_value = detloss / annotated_num / 2, [x/annotated_num/2 for x in each_stage_loss_value]
      # measure accuracy and record loss
      detlosses.update(detloss.item(), batch_size)
      each_stage_loss_value = show_stage_loss(each_stage_loss_value)
    else:
      detloss, each_stage_loss_value = 0, 'no-det-loss'

    if use_lk:
      lkloss, avaliable = lk_target_loss(batch_locs, batch_scos, batch_next, batch_fback, batch_back, lk_config, video_or_not, mask_np, nopoints)
      if lkloss is not None:
        lklosses.update(lkloss.item(), avaliable)
      else: lkloss = 0
      alk_points.update(float(avaliable)/batch_size, batch_size)
    else  : lkloss = 0
     
    loss = detloss + lkloss * lk_config.weight

    if isinstance(loss, numbers.Number):
      warnings.warn('The {:}-th iteration has no detection loss and no lk loss'.format(i))
    else:
      losses.update(loss.item(), batch_size)
      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    eval_time.update(time.time() - end)

    # measure elapsed time
    batch_time.update(time.time() - end)
    last_time = convert_secs2time(batch_time.avg * (len(loader)-i-1), True)
    end = time.time()

    if i % args.print_freq == 0 or i+1 == len(loader):
      logger.log(' -->>[Train]: [{:}][{:03d}/{:03d}] '
                'Time {batch_time.val:4.2f} ({batch_time.avg:4.2f}) '
                'Data {data_time.val:4.2f} ({data_time.avg:4.2f}) '
                'Forward {forward_time.val:4.2f} ({forward_time.avg:4.2f}) '
                'Loss {loss.val:7.4f} ({loss.avg:7.4f}) [LK={lk.val:7.4f} ({lk.avg:7.4f})] '.format(
                    epoch_str, i, len(loader), batch_time=batch_time,
                    data_time=data_time, forward_time=forward_time, loss=losses, lk=lklosses)
                  + each_stage_loss_value + ' ' + last_time \
                  + ' Vis-PTS : {:2d} ({:.1f})'.format(int(visible_points.val), visible_points.avg) \
                  + ' Ava-PTS : {:.1f} ({:.1f})'.format(alk_points.val, alk_points.avg))

  return losses.avg
