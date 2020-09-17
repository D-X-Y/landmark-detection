# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time, os, sys, torch, copy, numbers
from pathlib import Path
import numpy as np
# private lib
from xvision import Eval_Meta
from xvision import normalize_points, denormalize_points
from log_utils import AverageMeter, time_for_file, time_string, convert_secs2time
from .losses import compute_stage_loss, show_stage_loss
from .temporal_loss_regression import calculate_temporal_loss
from .debug_utils import pro_debug_save


# SBR train function 
def sbr_main_regression(args, loader, net, criterion, optimizer, epoch_str, logger, opt_config, sbr_config, use_sbr, mode):
  assert mode == 'train' or mode == 'test', 'invalid mode : {:}'.format(mode)
  args = copy.deepcopy(args)
  batch_time, data_time, forward_time, eval_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  visible_points, DetLosses, TotalLosses, TemporalLosses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  alk_points = AverageMeter()
  annotate_index = loader.dataset.video_L
  eval_meta = Eval_Meta()
  cpu = torch.device('cpu')

  if args.debug: save_dir = Path(args.save_path) / 'DEBUG' / ('{:}-'.format(mode) + epoch_str)
  else         : save_dir = None

  # switch to train mode
  if mode == 'train':
    logger.log('Temporal-Main-Regression: training : {:} .. SBR={:}'.format(sbr_config, use_sbr))
    print_freq = args.print_freq
    net.train() ; criterion.train()
  else:
    logger.log('Temporal-Main-Regression : evaluation mode.')
    print_freq = args.print_freq_eval
    net.eval()  ; criterion.eval()

  i_batch_size, v_batch_size = args.i_batch_size, args.v_batch_size
  end = time.time()
  for i, (frames, Fflows, Bflows, targets, masks, normpoints, transthetas, meanthetas, image_index, nopoints, shapes, is_images) in enumerate(loader):
    # frames : IBatch+VBatch, Frame, Channel, Height, Width
    # Fflows : IBatch+VBatch, Frame-1, Height, Width, 2
    # Bflows : IBatch+VBatch, Frame-1, Height, Width, 2

    # information
    image_index = image_index.squeeze(1).tolist()
    (batch_size, frame_length, C, H, W), num_pts = frames.size(), args.num_pts
    visible_point_num   = float(np.sum(masks.numpy()[:,:-1,:,:])) / batch_size
    visible_points.update(visible_point_num, batch_size)
    assert is_images[:i_batch_size].sum().item() == i_batch_size, '{:} vs. {:}'.format(is_images, i_batch_size)
    assert is_images[i_batch_size:].sum().item() == 0, '{:} vs. {:}'.format(is_images, v_batch_size)

    normpoints    = normpoints.permute(0, 2, 1)
    target_points = normpoints[:, :, :2].contiguous().cuda(non_blocking=True)
    target_scores = normpoints[:, :, 2:].contiguous().cuda(non_blocking=True)
    det_masks     = (1-nopoints).view(batch_size, 1, 1) * masks[:, :num_pts].contiguous().view(batch_size, num_pts, 1)
    have_det_loss = det_masks.sum().item() > 0
    det_masks     = det_masks.cuda(non_blocking=True)
    nopoints      = nopoints.squeeze(1).tolist()

    # measure data loading time
    data_time.update(time.time() - end)

    # batch_heatmaps is a list for stage-predictions, each element should be [Batch, Sequence, PTS, H/Down, W/Down]
    batch_locs, batch_past2now, batch_future2now, batch_FBcheck = net(frames, Fflows, Bflows, is_images)
    forward_time.update(time.time() - end)
  
    # detection loss
    if have_det_loss:
      det_loss = criterion(batch_locs[:,annotate_index], target_points, det_masks)
      DetLosses.update(det_loss.item(), batch_size)
    else:
      det_loss = 0

    # temporal loss
    if use_sbr:
      video_batch_locs = batch_locs[i_batch_size:, :]
      video_past2now, video_future2now, video_FBcheck = batch_past2now[i_batch_size:], batch_future2now[i_batch_size:], batch_FBcheck[i_batch_size:]
      video_mask = masks[i_batch_size:, :-1].contiguous().cuda(non_blocking=True)
      sbr_loss, available_nums = calculate_temporal_loss(criterion, video_batch_locs, video_past2now, video_future2now, video_FBcheck, video_mask, sbr_config)
      alk_points.update(float(available_nums)/v_batch_size, v_batch_size)
      if available_nums > sbr_config.available_thresh:
        TemporalLosses.update(sbr_loss.item(), v_batch_size)
      else:
        sbr_loss = 0
    else:
      sbr_loss = 0

    # measure accuracy and record loss
    #if sbr_config.weight != 0: total_loss = det_loss + sbr_loss * sbr_config.weight
    #else                     : total_loss = det_loss
    if use_sbr: total_loss = det_loss + sbr_loss * sbr_config.weight
    else      : total_loss = det_loss
    if isinstance(total_loss, numbers.Number):
      warnings.warn('The {:}-th iteration has no detection loss and no lk loss'.format(i))
    else:
      TotalLosses.update(total_loss.item(), batch_size)
      # compute gradient and do SGD step
      if mode == 'train': # training mode
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    eval_time.update(time.time() - end)

    with torch.no_grad():
      batch_locs = batch_locs.detach().to(cpu)[:, annotate_index]
      # evaluate the training data
      for ibatch, (imgidx, nopoint) in enumerate(zip(image_index, nopoints)):
        if nopoint == 1: continue
        norm_locs  = torch.cat((batch_locs[ibatch].permute(1,0), torch.ones(1, num_pts)), dim=0)
        transtheta = transthetas[ibatch][:2,:]
        norm_locs = torch.mm(transtheta, norm_locs)
        real_locs = denormalize_points(shapes[ibatch].tolist(), norm_locs)
        real_locs = torch.cat((real_locs, torch.ones(1, num_pts)), dim=0)
  
        image_path = loader.dataset.datas[imgidx][annotate_index]
        normDistce = loader.dataset.NormDistances[imgidx]
        xpoints    = loader.dataset.labels[imgidx].get_points()
        eval_meta.append(real_locs.numpy(), xpoints.numpy(), image_path, normDistce)
        if save_dir:
          pro_debug_save(save_dir, Path(image_path).name, frames[ibatch, annotate_index], targets[ibatch], normpoints[ibatch], meanthetas[ibatch], batch_heatmaps[-1][ibatch, annotate_index], args.tensor2imageF)

    # measure elapsed time
    batch_time.update(time.time() - end)
    last_time = convert_secs2time(batch_time.avg * (len(loader)-i-1), True)
    end = time.time()

    if i % print_freq == 0 or i+1 == len(loader):
      logger.log(' -->>[{:}]: [{:}][{:03d}/{:03d}] '
                'Time {batch_time.val:4.2f} ({batch_time.avg:4.2f}) '
                'Data {data_time.val:4.2f} ({data_time.avg:4.2f}) '
                'F-time {forward_time.val:4.2f} ({forward_time.avg:4.2f}) '
                'Det {dloss.val:7.4f} ({dloss.avg:7.4f}) '
                'SBR {sloss.val:7.4f} ({sloss.avg:7.4f}) '
                'Loss {loss.val:7.4f} ({loss.avg:7.4f})  '.format(
                    mode, epoch_str, i, len(loader), batch_time=batch_time,
                    data_time=data_time, forward_time=forward_time, \
                    dloss=DetLosses, sloss=TemporalLosses, loss=TotalLosses)
                  + last_time \
                  + ' I={:}'.format(list(frames.size())) \
                  + ' Vis-PTS : {:2d} ({:.1f})'.format(int(visible_points.val), visible_points.avg) \
                  + ' Ava-PTS : {:.1f} ({:.1f})'.format(alk_points.val, alk_points.avg))
      if args.debug:
        logger.log('  -->>Indexes : {:}'.format(image_index))
  nme, _, _ = eval_meta.compute_mse(loader.dataset.dataset_name, logger)
  return TotalLosses.avg, nme
