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
from .temporal_loss_heatmap import calculate_temporal_loss
from .debug_utils import pro_debug_save


# SBR train function 
def sbr_main_heatmap(args, loader, net, criterion, optimizer, epoch_str, logger, opt_config, sbr_config, use_sbr, mode):
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
    print_freq = args.print_freq
    logger.log('Temporal-Main-Heatmap : training mode : {:} .. SBR={:} .. Freq={:}'.format(sbr_config, use_sbr, print_freq))
    net.train() ; criterion.train()
  else:
    print_freq = args.print_freq_eval
    logger.log('Temporal-Main-Heatmap : evaluation mode .. Freq={:}.'.format(print_freq))
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
    targets       = targets.cuda(non_blocking=True)
    det_masks     = (1-nopoints).view(batch_size, 1, 1, 1) * masks
    have_det_loss = det_masks.sum().item() > 0
    det_masks     = det_masks.cuda(non_blocking=True)
    nopoints      = nopoints.squeeze(1).tolist()

    # measure data loading time
    data_time.update(time.time() - end)

    # batch_heatmaps is a list for stage-predictions, each element should be [Batch, Sequence, PTS, H/Down, W/Down]
    batch_heatmaps, batch_locs, batch_scos, batch_past2now, batch_future2now, batch_FBcheck = net(frames, Fflows, Bflows, is_images)
    annot_heatmaps = [x[:, annotate_index] for x in batch_heatmaps]
    forward_time.update(time.time() - end)
  
    # detection loss
    if have_det_loss:
      det_loss, each_stage_loss_value = compute_stage_loss(criterion, targets, annot_heatmaps, det_masks)
      DetLosses.update(det_loss.item(), batch_size)
      each_stage_loss_value = show_stage_loss(each_stage_loss_value)
    else:
      det_loss, each_stage_loss_value = 0, 'no-det-loss'

    # temporal loss
    if use_sbr:
      video_batch_locs = batch_locs[i_batch_size:, :, :num_pts]
      video_past2now, video_future2now = batch_past2now[i_batch_size:, :, :num_pts], batch_future2now[i_batch_size:, :, :num_pts]
      video_FBcheck    = batch_FBcheck[i_batch_size:, :num_pts]
      video_mask       = masks[i_batch_size:, :num_pts].contiguous().cuda(non_blocking=True)
      video_heatmaps   = [ x[i_batch_size:, :, :num_pts] for x in batch_heatmaps ]
      sbr_loss, available_nums, loss_string  = calculate_temporal_loss(criterion, video_heatmaps, video_batch_locs, video_past2now, video_future2now, video_FBcheck, video_mask, sbr_config)
      alk_points.update(float(available_nums)/v_batch_size, v_batch_size)
      if available_nums > sbr_config.available_thresh:
        TemporalLosses.update(sbr_loss.item(), v_batch_size)
      else:
        sbr_loss, loss_string = 0, 'non-sbr-loss'
    else:
      sbr_loss, loss_string = 0, 'non-sbr-loss'

    # measure accuracy and record loss
    total_loss = det_loss + sbr_loss * sbr_config.weight
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
      batch_locs = batch_locs.detach().to(cpu)[:, annotate_index, :num_pts]
      batch_scos = batch_scos.detach().to(cpu)[:, annotate_index, :num_pts]
      # evaluate the training data
      for ibatch, (imgidx, nopoint) in enumerate(zip(image_index, nopoints)):
        if nopoint == 1: continue
        #locations = batch_locs[ibatch,annotate_index,:-1,:]
        #norm_locs = normalize_points((H,W), locations.transpose(1,0))
        #norm_locs = torch.cat((norm_locs, torch.ones(1, num_pts)), dim=0)
        norm_locs  = torch.cat((batch_locs[ibatch].permute(1,0), torch.ones(1, num_pts)), dim=0)
        transtheta = transthetas[ibatch][:2,:]
        norm_locs = torch.mm(transtheta, norm_locs)
        real_locs = denormalize_points(shapes[ibatch].tolist(), norm_locs)
        real_locs = torch.cat((real_locs, batch_scos[ibatch].view(1, num_pts)), dim=0)
  
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
                'SBR {sloss.val:7.5f} ({sloss.avg:7.5f}) '
                'Loss {loss.val:7.4f} ({loss.avg:7.4f})  '.format(
                    mode, epoch_str, i, len(loader), batch_time=batch_time,
                    data_time=data_time, forward_time=forward_time, \
                    dloss=DetLosses, sloss=TemporalLosses, loss=TotalLosses)
                  + last_time + each_stage_loss_value \
                  + ' In={:} Tar={:}'.format(list(frames.size()), list(targets.size())) \
                  + ' Vis-PTS : {:2d} ({:.1f})'.format(int(visible_points.val), visible_points.avg) \
                  + ' Ava-PTS : {:.1f} ({:.1f})'.format(alk_points.val, alk_points.avg) + loss_string)
      if args.debug:
        logger.log('  -->>Indexes : {:}'.format(image_index))
  nme, _, _ = eval_meta.compute_mse(loader.dataset.dataset_name, logger)
  return TotalLosses.avg, nme
