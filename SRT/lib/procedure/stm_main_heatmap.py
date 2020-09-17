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
from xvision import normalize_points, denormalize_points, denormalize_points_batch, denormalize_L, normalize_L
from log_utils import AverageMeter, time_for_file, time_string, convert_secs2time
from multiview import ProjectKRT_Batch, TriangulateDLT_BatchCam
from .losses import compute_stage_loss, show_stage_loss
from .temporal_loss_heatmap import calculate_temporal_loss
from .multiview_loss_heatmap import calculate_multiview_loss
from .debug_utils import pro_debug_save, multiview_debug_save, multiview_debug_save_v2


def convert_theta(mv_preal_locs_ori, MV_Thetas_G):
  ones   = torch.ones(mv_preal_locs_ori.size(0), mv_preal_locs_ori.size(1), mv_preal_locs_ori.size(2), 1, device=mv_preal_locs_ori.device)
  points = torch.cat((mv_preal_locs_ori, ones), dim=-1)
  #trans_points, _ = torch.gesv(points.transpose(-1,-2), MV_Thetas_G)
  trans_points, _ = torch.solve(points.transpose(-1,-2), MV_Thetas_G)
  ok_points = trans_points.transpose(-1,-2)
  return ok_points[..., :2]


# STM train function 
def stm_main_heatmap(args, loader, net, criterion, optimizer, epoch_str, logger, opt_config, stm_config, use_stm, mode):
  assert mode == 'train' or mode == 'test', 'invalid mode : {:}'.format(mode)
  args = copy.deepcopy(args)
  batch_time, data_time, forward_time, eval_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  visible_points, DetLosses, TemporalLosses, MultiviewLosses, TotalLosses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  alk_points, a3d_points = AverageMeter(), AverageMeter()
  annotate_index = loader.dataset.video_L
  eval_meta = Eval_Meta()
  cpu = torch.device('cpu')

  if args.debug: save_dir = Path(args.save_path) / 'DEBUG' / ('{:}-'.format(mode) + epoch_str)
  else         : save_dir = None

  # switch to train mode
  if mode == 'train':
    logger.log('STM-Main-REG : training : {:} .. STM = {:}'.format(stm_config, use_stm))
    print_freq = args.print_freq
    net.train() ; criterion.train()
  else:
    logger.log('STM-Main-REG : evaluation mode.')
    print_freq = args.print_freq_eval
    net.eval()  ; criterion.eval()

  i_batch_size, v_batch_size, m_batch_size = args.i_batch_size, args.v_batch_size, args.m_batch_size
  iv_size = i_batch_size + v_batch_size
  end = time.time()
  for i, (frames, Fflows, Bflows, targets, masks, normpoints, transthetas, MV_Tensors, MV_Thetas, MV_Shapes, MV_KRT, torch_is_3D, torch_is_images \
            , image_index, nopoints, shapes, MultiViewPaths) in enumerate(loader):
    # frames : IBatch+VBatch+MBatch, Frame, Channel, Height, Width
    # Fflows : IBatch+VBatch+MBatch, Frame-1, Height, Width, 2
    # Bflows : IBatch+VBatch+MBatch, Frame-1, Height, Width, 2

    # information
    MV_Mask = masks[iv_size:]
    frames, Fflows, Bflows, targets, masks, normpoints, transthetas = frames[:iv_size], Fflows[:iv_size], Bflows[:iv_size], targets[:iv_size], masks[:iv_size], normpoints[:iv_size], transthetas[:iv_size]
    nopoints, shapes, torch_is_images = nopoints[:iv_size], shapes[:iv_size], torch_is_images[:iv_size]
    MV_Tensors, MV_Thetas, MV_Shapes, MV_KRT, torch_is_3D = \
      MV_Tensors[iv_size:], MV_Thetas[iv_size:], MV_Shapes[iv_size:], MV_KRT[iv_size:], torch_is_3D[iv_size:]
    assert torch.sum(torch_is_images[:i_batch_size]).item() == i_batch_size, 'Image Check Fail : {:} vs. {:}'.format(torch_is_images[:i_batch_size], i_batch_size)
    assert v_batch_size == 0 or torch.sum(torch_is_images[i_batch_size:]).item() == 0           , 'Video Check Fail : {:} vs. {:}'.format(torch_is_images[i_batch_size:], v_batch_size)
    assert torch_is_3D.sum().item() == m_batch_size                        , 'Multiview Check Fail : {:} vs. {:}'.format(torch_is_3D, m_batch_size)
    image_index = image_index.squeeze(1).tolist()
    (batch_size, frame_length, C, H, W), num_pts, num_views = frames.size(), args.num_pts, stm_config.max_views
    visible_point_num   = float(np.sum(masks.numpy()[:,:-1,:,:])) / batch_size
    visible_points.update(visible_point_num, batch_size)

    normpoints    = normpoints.permute(0, 2, 1)
    target_heats  = targets.cuda(non_blocking=True)
    target_points = normpoints[:, :, :2].contiguous().cuda(non_blocking=True)
    target_scores = normpoints[:, :, 2:].contiguous().cuda(non_blocking=True)
    det_masks     = (1-nopoints).view(batch_size, 1, 1, 1) * masks
    have_det_loss = det_masks.sum().item() > 0
    det_masks     = det_masks.cuda(non_blocking=True)
    nopoints      = nopoints.squeeze(1).tolist()

    # measure data loading time
    data_time.update(time.time() - end)

    # batch_heatmaps is a list for stage-predictions, each element should be [Batch, Sequence, PTS, H/Down, W/Down]
    batch_heatmaps, batch_locs, batch_scos, batch_past2now, batch_future2now, batch_FBcheck, multiview_heatmaps, multiview_locs = net(frames, Fflows, Bflows, MV_Tensors, torch_is_images)
    annot_heatmaps = [x[:, annotate_index] for x in batch_heatmaps]
    forward_time.update(time.time() - end)
  
    # detection loss
    if have_det_loss:
      det_loss, each_stage_loss_value = compute_stage_loss(criterion, target_heats, annot_heatmaps, det_masks)
      DetLosses.update(det_loss.item(), batch_size)
      each_stage_loss_value = show_stage_loss(each_stage_loss_value)
    else:
      det_loss, each_stage_loss_value = 0, 'no-det-loss'

    # temporal loss
    if use_stm[0]:
      video_batch_locs = batch_locs[i_batch_size:, :, :num_pts]
      video_past2now, video_future2now = batch_past2now[i_batch_size:, :, :num_pts], batch_future2now[i_batch_size:, :, :num_pts]
      video_FBcheck    = batch_FBcheck[i_batch_size:, :num_pts]
      video_mask       = masks[i_batch_size:, :num_pts].contiguous().cuda(non_blocking=True)
      video_heatmaps   = [ x[i_batch_size:, :, :num_pts] for x in batch_heatmaps ]
      sbr_loss, available_nums, loss_string  = calculate_temporal_loss(criterion, video_heatmaps, video_batch_locs, video_past2now, video_future2now, video_FBcheck, video_mask, stm_config)
      alk_points.update(float(available_nums)/v_batch_size, v_batch_size)
      if available_nums > stm_config.available_sbr_thresh:
        TemporalLosses.update(sbr_loss.item(), v_batch_size)
      else:
        sbr_loss, sbr_loss_string = 0, 'non-sbr-loss'
    else:
      sbr_loss, sbr_loss_string = 0, 'non-sbr-loss'

    # multiview loss
    if use_stm[1]:
      MV_Mask_G         = MV_Mask[:,:-1].view(m_batch_size, 1, -1, 1).contiguous().cuda(non_blocking=True)
      MV_Thetas_G       = MV_Thetas.to(multiview_locs.device)
      MV_Shapes_G       = MV_Shapes.to(multiview_locs.device).view(m_batch_size, num_views, 1, 2)
      MV_KRT_G          = MV_KRT.to(multiview_locs.device)
      mv_norm_locs_trs  = torch.cat((multiview_locs[:,:,:num_pts].permute(0,1,3,2), torch.ones(m_batch_size, num_views, 1, num_pts, device=multiview_locs.device)), dim=2)
      mv_norm_locs_ori  = torch.matmul(MV_Thetas_G[:,:,:2], mv_norm_locs_trs)
      mv_norm_locs_ori  = mv_norm_locs_ori.permute(0,1,3,2)
      mv_real_locs_ori  = denormalize_L(mv_norm_locs_ori, MV_Shapes_G)
      mv_3D_locs_ori    = TriangulateDLT_BatchCam(MV_KRT_G, mv_real_locs_ori)
      mv_proj_locs_ori  = ProjectKRT_Batch(MV_KRT_G, mv_3D_locs_ori.view(m_batch_size, 1, num_pts, 3))
      mv_pnorm_locs_ori = normalize_L(mv_proj_locs_ori, MV_Shapes_G)
      mv_pnorm_locs_trs = convert_theta(mv_pnorm_locs_ori, MV_Thetas_G)
      MV_locs           = multiview_locs[:,:,:num_pts].contiguous()
      MV_heatmaps       = [ x[:,:,:num_pts] for x in multiview_heatmaps ]
  
      if args.debug:
        with torch.no_grad():
          for ims in range(m_batch_size):
            x_index = image_index[iv_size+ims]
            x_paths = [xlist[iv_size+ims] for xlist in MultiViewPaths]
            x_mv_locs, p_mv_locs = mv_real_locs_ori[ims], mv_proj_locs_ori[ims]
            multiview_debug_save(save_dir, '{:}'.format(x_index), x_paths, x_mv_locs.cpu().numpy(), p_mv_locs.cpu().numpy())
            y_mv_locs = denormalize_points_batch((H,W), MV_locs[ims])
            q_mv_locs = denormalize_points_batch((H,W), mv_pnorm_locs_trs[ims])
            temp_tensors = MV_Tensors[ims]
            temp_images = [args.tensor2imageF(x) for x in temp_tensors]
            temp_names  = [Path(x).name for x in x_paths]
            multiview_debug_save_v2(save_dir, '{:}'.format(x_index), temp_names, temp_images, y_mv_locs.cpu().numpy(), q_mv_locs.cpu().numpy())

      stm_loss, available_nums = calculate_multiview_loss(criterion, MV_heatmaps, MV_locs, mv_pnorm_locs_trs, MV_Mask_G, stm_config)
      a3d_points.update(float(available_nums)/m_batch_size, m_batch_size)
      if available_nums > stm_config.available_stm_thresh:
        MultiviewLosses.update(stm_loss.item(), m_batch_size)
      else:
        stm_loss = 0
    else:
      stm_loss = 0

    # measure accuracy and record loss
    if use_stm[0]: total_loss = det_loss + sbr_loss * stm_config.sbr_weights + stm_loss * stm_config.stm_weights
    else         : total_loss = det_loss + stm_loss * stm_config.stm_weights
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
      for ibatch in range(iv_size):
        imgidx, nopoint = image_index[ibatch], nopoints[ibatch]
        if nopoint == 1: continue
        norm_locs  = torch.cat((batch_locs[ibatch].permute(1,0), torch.ones(1, num_pts)), dim=0)
        transtheta = transthetas[ibatch][:2,:]
        norm_locs  = torch.mm(transtheta, norm_locs)
        real_locs  = denormalize_points(shapes[ibatch].tolist(), norm_locs)
        real_locs  = torch.cat((real_locs, batch_scos[ibatch].view(1, num_pts)), dim=0)
  
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
                'SBR {sloss.val:7.6f} ({sloss.avg:7.6f}) '
                'STM {mloss.val:7.6f} ({mloss.avg:7.6f}) '
                'Loss {loss.val:7.4f} ({loss.avg:7.4f})  '.format(
                    mode, epoch_str, i, len(loader), batch_time=batch_time,
                    data_time=data_time, forward_time=forward_time, \
                    dloss=DetLosses, sloss=TemporalLosses, mloss=MultiviewLosses, loss=TotalLosses)
                  + last_time + each_stage_loss_value \
                  + ' I={:}'.format(list(frames.size())) \
                  + ' Vis-PTS : {:2d} ({:.1f})'.format(int(visible_points.val), visible_points.avg) \
                  + ' Ava-PTS : {:.1f} ({:.1f})'.format(alk_points.val, alk_points.avg) \
                  + ' A3D-PTS : {:.1f} ({:.1f})'.format(a3d_points.val, a3d_points.avg) )
      if args.debug:
        logger.log('  -->>Indexes : {:}'.format(image_index))
  nme, _, _ = eval_meta.compute_mse(loader.dataset.dataset_name, logger)
  return TotalLosses.avg, nme
