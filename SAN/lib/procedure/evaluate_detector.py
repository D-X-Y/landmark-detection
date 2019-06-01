##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import os, time, numpy as np
import numbers
import os.path as osp
import torch
from utils import AverageMeter, print_log, convert_size2str, convert_secs2time, time_string, time_for_file
from san_vision import Eval_Meta
from models import variable2np
from debug import main_debug_save

def evaluation(eval_loaders, net, log, save_path, opt):
  print_log('Evaluation => {} datasets, save into {}'.format(len(eval_loaders), save_path), log)
  if not osp.isdir(save_path): os.makedirs( save_path )
  assert osp.isdir(save_path), 'The save path {} is not a dir'.format(save_path)
  for i, eval_loader in enumerate(eval_loaders):
    print_log('  Evaluate => [{:2d}/{:2d}]-th image dataset : {:}'.format(i, len(eval_loaders), opt.eval_lists[i]), log)
    isave_path = osp.join(save_path, 'eval-set-{:02d}'.format(i))
    with torch.no_grad():
      meta = evaluation_image(eval_loader, net, log, isave_path, opt)
    meta.compute_mse(log)
    meta_path = osp.join(isave_path, 'evaluation.pth.tar')
    meta.save(meta_path)
  if len(eval_loaders) > 0: print_log('====>> Evaluate all image datasets done', log)

def evaluation_image(eval_loader, net, log, save_path, opt):
  if not osp.isdir(save_path): os.makedirs(save_path)
  if opt.debug_save:
    debug_save_dir = osp.join(save_path, 'debug-eval')
    if not os.path.isdir(debug_save_dir): os.makedirs(debug_save_dir)
  else: debug_save_dir = None
  print_log(' ==>start evaluation image dataset, save into {} with the error bar : {}, using the last stage, DEBUG SAVE DIR : {}'.format(save_path, opt.error_bar, debug_save_dir), log)

  # switch to eval mode
  net.eval()
  # calculate the time
  batch_time, end = AverageMeter(), time.time()
  # save the evaluation results
  eval_meta = Eval_Meta()
  # save the number of points
  num_pts, scale_eval = opt.num_pts, opt.scale_eval

  for i, (inputs, target, mask, points, image_index, label_sign, ori_size) in enumerate(eval_loader):

    # inputs : Batch, Squence, Channel, Height, Width
    target = target.cuda(async=True)
    mask = mask.cuda(async=True)

    # forward, batch_locs [1 x points] [batch, 2]
    batch_cpms, batch_locs, batch_scos, generated = net(inputs)
    assert batch_locs.size(0) == batch_scos.size(0) and batch_locs.size(0) == inputs.size(0)
    assert batch_locs.size(1) == num_pts + 1 and batch_scos.size(1) == num_pts + 1
    assert batch_locs.size(2) == 2 and len(batch_scos.size()) == 2
    np_batch_locs, np_batch_scos = variable2np(batch_locs), variable2np(batch_scos)
    image_index = np.squeeze(variable2np( image_index )).tolist()
    if isinstance(image_index, numbers.Number): image_index = [image_index]
    sign_list = variable2np(label_sign).astype('bool').squeeze(1).tolist()
    # recover the ground-truth label
    real_sizess = variable2np( ori_size )

    for ibatch, imgidx in enumerate(image_index):
      locations, scores = np_batch_locs[ibatch,:-1,:], np_batch_scos[ibatch,:-1]
      if len(scores.shape) == 1: scores = np.expand_dims(scores, axis=1)
      xpoints = eval_loader.dataset.labels[imgidx].get_points()
      assert real_sizess[ibatch,0] > 0 and real_sizess[ibatch,1] > 0, 'The ibatch={}, imgidx={} is not right.'.format(ibatch, imgidx)
      scale_h, scale_w = real_sizess[ibatch,0] * 1. / inputs.size(-2) , real_sizess[ibatch,1] * 1. / inputs.size(-1)
      locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + real_sizess[ibatch,2], locations[:, 1] * scale_h + real_sizess[ibatch,3]
      assert xpoints.shape[1] == num_pts and locations.shape[0] == num_pts and scores.shape[0] == num_pts, 'The number of points is {} vs {} vs {} vs {}'.format(num_pts, xpoints.shape, locations.shape, scores.shape)
      # recover the original resolution
      locations = np.concatenate((locations, scores), axis=1)
      image_path = eval_loader.dataset.datas[imgidx]
      face_size  = eval_loader.dataset.face_sizes[imgidx]
      image_name = osp.basename( image_path )
      locations = locations.transpose(1,0)
      eval_meta.append(locations, xpoints, image_path, face_size)

      if opt.error_bar is not None:
        errorpath = osp.join(save_path, image_name)
        save_error_image(image_path, xpoints, locations, opt.error_bar, errorpath, radius=5, color=(30,255,30), rev_color=(255,30,30), fontScale=10, text_color=(255,255,255))
    if opt.debug_save:
      print_log('DEBUG --- > [{:03d}/{:03d}] '.format(i, len(eval_loader)), log)
      main_debug_save(debug_save_dir, eval_loader, image_index, inputs, batch_locs, target, points, sign_list, batch_cpms, generated, log)

    # measure elapsed time
    batch_time.update(time.time() - end)
    need_hour, need_mins, need_secs = convert_secs2time(batch_time.avg * (len(eval_loader)-i-1))
    end = time.time()

    if i % opt.print_freq_eval == 0 or i+1 == len(eval_loader):
      print_log('  Evaluation: [{:03d}/{:03d}] Time {:6.3f} ({:6.3f}) Need [{:02d}:{:02d}:{:02d}]'.format(i, len(eval_loader), batch_time.val, batch_time.avg, need_hour, need_mins, need_secs) + ' locations.shape={}'.format(np_batch_locs.shape), log)
      
  return eval_meta
