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


def calculate_robust(predictions, labels, normalizations, num_pts):
  assert isinstance(predictions, tuple) or isinstance(predictions, list), 'invalid type : {:}'.format( type(predictions) )
  assert isinstance(     labels, tuple) or isinstance(     labels, list), 'invalid type : {:}'.format( type(     labels) )
  assert isinstance(normalizations, tuple) or isinstance(normalizations, list), 'invalid type : {:}'.format( type(normalizations) )

  num = len(predictions)
  assert len(labels)         == num, 'The number of labels : {:} vs {:}'.format( len(labels), num )
  assert len(normalizations) == num, 'The number of normalizations : {:} vs {:}'.format( len(normalizations), num )
  
  errors, valids, invalid_num = [], [], 0
  for i in range( num ):
    pred, label, norm = predictions[i], labels[i], normalizations[i]
    masks = [x[:,2] for x in pred] + [label[:,2]]
    masks = torch.stack( masks ).sum(0) == len(masks)
    valids.append( masks.float().mean().item() )
    if sum(masks).item() == 0:
      invalid_num += 1
      continue
    assert sum(masks) > 0, 'The [{:03d}]-th data has no avaliable landmark predictions'.format(i)
    assert len(pred ) > 1, 'The [{:03d}]-th data only has {:} groups of landmark prediction'.format(i, len(pred))
    discrepancies = []
    for j, transA in enumerate(pred):
      for k in range(j):
        transB = pred[k]
        discrepancy = torch.norm((transA-transB)[masks, :2], dim=1) / norm
        discrepancies.append( discrepancy.mean().item() * 1000 )
    errors.append( float( np.mean(discrepancies) ) )
  print ('There are {:} / {:} invalid samples'.format(invalid_num, num))
  return errors, valids


def eval_robust_heatmap(detector, xloader, print_freq, logger):
  batch_time, NUM_PTS = AverageMeter(), xloader.dataset.NUM_PTS
  Preds, GT_locs, Distances = [], [], []
  eval_meta, end = Eval_Meta(), time.time()

  with torch.no_grad():
    detector.eval()
    for i, (inputs, heatmaps, masks, norm_points, thetas, data_index, nopoints, xshapes) in enumerate(xloader):
      data_index = data_index.squeeze(1).tolist()
      batch_size, iters, C, H, W = inputs.size()
      for ibatch in range(batch_size):
        xinputs, xpoints, xthetas = inputs[ibatch], norm_points[ibatch].permute(0, 2, 1).contiguous(), thetas[ibatch]
        batch_features, batch_heatmaps, batch_locs, batch_scos = detector( xinputs.cuda(non_blocking=True) )
        batch_locs = batch_locs.cpu()[:,:-1]
        all_locs   = []
        for _iter in range(iters):
          _locs = normalize_points((H,W), batch_locs[_iter].permute(1,0))
          xlocs = torch.cat((_locs, torch.ones(1, NUM_PTS)), dim=0)
          nlocs = torch.mm(xthetas[_iter,:2], xlocs)
          rlocs = denormalize_points(xshapes[ibatch].tolist(), nlocs)
          rlocs = torch.cat((rlocs.permute(1,0), xpoints[_iter,:,2:]), dim=1)
          all_locs.append( rlocs.clone() )
        GT_loc = xloader.dataset.labels[ data_index[ibatch] ].get_points()
        norm_distance = xloader.dataset.get_normalization_distance( data_index[ibatch] )
        # save the results
        eval_meta.append((sum(all_locs)/len(all_locs)).numpy().T, GT_loc.numpy(), xloader.dataset.datas[ data_index[ibatch] ], norm_distance)
        Distances.append( norm_distance )
        Preds.append( all_locs )
        GT_locs.append( GT_loc.permute(1,0) )
      # compute time
      batch_time.update(time.time() - end)
      end = time.time()
      if i % print_freq == 0 or i+1 == len(xloader):
        last_time = convert_secs2time(batch_time.avg * (len(xloader)-i-1), True)
        logger.log(' -->>[Robust HEATMAP-based Evaluation] [{:03d}/{:03d}] Time : {:}'.format(i, len(xloader), last_time))
  # evaluate the results  
  errors, valids = calculate_robust(Preds, GT_locs, Distances, NUM_PTS)
  return errors, valids, eval_meta


def eval_robust_regression(detector, xloader, print_freq, logger):
  batch_time, NUM_PTS = AverageMeter(), xloader.dataset.NUM_PTS
  Preds, GT_locs, Distances = [], [], []
  eval_meta, end = Eval_Meta(), time.time()
  #xloader.dataset.get_normalization_distance(None, True)
  with torch.no_grad():
    detector.eval()
    for i, (inputs, heatmaps, masks, norm_points, thetas, data_index, nopoints, xshapes) in enumerate(xloader):
      data_index = data_index.squeeze(1).tolist()
      batch_size, iters, C, H, W = inputs.size()
      for ibatch in range(batch_size):
        xinputs, xpoints, xthetas = inputs[ibatch], norm_points[ibatch].permute(0, 2, 1).contiguous(), thetas[ibatch]
        batch_locs = detector( xinputs.cuda(non_blocking=True) ).cpu()
        all_locs   = []
        for _iter in range(iters):
          xlocs = torch.cat((batch_locs[_iter].permute(1,0), torch.ones(1, NUM_PTS)), dim=0)
          nlocs = torch.mm(xthetas[_iter,:2], xlocs)
          rlocs = denormalize_points(xshapes[ibatch].tolist(), nlocs)
          rlocs = torch.cat((rlocs.permute(1,0), xpoints[_iter,:,2:]), dim=1)
          all_locs.append( rlocs.clone() )
        GT_loc = xloader.dataset.labels[ data_index[ibatch] ].get_points()
        norm_distance = xloader.dataset.get_normalization_distance( data_index[ibatch] )
        # save the results
        eval_meta.append((sum(all_locs)/len(all_locs)).numpy().T, GT_loc.numpy(), xloader.dataset.datas[ data_index[ibatch] ], norm_distance)
        Distances.append( norm_distance )
        Preds.append( all_locs )
        GT_locs.append( GT_loc.permute(1,0) )
      # compute time
      batch_time.update(time.time() - end)
      end = time.time()
      if i % print_freq == 0 or i+1 == len(xloader):
        last_time = convert_secs2time(batch_time.avg * (len(xloader)-i-1), True)
        logger.log(' -->>[Robust LINEAR-based Evaluation] [{:03d}/{:03d}] Time : {:}'.format(i, len(xloader), last_time))
  # evaluate the results  
  errors, valids = calculate_robust(Preds, GT_locs, Distances, NUM_PTS)
  return errors, valids, eval_meta
