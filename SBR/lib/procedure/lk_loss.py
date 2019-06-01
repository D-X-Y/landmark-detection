# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import models
import torch
import numpy as np
import pdb, math, numbers

def lk_input_check(batch_locs, batch_scos, batch_next, batch_fback, batch_back):
  batch, sequence, num_pts, _ = list(batch_locs.size())
  assert batch_locs.size() == batch_next.size() == batch_fback.size() == batch_back.size(), '{:} vs {:} vs {:} vs {:}'.format(batch_locs.size(), batch_next.size(), batch_fback.size(), batch_back.size())
  assert _ == 2, '{:}'.format(batch_locs.size())
  assert batch_scos.size(0) == batch and batch_scos.size(1) == sequence and batch_scos.size(2) == num_pts, '{:} vs {:}'.format(batch_locs.size(), batch_scos.size())
  return batch, sequence, num_pts

def p2string(point):
  if isinstance(point, numbers.Number):
    return '{:.1f}'.format(point*1.0)
  elif point.size == 2:
    return '{:.1f},{:.1f}'.format(point[0], point[1])
  else:
    return '{}'.format(point)

def lk_target_loss(batch_locs, batch_scos, batch_next, batch_fbak, batch_back, lk_config, video_or_not, mask, nopoints):
  # return the calculate target from the first frame to the whole sequence.
  batch, sequence, num_pts = lk_input_check(batch_locs, batch_scos, batch_next, batch_fbak, batch_back)

  # remove the background
  num_pts = num_pts - 1
  sequence_checks = np.ones((batch, num_pts), dtype='bool')

  # Check the confidence score for each point
  for ibatch in range(batch):
    if video_or_not[ibatch] == False:
      sequence_checks[ibatch, :] = False
    else:
      for iseq in range(sequence):
        for ipts in range(num_pts):
          score = batch_scos[ibatch, iseq, ipts]
          if mask[ibatch, ipts] == False and nopoints[ibatch] == 0:
            sequence_checks[ibatch, ipts] = False
          if score.item() < lk_config.conf_thresh:
            sequence_checks[ibatch, ipts] = False

  losses = []
  for ibatch in range(batch):
    for ipts in range(num_pts):
      if not sequence_checks[ibatch, ipts]: continue
      loss = 0
      for iseq in range(sequence):
      
        targets = batch_locs[ibatch, iseq, ipts]
        nextPts = batch_next[ibatch, iseq, ipts]
        fbakPts = batch_fbak[ibatch, iseq, ipts]
        backPts = batch_back[ibatch, iseq, ipts]

        with torch.no_grad():
          fbak_distance = torch.dist(nextPts, fbakPts)
          back_distance = torch.dist(targets, backPts)
          forw_distance = torch.dist(targets, nextPts)

        #print ('[{:02d},{:02d},{:02d}] : {:.2f}, {:.2f}, {:.2f}'.format(ibatch, ipts, iseq, fbak_distance.item(), back_distance.item(), forw_distance.item()))
        #loss += back_distance + forw_distance

        if fbak_distance.item() > lk_config.fb_thresh or fbak_distance.item() < lk_config.eps: # forward-backward-check
          if iseq+1 < sequence: sequence_checks[ibatch, ipts] = False
        if forw_distance.item() > lk_config.forward_max or forw_distance.item() < lk_config.eps: # to avoid the tracker point is too far
          if iseq   > 0       : sequence_checks[ibatch, ipts] = False
        if back_distance.item() > lk_config.forward_max or back_distance.item() < lk_config.eps: # to avoid the tracker point is too far
          if iseq+1 < sequence: sequence_checks[ibatch, ipts] = False

        if iseq > 0:
          if lk_config.stable: loss += torch.dist(targets, backPts.detach())
          else               : loss += torch.dist(targets, backPts)
        if iseq + 1 < sequence:
          if lk_config.stable: loss += torch.dist(targets, nextPts.detach())
          else               : loss += torch.dist(targets, nextPts)

      if sequence_checks[ibatch, ipts]:
        losses.append(loss)

  avaliable = int(np.sum(sequence_checks))
  if avaliable == 0: return None, avaliable
  else             : return torch.mean(torch.stack(losses)), avaliable
