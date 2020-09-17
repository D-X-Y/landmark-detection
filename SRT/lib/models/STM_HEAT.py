# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch, copy
import torch.nn as nn
from sbr import batch_interpolate_flow
from xvision import normalize_points_batch, denormalize_points_batch


class SpatialTemporalMultiviewHEAT(nn.Module):
  def __init__(self, model, config, points):
    super(SpatialTemporalMultiviewHEAT, self).__init__()
    self.detector   = model
    self.downsample = self.detector.downsample
    self.config     = copy.deepcopy(config)
    self.points     = points
    self.center_idx = self.config.video_L


  def forward(self, inputs, Fflows, Bflows, stm_inputs, is_images):
    assert inputs.dim()     == 5, 'This model accepts 5 dimension input tensor: {:}'.format(inputs.size())
    assert stm_inputs.dim() == 5, 'This model accepts 5 dimension stm-input tensor: {:}'.format(stm_inputs.size())
    vbatch_size, sequence, C, H, W = inputs.size()
    mbatch_size, views   , C, H, W = stm_inputs.size()
    gathered_N_inputs = inputs.view(vbatch_size * sequence, C, H, W)
    gathered_M_inputs = stm_inputs.view(mbatch_size * views, C, H, W)
    gathered_inputs   = torch.cat((gathered_N_inputs, gathered_M_inputs))
    #gathered_batch_locs = self.detector(gathered_inputs)
    gathered_F, gathered_heatmaps, gathered_batch_locs, gathered_batch_scos = self.detector(gathered_inputs)
    batch_heatmaps = [x[:vbatch_size*sequence].view(vbatch_size, sequence, self.points, H//self.downsample, W//self.downsample) for x in gathered_heatmaps]
    multiview_heatmaps = \
                     [x[vbatch_size*sequence:].view(mbatch_size,   views , self.points, H//self.downsample, W//self.downsample) for x in gathered_heatmaps]
    
    batch_locs     = gathered_batch_locs[:vbatch_size*sequence].view(vbatch_size, sequence, self.points, 2)
    batch_scos     = gathered_batch_scos[:vbatch_size*sequence].view(vbatch_size, sequence, self.points)
    multiview_locs = gathered_batch_locs[vbatch_size*sequence:].view(mbatch_size, views   , self.points, 2)
    multiview_scos = gathered_batch_scos[vbatch_size*sequence:].view(mbatch_size, views   , self.points)

    #real_locs      = denormalize_points_batch((H,W), batch_locs)
    #real_locs      = real_locs.view(vbatch_size, sequence, self.points, 2)
    batch_locs     = batch_locs.view(vbatch_size, sequence, self.points, 2)

    batch_next, batch_fback, batch_back = [], [], []

    past2now, future2now, fb_check = self.SBR_forward_backward(batch_locs, Fflows, Bflows)
    batch_locs           = normalize_points_batch((H,W), batch_locs)
    multiview_locs       = normalize_points_batch((H,W), multiview_locs)
    past2now, future2now = normalize_points_batch((H,W), past2now), normalize_points_batch((H,W), future2now)
    return batch_heatmaps, batch_locs, batch_scos, past2now, future2now, fb_check, multiview_heatmaps, multiview_locs

  
  def SBR_forward_backward(self, landmarks, Fflows, Bflows):
    # calculate the landmarks from the past frame to the current frame
    batch, frames, points, _ = landmarks.size()
    batch, _, H, W, _ = Fflows.size()
    F_gather_flows = Fflows.view(batch*(frames-1), H, W, 2)
    F_landmarks    = landmarks[:,:-1].contiguous().view(batch*(frames-1), points, 2)
    _, past2now_locs   = batch_interpolate_flow(F_gather_flows, F_landmarks, True)
    past2now_locs = past2now_locs.view(batch, frames-1, points, 2)

    B_gather_flows = Bflows.view(batch*(frames-1), H, W, 2)
    B_landmarks    = landmarks[:,1:].contiguous().view(batch*(frames-1), points, 2)
    _, future2now_locs = batch_interpolate_flow(B_gather_flows, B_landmarks, True)
    future2now_locs = future2now_locs.view(batch, frames-1, points, 2)
    # forward-backward check
    current_locs = start_locs = landmarks[:, 0]
    for i in range(1, frames):
      _, current_locs = batch_interpolate_flow(Fflows[:, i-1], current_locs, True)
    finish_locs = current_locs
    for i in range(frames-1, 0, -1):
      _, finish_locs  = batch_interpolate_flow(Bflows[:, i-1], finish_locs, True)
    return past2now_locs, future2now_locs, start_locs - finish_locs
