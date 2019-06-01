# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch, copy
import torch.nn as nn
import lk

class LK(nn.Module):
  def __init__(self, model, lkconfig, points):
    super(LK, self).__init__()
    self.detector = model
    self.downsample = self.detector.downsample
    self.config = copy.deepcopy(lkconfig)
    self.points = points

  def forward(self, inputs):
    assert inputs.dim() == 5, 'This model accepts 5 dimension input tensor: {}'.format(inputs.size())
    batch_size, sequence, C, H, W = list( inputs.size() )
    gathered_inputs = inputs.view(batch_size * sequence, C, H, W)
    heatmaps, batch_locs, batch_scos = self.detector(gathered_inputs)
    heatmaps = [x.view(batch_size, sequence, self.points, H//self.downsample, W//self.downsample) for x in heatmaps]
    batch_locs, batch_scos = batch_locs.view(batch_size, sequence, self.points, 2), batch_scos.view(batch_size, sequence, self.points)
    batch_next, batch_fback, batch_back = [], [], []

    for ibatch in range(batch_size):
      feature_old = inputs[ibatch]
      nextPts, fbackPts, backPts = lk.lk_forward_backward_batch(inputs[ibatch], batch_locs[ibatch], self.config.window, self.config.steps)

      batch_next.append(nextPts)
      batch_fback.append(fbackPts)
      batch_back.append(backPts)
    batch_next, batch_fback, batch_back = torch.stack(batch_next), torch.stack(batch_fback), torch.stack(batch_back)
    return heatmaps, batch_locs, batch_scos, batch_next, batch_fback, batch_back
