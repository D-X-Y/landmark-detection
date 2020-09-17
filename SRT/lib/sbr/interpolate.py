# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
import torch


# tensor has the shape of [C, H, W]
# x and y are the batch of locations, [N]
def bilinear_interpolate(tensor, x, y):
  assert tensor.dim() == 3 and x.dim() == 1 and y.dim() == 1, 'invalid size : {:}'.format(tensor.size())
  x0, y0 = torch.floor(x).long(), torch.floor(y).long()
  x1, y1 = x0 + 1, y0 + 1
  x0 = torch.clamp(x0, 0, tensor.shape[2]-1)
  x1 = torch.clamp(x1, 0, tensor.shape[2]-1)
  y0 = torch.clamp(y0, 0, tensor.shape[1]-1)
  y1 = torch.clamp(y1, 0, tensor.shape[1]-1)
  
  Ia, Ib = tensor[:,y0,x0], tensor[:,y1,x0]
  Ic, Id = tensor[:,y0,x1], tensor[:,y1,x1]

  wa = (x1.type(x.dtype) - x) * (y1.type(y.dtype) - y)
  wb = (x1.type(x.dtype) - x) * (y - y0.type(y.dtype))
  wc = (x - x0.type(x.dtype)) * (y1.type(y.dtype) - y)
  wd = (x - x0.type(x.dtype)) * (y - y0.type(y.dtype))
  outs = torch.t(Ia) * wa.unsqueeze(1) + torch.t(Ib) * wb.unsqueeze(1) + torch.t(Ic) * wc.unsqueeze(1) + torch.t(Id) * wd.unsqueeze(1)
  return outs


# flows has the shape of [Batch, H, W, 2]
# locs  has the shape of [Batch, Points, 2]
# outputs : [Batch, Points, 2]
def batch_interpolate_flow(flows, locs, return_absolute):
  assert flows.dim() == 4 and flows.size(-1) == 2, 'invalid size of flows : {:}'.format(flows.size())
  assert locs.dim() == 3 and locs.size(-1) == 2, 'invalid size of locs : {:}'.format(locs.size())
  batch, H, W, _ = flows.size()
  x0, y0 = torch.floor(locs[:,:,0]).long(), torch.floor(locs[:,:,1]).long()
  x1, y1 = x0 + 1, y0 + 1
  x0 = torch.clamp(x0, 0, W-1)
  x1 = torch.clamp(x1, 0, W-1)
  y0 = torch.clamp(y0, 0, H-1)
  y1 = torch.clamp(y1, 0, H-1)

  """
  def get_matrix_by_index_trivial(tensor, _x, _y):
    assert tensor.dim() == 4 and tensor.size(0) == _x.size(0) == _y.size(0)
    (_B, _H, _W, _), _P = tensor.size(), _x.size(1)
    # get the index matrix with shape [batch, points, 2]
    indexes = []
    for i in range(_B):
      Bidx = []
      for j in range(_P):
        a = i*_H*_W*2 + _y[i,j].item()*_W*2 + _x[i,j].item()*2
        Bidx.append([a, a+1])
      indexes.append( Bidx )
    indexes = torch.tensor(indexes, device=tensor.device)
    offsets = torch.take(tensor, indexes)
    return offsets
  """

  def get_matrix_by_index(tensor, _x, _y):
    assert tensor.dim() == 4 and tensor.size(0) == _x.size(0) == _y.size(0)
    (_B, _H, _W, _), _P = tensor.size(), _x.size(1)
    indexes = torch.arange(0, _B, device=_x.device).unsqueeze(-1) * _H * _W * 2 + _y * _W * 2 + _x * 2
    indexes = torch.stack([indexes, indexes+1], dim=-1)
    return torch.take(tensor, indexes)

  Ia = get_matrix_by_index(flows, x0, y0)
  Ib = get_matrix_by_index(flows, x0, y1)
  Ic = get_matrix_by_index(flows, x1, y0)
  Id = get_matrix_by_index(flows, x1, y1)

  x0, x1, y0, y1 = x0.type(locs.dtype), x1.type(locs.dtype), y0.type(locs.dtype), y1.type(locs.dtype)
  wa = (x1 - locs[:,:,0]) * (y1 - locs[:,:,1])
  wb = (x1 - locs[:,:,0]) * (locs[:,:,1] - y0)
  wc = (locs[:,:,0] - x0) * (y1 - locs[:,:,1])
  wd = (locs[:,:,0] - x0) * (locs[:,:,1] - y0)

  outs = Ia * wa.unsqueeze(-1) + Ib * wb.unsqueeze(-1) + Ic * wc.unsqueeze(-1) + Id * wd.unsqueeze(-1)
  if return_absolute: return outs, outs + locs
  else              : return outs
