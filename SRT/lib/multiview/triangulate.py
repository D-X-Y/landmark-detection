# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
import math, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def GetKRT(K, R, t, dtype=torch.FloatTensor):
  if isinstance(K, np.ndarray): K = torch.from_numpy(K)
  if isinstance(R, np.ndarray): R = torch.from_numpy(R)
  if isinstance(t, np.ndarray): t = torch.from_numpy(t)
  Rt = torch.cat((R, t), dim=1)
  KRT = torch.mm(K, Rt).type( dtype )
  return KRT


# a list of KRTs and a list of points for different cameras
# KRTs   : N * 3 * 4
# points : N * 2 
def TriangulateDLT(KRTs, points):
  assert KRTs.dim() == 3 and points.dim() == 2 and KRTs.size(0) == points.size(0), 'KRTs : {:}, points : {:}'.format(KRTs.shape, points.shape)
  assert KRTs.size(1) == 3 and KRTs.size(2) == 4 and points.size(-1) == 2, 'KRTs : {:}, points : {:}'.format(KRTs.shape, points.shape)
  U = KRTs[:,0,:] - KRTs[:,2,:] * points[:,0].view(-1, 1)
  V = KRTs[:,1,:] - KRTs[:,2,:] * points[:,1].view(-1, 1)
  Dmatrix = torch.cat((U,V))
  A   = Dmatrix[:,:3]
  At  = torch.transpose(A, 0, 1)
  AtA = torch.mm(At, A)
  invAtA = torch.inverse( AtA )
  P3D = torch.mm(invAtA, torch.mm(At, -Dmatrix[:,3:]))
  return P3D.view(-1, 3)


# one KRT and 3D-points
# KRT : 3x4 ; PTS3D : Nx3
def ProjectKRT(KRT, PTS3D):
  assert KRT.dim() == 2 and KRT.size(0) == 3 and KRT.size(1) == 4, 'KRT : {:}'.format(KRT.shape)
  assert PTS3D.dim() == 2 and PTS3D.size(-1) == 3, 'PTS3D : {:}'.format(PTS3D.shape)
  MPs = torch.matmul(KRT[:,:3], PTS3D.transpose(1,0)) + KRT[:, 3:]
  X = MPs[0] / MPs[2]
  Y = MPs[1] / MPs[2]
  return torch.stack((X,Y), dim=1)


# Batch KRT and Batch PTS3D
# KRT     : .... x 3 x 4
# PTS3D   : .... x N x 3
# projPTS : .... x N x 2
def ProjectKRT_Batch(KRT, PTS3D):
  assert KRT.dim() == PTS3D.dim() and PTS3D.size(-1) == 3 and KRT.size(-2) == 3 and KRT.size(-1) == 4, 'KRT : {:} | PTS3D : {:}'.format(KRT.shape, PTS3D.shape)
  MPs  = torch.matmul(KRT[...,:3], PTS3D.transpose(-1,-2)) + KRT[...,3:]
  NMPs = MPs.transpose(-1,-2)
  projPTS = NMPs[..., :2] / NMPs[..., 2:]
  return projPTS


# http://cmp.felk.cvut.cz/cmp/courses/TDV/2012W/lectures/tdv-2012-07-anot.pdf
# a multiview system has N cameras with N KRTs, [N * P * 2] points
# KRTs   : N * 3 * 4
# points : N * P * 2
def TriangulateDLT_BatchPoints(KRTs, points):
  assert KRTs.dim() == 3 and points.dim() == 3 and KRTs.size(0) == points.size(0), 'KRTs : {:}, points : {:}'.format(KRTs.shape, points.shape)
  assert KRTs.size(1) == 3 and KRTs.size(2) == 4 and points.size(-1) == 2, 'KRTs : {:}, points : {:}'.format(KRTs.shape, points.shape)
  assert points.size(1) >= 3, 'There should be at least 3 points'.format(points.shape)
  KRTs = KRTs.view(KRTs.size(0), 1, 3, 4)             # size = N * 1 * 3 * 4
  U = KRTs[:,:,0,:] - KRTs[:,:,2,:] * points[...,0:1] # size = N * P * 4
  V = KRTs[:,:,1,:] - KRTs[:,:,2,:] * points[...,1:2] 
  Dmatrix = torch.cat((U,V), dim=0).transpose(1,0)    # size = P * 2N * 4
  A      = Dmatrix[:,:,:3]
  At     = torch.transpose(A, 2, 1)
  AtA    = torch.matmul(At, A)
  invAtA = torch.inverse( AtA )
  P3D    = torch.matmul(invAtA, torch.matmul(At, -Dmatrix[:,:,3:]))
  return P3D.view(-1, 3)


# many multiview systems, each of them has N cameras with N KRTs and [N * P * 2] points
# KRTs   : M * N * 3 * 4
# points : M * N * P * 2
def TriangulateDLT_BatchCam(KRTs, points):
  assert KRTs.dim() == 4 and points.dim() == 4 and KRTs.size(0) == points.size(0) and KRTs.size(1) == points.size(1), 'KRTs : {:}, points : {:}'.format(KRTs.shape, points.shape)
  assert KRTs.size(2) == 3 and KRTs.size(3) == 4 and points.size(-1) == 2, 'KRTs : {:}, points : {:}'.format(KRTs.shape, points.shape)
  assert points.size(-2) >= 3, 'There should be at least 3 points'.format(points.shape)
  batch_mv, batch_cam = KRTs.size(0), KRTs.size(1)
  KRTs = KRTs.view(batch_mv, batch_cam, 1, 3, 4)      # size = M * N * 1 * 3 * 4
  U = KRTs[...,0,:] - KRTs[...,2,:] * points[...,0:1] # size = M * N * P * 4
  V = KRTs[...,1,:] - KRTs[...,2,:] * points[...,1:2] 
  Dmatrix = torch.cat((U,V), dim=1).transpose(1,2)    # size = M * P * 2N * 4
  A      = Dmatrix[...,:3]                            # size = M * P * 2N * 3
  At     = torch.transpose(A, 2, 3)                   # size = M * P * 3 * 2N
  AtA    = torch.matmul(At, A)                        # size = M * P * 3 * 3
  invAtA = torch.inverse( AtA )
  P3D    = torch.matmul(invAtA, torch.matmul(At, -Dmatrix[...,3:]))
  return P3D.view(batch_mv, -1, 3)

"""
def calculate_project_points(multiview_real_locs, multiview_krts):
  assert multiview_krts.dim() == 4 and multiview_krts.size(2) == 3 and multiview_krts.size(3) == 4, 'invalid multiview_krts size : {:}'.format(multiview_krts)
  assert multiview_krts.size(0) == multiview_real_locs.size(0) and multiview_krts.size(1) == multiview_real_locs.size(1)
  batch, views, points, _ = multiview_real_locs.size()
  points3Ds = []
  for Mlocs, Mkrts in zip(multiview_real_locs, multiview_krts):
    _points3D = TriangulateDLT_BatchPoints(Mkrts, Mlocs)
    points3Ds.append( _points3D )
  points3Ds = torch.stack(points3Ds)
  projPoints2Ds = ProjectKRT_Batch(multiview_krts, points3Ds.view(batch, 1, points, 3))
  return projPoints2Ds
"""
