# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from scipy.ndimage.interpolation import zoom
import numbers, math
import numpy as np

## pts = 3 * N numpy array; points location is based on the image with size (height*downsample, width*downsample)

def generate_label_map(pts, height, width, sigma, downsample, nopoints, ctype):
  #if isinstance(pts, numbers.Number):
  # this image does not provide the annotation, pts is a int number representing the number of points
  #return np.zeros((height,width,pts+1), dtype='float32'), np.ones((1,1,1+pts), dtype='float32')
  # nopoints == True means this image does not provide the annotation, pts is a int number representing the number of points
  
  assert isinstance(pts, np.ndarray) and len(pts.shape) == 2 and pts.shape[0] == 3, 'The shape of points : {}'.format(pts.shape)
  if isinstance(sigma, numbers.Number):
    sigma = np.zeros((pts.shape[1])) + sigma
  assert isinstance(sigma, np.ndarray) and len(sigma.shape) == 1 and sigma.shape[0] == pts.shape[1], 'The shape of sigma : {}'.format(sigma.shape)

  offset = downsample / 2.0 - 0.5
  num_points, threshold = pts.shape[1], 0.01

  if nopoints == False: visiable = pts[2, :].astype('bool')
  else                : visiable = (pts[2, :]*0).astype('bool')
  #assert visiable.shape[0] == num_points

  transformed_label = np.fromfunction( lambda y, x, pid : ((offset + x*downsample - pts[0,pid])**2 \
                                                        + (offset + y*downsample - pts[1,pid])**2) \
                                                          / -2.0 / sigma[pid] / sigma[pid],
                                                          (height, width, num_points), dtype=int)

  mask_heatmap      = np.ones((1, 1, num_points+1), dtype='float32')
  mask_heatmap[0, 0, :num_points] = visiable
  mask_heatmap[0, 0, num_points]  = (nopoints==False)
  
  if ctype == 'laplacian':
    transformed_label = (1+transformed_label) * np.exp(transformed_label)
  elif ctype == 'gaussian':
    transformed_label = np.exp(transformed_label)
  else:
    raise TypeError('Does not know this type [{:}] for label generation'.format(ctype))
  transformed_label[ transformed_label < threshold ] = 0
  transformed_label[ transformed_label >         1 ] = 1
  transformed_label = transformed_label * mask_heatmap[:, :, :num_points]

  background_label  = 1 - np.amax(transformed_label, axis=2)
  background_label[ background_label < 0 ] = 0
  heatmap           = np.concatenate((transformed_label, np.expand_dims(background_label, axis=2)), axis=2).astype('float32')
  
  return heatmap*mask_heatmap, mask_heatmap
