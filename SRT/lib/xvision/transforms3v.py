# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import torch, time
import sys, math, random, PIL
from PIL import Image, ImageOps
import numpy as np, numbers, types
from pathlib import Path
from .affine_utils import identity2affine
from .affine_utils import crop2affine
from .affine_utils import offset2affine
from .affine_utils import scale2affine
from .affine_utils import rotate2affine
from .affine_utils import horizontalmirror2affine
from .functional   import to_tensor


def get_HW(image):
  if isinstance(image, list) or isinstance(image, tuple): image = image[0]
  assert isinstance(image, torch.Tensor), 'invalid type : {:}'.format( type(image) )
  C, H, W = image.size()
  return H, W


class Compose3V(object):
  def __init__(self, transforms):
    self.transforms = transforms

  def __repr__(self):
    xstrs = [str(x) for x in self.transforms]
    xstr = ', '.join(xstrs)
    return ('{name}('.format(name=self.__class__.__name__, **self.__dict__) + xstr + ')')

  def __call__(self, img, point, index):
    assert not isinstance(point, list) and not isinstance(point, tuple), 'invalid point type : {:}'.format( type(point) )
    point, theta = point.copy(), identity2affine(True)
    
    for t in self.transforms:
      with torch.no_grad():
        if isinstance(t, RandomTrans):
          img, point, theta = t(img, point, theta, index)
        else:
          img, point, theta = t(img, point, theta)
      #print('{:} : {:}'.format(t, time.time()))
    return img, point, theta


class ToTensor(object):
  def __repr__(self):
    return ('{name}()'.format(name=self.__class__.__name__, **self.__dict__))
  def __call__(self, pic, point, theta):
    return to_tensor(pic), point.copy(), theta


class Normalize(object):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std
  def __repr__(self):
    return ('{name}(mean={mean}, std={std}])'.format(name=self.__class__.__name__, **self.__dict__))
  def __call__(self, tensor, point, theta):
    assert tensor.size(0) == len(self.mean) == len(self.std), '{:} vs {:} vs {:}'.format(tensor.size(), len(self.mean), len(self.std))
    for t, m, s in zip(tensor, self.mean, self.std):
      t.sub_(m).div_(s)
    return tensor, point.copy(), theta


class PreCrop(object):
  def __init__(self, expand_ratio):
    assert expand_ratio is None or isinstance(expand_ratio, numbers.Number), 'The expand-ratio should not be {:}.'.format(expand_ratio)
    if expand_ratio is None: self.expand_ratio = 0
    else                   : self.expand_ratio = expand_ratio
    assert self.expand_ratio >= 0, 'The expand-ratio should not be {:}.'.format(expand_ratio)
  def __repr__(self):
    return ('{name}(expand={expand_ratio}])'.format(name=self.__class__.__name__, **self.__dict__))
  def func(self, h, w, box, theta):
    face_ex_w, face_ex_h = (box[2] - box[0]) * self.expand_ratio, (box[3] - box[1]) * self.expand_ratio
    x1, y1 = max(box[0]-face_ex_w, 0.0), max(box[1]-face_ex_h, 0.0)
    x2, y2 = min(box[2]+face_ex_w, w-1), min(box[3]+face_ex_h, h-1)
    xtheta = crop2affine((x1,y1,x2,y2), w, h)
    xtheta = torch.mm(theta, xtheta)
    return xtheta
  def __call__(self, img, point, theta):
    assert not isinstance(img, list) and not isinstance(point, list) and not isinstance(theta, list)
    (H, W) = get_HW( img )
    if point.get_box() is None: box = [0, 0, W, H]
    else                      : box = point.get_box().tolist()
    return_theta = self.func(H, W, box, theta)
    return img, point.copy(), return_theta


class RandomTrans(object):
  
  def __init__(self, scale, offset, rotate, iters, cache_dir, check_cache):
    assert scale  >= 0.0 and scale  <= 1.0  , 'invalid scale  value : {:}'.format(scale )
    assert offset >= 0.0 and offset <= 1.0  , 'invalid offset value : {:}'.format(offset)
    assert rotate >= 0 and rotate <= 360, 'invalid rotate value : {:}'.format(rotate)
    assert isinstance(iters, int) and iters > 0, 'invalid iters : {:}'.format(iters)
    self.scaleRange = [1-scale, 1+scale]
    self.offset_max = offset
    self.rotate_max = rotate
    cache_dir, num  = Path(cache_dir), 100000
    assert cache_dir.exists(), 'invalid cache direcroty : {:}'.format(cache_dir)
    robust_param_path = cache_dir / 'scale-{:}-offset-{:}-rotate-{:}-N{:}.pth'.format(scale, offset, rotate, num)
    print ('robust_param_path : {:}'.format(robust_param_path))
    if check_cache:
      assert robust_param_path.exists(), 'can not find {:}'.format( robust_param_path )
      self.params = torch.load(robust_param_path)
      print ('directly load params from cache : {:}'.format(robust_param_path))
    else:
      self.params = tuple( self.get_transform_params() for i in range(num) )
      print ('randomly generate params')
      # torch.save(self.params, robust_param_path)
    assert isinstance(self.params, tuple), 'invalid type : {:}'.format( type(self.params) )
    self.iters      = iters

  def get_transform_params(self):
    scale = random.uniform(self.scaleRange[0], self.scaleRange[1])
    offx = random.uniform(-1, 1) * self.offset_max
    offy = random.uniform(-1, 1) * self.offset_max
    degree = random.uniform(-self.rotate_max, self.rotate_max)
    return {'scale': scale, 'offset': (offx, offy), 'degree': degree}

  def __repr__(self):
    return ('{name}(scale={scaleRange}, offset={offset_max}, rotate={rotate_max}, iters={iters})'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, imgs, point_meta, theta, index):
    point_meta = point_meta.copy()

    thetas = []
    for _iter in range(self.iters):
      xindex = ( index * self.iters + _iter ) % len(self.params)
      xparam = self.params[ xindex ]
      #print ( '[{:}] : {:}'.format(xindex, xparam) )
      # scale: 
      _theta = theta.clone()
      _theta = torch.mm(_theta, scale2affine(xparam['scale'], xparam['scale']))
      # random crop
      parameters = offset2affine(xparam['offset'][0], xparam['offset'][1])
      _theta = torch.mm(_theta, parameters)
      # random 
      degree = xparam['degree']
      if degree < 0: degree = 360 + degree
      _theta = torch.mm(_theta, rotate2affine(degree))
      thetas.append(_theta)
    return imgs, point_meta, thetas
