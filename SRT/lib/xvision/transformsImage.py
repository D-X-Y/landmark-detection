# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import torch, cv2
import sys, math, random, PIL
from PIL import Image, ImageOps
import numpy as np
import numbers
import types
from .affine_utils import identity2affine
from .affine_utils import crop2affine
from .affine_utils import offset2affine
from .affine_utils import scale2affine
from .affine_utils import rotate2affine
from .affine_utils import horizontalmirror2affine


def sample_from_bounded_gaussian(x):
  return max(-2*x, min(2*x, random.gauss(0,1)*x))


class Compose2V(object):
  def __init__(self, transforms):
    self.transforms = transforms

  def __repr__(self):
    xstrs = [str(x) for x in self.transforms]
    xstr = ', '.join(xstrs)
    return ('{name}('.format(name=self.__class__.__name__, **self.__dict__) + xstr + ')')

  def __call__(self, img, point, init_theta=None):
    if init_theta is None:
      theta = identity2affine(True)
    else:
      theta = init_thet.clone()
    for t in self.transforms:
      img, point, theta = t(img, point, theta)
    return img, point, theta


class ToPILImage(object):
  """Convert a tensor to PIL Image.
  Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
  H x W x C to a PIL.Image while preserving the value range.
  """
  def __init__(self, normalize=None, return_mode='PIL'):
    if normalize is None:
      self.mean = None
      self.std = None
    else:
      self.mean = normalize.mean
      self.std  = normalize.std
    self.return_mode = return_mode

  def __repr__(self):
    return ('{name}(mean={mean}, std={std}, mode={return_mode})'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, pic):
    """
    Args:
      pic (Tensor or numpy.ndarray): Image to be converted to PIL.Image.
    Returns:
      PIL.Image: Image converted to PIL.Image.
    """
    xinput = []
    with torch.no_grad():
      for idx, t in enumerate(pic):
        if self.std is not None:
          t = torch.mul(t, self.std[idx])
        if self.mean is not None:
          t = torch.add(t, self.mean[idx])
        xinput.append( t )
      pic = torch.stack(xinput).cpu()

    npimg = pic
    mode = None
    if isinstance(pic, torch.FloatTensor):
      pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
      npimg = np.transpose(pic.numpy(), (1, 2, 0))
    assert isinstance(npimg, np.ndarray), 'pic should be Tensor or ndarray'
    if npimg.shape[2] == 1:
      npimg = npimg[:, :, 0]

      if npimg.dtype == np.uint8:
        mode = 'L'
      if npimg.dtype == np.int16:
        mode = 'I;16'
      if npimg.dtype == np.int32:
        mode = 'I'
      elif npimg.dtype == np.float32:
        mode = 'F'
    else:
      if npimg.dtype == np.uint8:
        mode = 'RGB'
    assert mode is not None, '{:} is not supported'.format(npimg.dtype)
    if self.return_mode == 'PIL':
      return Image.fromarray(npimg, mode=mode)
    elif self.return_mode == 'cv2':
      if npimg.ndim == 3: npimg = npimg[:,:,::-1]
      return npimg
    elif self.return_mode == 'cv2gray':
      if npimg.ndim == 2: return npimg
      else              : return cv2.cvtColor(npimg[:,:,::-1], cv2.COLOR_BGR2GRAY)
    else: raise ValueError('invalid return_mode : {:}'.format(self.return_mode))


class ToTensor(object):
  """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
  Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
  [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
  """

  def __repr__(self):
    return ('{name}()'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, pics, points, theta):
    """
    Args:
      pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
      points 3 * N numpy.ndarray [x, y, visiable] or Point_Meta
    Returns:
      Tensor: Converted image.
    """
    ## add to support list
    if isinstance(pics, list): is_list = True
    else:                      is_list, pics = False, [pics]

    returned = []
    for pic in pics:
      if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        returned.append( img.float().div(255) )
        continue

      # handle PIL Image
      if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
      elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
      else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
      # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
      if pic.mode == 'YCbCr':
        nchannel = 3
      elif pic.mode == 'I;16':
        nchannel = 1
      else:
        nchannel = len(pic.mode)
      img = img.view(pic.size[1], pic.size[0], nchannel)
      # put it from HWC to CHW format
      # yikes, this transpose takes 80% of the loading time/CPU
      img = img.transpose(0, 1).transpose(0, 2).contiguous()
      if isinstance(img, torch.ByteTensor):
        img = img.float().div(255)
      returned.append(img)

    if is_list == False:
      assert len(returned) == 1, 'For non-list data, length of answer must be one not {}'.format(len(returned))
      returned = returned[0]

    return returned, points, theta.clone()


class ColorDisturb(object):
  def __init__(self, scale_max):
    assert isinstance(scale_max, numbers.Number) and scale_max>0 and scale_max<1, 'The scale_max is wrong : {:}'.format(scale_max)
    self.scale_max = scale_max
  def __repr__(self):
    return ('{name}(scale={scale_max})'.format(name=self.__class__.__name__, **self.__dict__))
  def __call__(self, tensors, points, theta):
    if isinstance(tensors, list): is_list = True
    else                        : is_list, tensors = False, [tensors]
    for tensor in tensors:
      for t in tensor:
        t.mul_( random.uniform(1-self.scale_max, 1+self.scale_max) ).clamp_(0, 1)
    if is_list == False: tensors = tensors[0]
    return tensors, points, theta.clone()


class Normalize(object):
  """Normalize an tensor image with mean and standard deviation.
  Given mean: (R, G, B) and std: (R, G, B),
  will normalize each channel of the torch.*Tensor, i.e.
  channel = (channel - mean) / std
  Args:
    mean (sequence): Sequence of means for R, G, B channels respecitvely.
    std (sequence): Sequence of standard deviations for R, G, B channels
      respecitvely.
  """

  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __repr__(self):
    return ('{name}(mean={mean}, std={std}])'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, tensors, points, theta):
    """
    Args:
      tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
      Tensor: Normalized image.
    """
    # TODO: make efficient
    if isinstance(tensors, list): is_list = True
    else                        : is_list, tensors = False, [tensors]

    for tensor in tensors:
      assert tensor.size(0) == len(self.mean) == len(self.std), '{:} vs {:} vs {:}'.format(tensor.size(), len(self.mean), len(self.std))
      for t, m, s in zip(tensor, self.mean, self.std):
        t.sub_(m).div_(s)
    
    if is_list == False: tensors = tensors[0]

    return tensors, points, theta.clone()


class PreCrop(object):

  def __init__(self, expand_ratio):
    assert expand_ratio is None or isinstance(expand_ratio, numbers.Number), 'The expand_ratio should not be {}'.format(expand_ratio)
    if expand_ratio is None:
      self.expand_ratio = 0
    else:
      self.expand_ratio = expand_ratio
    assert self.expand_ratio >= 0, 'The expand_ratio should not be {}'.format(expand_ratio)

  def __repr__(self):
    return ('{name}(expand={expand_ratio}])'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, imgs, point_meta, theta):
    point_meta = point_meta.copy()

    if isinstance(imgs, list):
      _, h, w = imgs[0].size()
    else:
      _, h, w = imgs.size()

    box = point_meta.get_box().tolist()
    face_ex_w, face_ex_h = (box[2] - box[0]) * self.expand_ratio, (box[3] - box[1]) * self.expand_ratio
    x1, y1 = max(box[0]-face_ex_w, 0.0), max(box[1]-face_ex_h, 0.0)
    x2, y2 = min(box[2]+face_ex_w, w-1), min(box[3]+face_ex_h, h-1)
    xtheta = crop2affine((x1,y1,x2,y2), w, h)
    xtheta = torch.mm(theta, xtheta)

    return imgs, point_meta, xtheta


class RandomOffset(object):

  def __init__(self, ratios):
    if ratios is None:
      ratios = (0, 0)
    elif isinstance(ratios, numbers.Number):
      ratios = (ratios, ratios)
    assert isinstance(ratios, tuple) and len(ratios) == 2, 'ratios is wrong : {:}'.format(ratios)
    self.vertical_ratio   = ratios[0]
    self.horizontal_ratio = ratios[1]

  def __repr__(self):
    return ('{name}(vertical={vertical_ratio}, horizontal={horizontal_ratio}])'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, imgs, point_meta, theta):
    point_meta = point_meta.copy()
  
    offx = random.uniform(-1, 1) * self.horizontal_ratio
    offy = random.uniform(-1, 1) * self.vertical_ratio
    parameters = offset2affine(offx, offy)
    xtheta = torch.mm(theta, parameters)
    return imgs, point_meta, xtheta


class AugScale(object):

  def __init__(self, scale_prob, scale_min, scale_max):
    assert isinstance(scale_prob, numbers.Number) and scale_prob >= 0, 'scale_prob : {:}'.format(scale_prob)
    assert isinstance(scale_min,  numbers.Number) and isinstance(scale_max, numbers.Number), 'scales : {:}, {:}'.format(scale_min, scale_max)
    self.scale_prob = scale_prob
    self.scale_min  = scale_min
    self.scale_max  = scale_max

  def __repr__(self):
    return ('{name}(prob={scale_prob}, range=[{scale_min}, {scale_max}])'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, imgs, point_meta, theta):
    point_meta = point_meta.copy()

    dice = random.random()
    if dice > self.scale_prob:
      return imgs, point_meta, theta.clone()

    scale = random.uniform(self.scale_min, self.scale_max)

    parameters = scale2affine(scale, scale)
    xtheta = torch.mm(theta, parameters)
    return imgs, point_meta, xtheta


class CenterCrop(object):

  def __init__(self, ratios):
    if isinstance(ratios, numbers.Number):
      ratios = (ratios, ratios)
    if ratios is None:
      self.ratios = ratios
    else:
      assert isinstance(ratios, tuple) and len(ratios) == 2, 'Invalid ratios : {:}'.format(ratios)
      self.ratios = ratios
      assert ratios[0] <= 1.0 and ratios[1] <= 1.0

  def __repr__(self):
    return ('{name}(range={ratios})'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, imgs, point_meta, theta):
    if self.ratios is None: return imgs, point_meta, theta
    point_meta = point_meta.copy()

    xtheta = scale2affine(*self.ratios)
    xparam = torch.mm(theta, xtheta)
    return imgs, point_meta, xparam


class AugCrop(object):

  def __init__(self, ratios):
    if isinstance(ratios, numbers.Number):
      ratios = (ratios, ratios)
    if ratios is None:
      self.ratios = ratios
    else:
      assert isinstance(ratios, tuple) and len(ratios) == 2, 'Invalid ratios : {:}'.format(ratios)
      self.ratios = ratios
      assert ratios[0] <= 1.0 and ratios[1] <= 1.0

  def __repr__(self):
    return ('{name}(range={ratios})'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, imgs, point_meta, theta):
    if self.ratios is None: return imgs, point_meta, theta

    point_meta = point_meta.copy()

    offx = (1-self.ratios[0])/2/self.ratios[0]
    offy = (1-self.ratios[1])/2/self.ratios[1]
    OFFX, OFFY = random.uniform(-offx, offx), random.uniform(-offy, offy)

    offsetP = offset2affine(OFFX, OFFY)
    scalerP = scale2affine(*self.ratios)
    xtheta = torch.mm(scalerP, offsetP)
    xparam = torch.mm(theta, xtheta)
    return imgs, point_meta, xparam


class AugRotate(object):

  def __init__(self, max_rotate_degree, rotate_prob=1):
    assert isinstance(max_rotate_degree, numbers.Number), 'max_rotate_degree : {:}'.format(max_rotate_degree)
    assert isinstance(rotate_prob, numbers.Number) and rotate_prob>=0 and rotate_prob<=1, 'The probablity is wrong : {:}'.format(rotate_prob)
    self.max_rotate_degree = max_rotate_degree
    self.rotate_prob = rotate_prob

  def __repr__(self):
    return ('{name}(max-degree={max_rotate_degree})'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, imgs, point_meta, theta):
    point_meta = point_meta.copy()

    if random.random() < self.rotate_prob:
      #degree = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
      degree = sample_from_bounded_gaussian(self.max_rotate_degree)
      if degree < 0: degree = 360 + degree
      params = rotate2affine(degree)
      theta = torch.mm(theta, params)
    return imgs, point_meta, theta


class AugHorizontalFlip(object):
  def __init__(self, p=0.5):
    assert isinstance(p, numbers.Number) and p>=0 and p<=1, 'The probablity is wrong : {:}'.format(p)
    self.probablity = p
  
  def __repr__(self):
    return ('{name}(flip_probability={max_rotate_degree})'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, imgs, point_meta, theta):
    point_meta = point_meta.copy()

    if random.random() < self.probablity:
      point_meta.apply_horizontal_flip()
      params = horizontalmirror2affine()
      theta = torch.mm(theta, params)
    return imgs, point_meta, theta


class RandomTransf(object):
  
  def __init__(self, scales, offset, rotate, iters):
    assert isinstance(scales, tuple) or isinstance(scales, list), 'scales were wrong : {:}'.format(scales)
    assert scales[0] < scales[1], 'scales : {:}'.format(scales)
    assert offset >= 0.0 and offset <= 1.0  , 'invalid crop value : {:}'.format(offset)
    assert rotate >= 0 and rotate <= 360, 'invalid rotate value : {:}'.format(rotate)
    assert isinstance(iters, int) and iters > 0, 'invalid iters : {:}'.format(iters)
    self.scale_range = scales
    self.offset_max = offset
    self.rotate_max = rotate
    self.iters = iters

  def __repr__(self):
    return ('{name}(scale={scale_range}, offset={offset_max}, rotate={rotate_max}, iters={iters})'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, imgs, point_meta, theta):
    point_meta = point_meta.copy()

    thetas = []
    for _iter in range(self.iters):
      # scale: 
      _theta = theta.clone()
      scale = random.uniform(self.scale_range[0], self.scale_range[1])
      _theta = torch.mm(_theta, scale2affine(scale, scale))
      # random crop
      offx = random.uniform(-1, 1) * self.offset_max
      offy = random.uniform(-1, 1) * self.offset_max
      parameters = offset2affine(offx, offy)
      _theta = torch.mm(_theta, parameters)
      # random 
      degree = random.uniform(-self.rotate_max, self.rotate_max)
      if degree < 0: degree = 360 + degree
      _theta = torch.mm(_theta, rotate2affine(degree))
      thetas.append(_theta)
    return imgs, point_meta, thetas
