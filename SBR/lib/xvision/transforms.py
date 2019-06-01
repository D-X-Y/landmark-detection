# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import torch
import sys, math, random, PIL
from PIL import Image, ImageOps
import numpy as np
import numbers
import types
import collections

if sys.version_info.major == 2:
  import cPickle as pickle
else:
  import pickle

class Compose(object):
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img, points):
    for t in self.transforms:
      img, points = t(img, points)
    return img, points

class TrainScale2WH(object):
  """Rescale the input PIL.Image to the given size.
  Args:
    size (sequence or int): Desired output size. If size is a sequence like
      (w, h), output size will be matched to this. If size is an int,
      smaller edge of the image will be matched to this number.
      i.e, if height > width, then image will be rescaled to
      (size * height / width, size)
    interpolation (int, optional): Desired interpolation. Default is
      ``PIL.Image.BILINEAR``
  """

  def __init__(self, target_size, interpolation=Image.BILINEAR):
    assert isinstance(target_size, tuple) or isinstance(target_size, list), 'The type of target_size is not right : {}'.format(target_size)
    assert len(target_size) == 2, 'The length of target_size is not right : {}'.format(target_size)
    assert isinstance(target_size[0], int) and isinstance(target_size[1], int), 'The type of target_size is not right : {}'.format(target_size)
    self.target_size   = target_size
    self.interpolation = interpolation

  def __call__(self, imgs, point_meta):
    """
    Args:
      img (PIL.Image): Image to be scaled.
      points 3 * N numpy.ndarray [x, y, visiable]
    Returns:
      PIL.Image: Rescaled image.
    """
    point_meta = point_meta.copy()

    if isinstance(imgs, list): is_list = True
    else:                      is_list, imgs = False, [imgs]
    
    w, h = imgs[0].size
    ow, oh = self.target_size[0], self.target_size[1]
    point_meta.apply_scale( [ow*1./w, oh*1./h] )

    imgs = [ img.resize((ow, oh), self.interpolation) for img in imgs ]
    if is_list == False: imgs = imgs[0]

    return imgs, point_meta



class ToPILImage(object):
  """Convert a tensor to PIL Image.
  Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
  H x W x C to a PIL.Image while preserving the value range.
  """

  def __call__(self, pic):
    """
    Args:
      pic (Tensor or numpy.ndarray): Image to be converted to PIL.Image.
    Returns:
      PIL.Image: Image converted to PIL.Image.
    """
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
    assert mode is not None, '{} is not supported'.format(npimg.dtype)
    return Image.fromarray(npimg, mode=mode)



class ToTensor(object):
  """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
  Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
  [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
  """

  def __call__(self, pics, points):
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

    return returned, points


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

  def __call__(self, tensors, points):
    """
    Args:
      tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
      Tensor: Normalized image.
    """
    # TODO: make efficient
    if isinstance(tensors, list): is_list = True
    else:                         is_list, tensors = False, [tensors]

    for tensor in tensors:
      for t, m, s in zip(tensor, self.mean, self.std):
        t.sub_(m).div_(s)
    
    if is_list == False: tensors = tensors[0]

    return tensors, points


class PreCrop(object):
  """Crops the given PIL.Image at the center.

  Args:
    size (sequence or int): Desired output size of the crop. If size is an
      int instead of sequence like (w, h), a square crop (size, size) is
      made.
  """

  def __init__(self, expand_ratio):
    assert expand_ratio is None or isinstance(expand_ratio, numbers.Number), 'The expand_ratio should not be {}'.format(expand_ratio)
    if expand_ratio is None:
      self.expand_ratio = 0
    else:
      self.expand_ratio = expand_ratio
    assert self.expand_ratio >= 0, 'The expand_ratio should not be {}'.format(expand_ratio)

  def __call__(self, imgs, point_meta):
    ## AugCrop has something wrong... For unsupervised data

    if isinstance(imgs, list): is_list = True
    else:                      is_list, imgs = False, [imgs]

    w, h = imgs[0].size
    box = point_meta.get_box().tolist()
    face_ex_w, face_ex_h = (box[2] - box[0]) * self.expand_ratio, (box[3] - box[1]) * self.expand_ratio
    x1, y1 = int(max(math.floor(box[0]-face_ex_w), 0)), int(max(math.floor(box[1]-face_ex_h), 0))
    x2, y2 = int(min(math.ceil(box[2]+face_ex_w), w)), int(min(math.ceil(box[3]+face_ex_h), h))
    
    imgs = [ img.crop((x1, y1, x2, y2)) for img in imgs ]
    point_meta.set_precrop_wh( imgs[0].size[0], imgs[0].size[1], x1, y1, x2, y2)
    point_meta.apply_offset(-x1, -y1)
    point_meta.apply_bound(imgs[0].size[0], imgs[0].size[1])

    if is_list == False: imgs = imgs[0]
    return imgs, point_meta


class AugScale(object):
  """Rescale the input PIL.Image to the given size.

  Args:
    size (sequence or int): Desired output size. If size is a sequence like
      (w, h), output size will be matched to this. If size is an int,
      smaller edge of the image will be matched to this number.
      i.e, if height > width, then image will be rescaled to
      (size * height / width, size)
    interpolation (int, optional): Desired interpolation. Default is
      ``PIL.Image.BILINEAR``
  """

  def __init__(self, scale_prob, scale_min, scale_max, interpolation=Image.BILINEAR):
    assert isinstance(scale_prob, numbers.Number) and scale_prob >= 0, 'scale_prob : {:}'.format(scale_prob)
    assert isinstance(scale_min,  numbers.Number) and isinstance(scale_max, numbers.Number), 'scales : {:}, {:}'.format(scale_min, scale_max)
    self.scale_prob = scale_prob
    self.scale_min  = scale_min
    self.scale_max  = scale_max
    self.interpolation = interpolation

  def __call__(self, imgs, point_meta):
    """
    Args:
      img (PIL.Image): Image to be scaled.
      points 3 * N numpy.ndarray [x, y, visiable]
    Returns:
      PIL.Image: Rescaled image.
    """
    point_meta = point_meta.copy()

    dice = random.random()
    if dice > self.scale_prob:
      return imgs, point_meta

    if isinstance(imgs, list): is_list = True
    else:                      is_list, imgs = False, [imgs]
    
    scale_multiplier = (self.scale_max - self.scale_min) * random.random() + self.scale_min
    
    w, h = imgs[0].size
    ow, oh = int(w * scale_multiplier), int(h * scale_multiplier)

    imgs = [ img.resize((ow, oh), self.interpolation) for img in imgs ]
    point_meta.apply_scale( [scale_multiplier] )

    if is_list == False: imgs = imgs[0]

    return imgs, point_meta


class AugCrop(object):

  def __init__(self, crop_x, crop_y, center_perterb_max, fill=0):
    assert isinstance(crop_x, int) and isinstance(crop_y, int) and isinstance(center_perterb_max, numbers.Number)
    self.crop_x = crop_x
    self.crop_y = crop_y
    self.center_perterb_max = center_perterb_max
    assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
    self.fill   = fill

  def __call__(self, imgs, point_meta=None):
    ## AugCrop has something wrong... For unsupervised data
  
    point_meta = point_meta.copy()
    if isinstance(imgs, list): is_list = True
    else:                      is_list, imgs = False, [imgs]

    dice_x, dice_y = random.random(), random.random()
    x_offset = int( (dice_x-0.5) * 2 * self.center_perterb_max)
    y_offset = int( (dice_y-0.5) * 2 * self.center_perterb_max)
    
    x1 = int(round( point_meta.center[0] + x_offset - self.crop_x / 2. ))
    y1 = int(round( point_meta.center[1] + y_offset - self.crop_y / 2. ))
    x2 = x1 + self.crop_x
    y2 = y1 + self.crop_y

    w, h = imgs[0].size
    if x1 < 0 or x2 < 0 or x2 >= w or y2 >= h:
      pad = max(0-x1, 0-x2, x2-w+1, y2-h+1)
      assert pad > 0, 'padding operation in crop must be greater than 0'
      imgs = [ ImageOps.expand(img, border=pad, fill=self.fill) for img in imgs ]
      x1, x2, y1, y2 = x1 + pad, x2 + pad, y1 + pad, y2 + pad
      point_meta.apply_offset(pad, pad)
      point_meta.apply_bound(imgs[0].size[0], imgs[0].size[1])

    point_meta.apply_offset(-x1, -y1)
    imgs = [ img.crop((x1, y1, x2, y2)) for img in imgs ]
    point_meta.apply_bound(imgs[0].size[0], imgs[0].size[1])

    if is_list == False: imgs = imgs[0]
    return imgs, point_meta

class AugRotate(object):
  """Rotate the given PIL.Image at the center.
  Args:
    size (sequence or int): Desired output size of the crop. If size is an
      int instead of sequence like (w, h), a square crop (size, size) is
      made.
  """

  def __init__(self, max_rotate_degree):
    assert isinstance(max_rotate_degree, numbers.Number)
    self.max_rotate_degree = max_rotate_degree

  def __call__(self, imgs, point_meta):
    """
    Args:
      img (PIL.Image): Image to be cropped.
      point_meta : Point_Meta
    Returns:
      PIL.Image: Rotated image.
    """
    point_meta = point_meta.copy()
    if isinstance(imgs, list): is_list = True
    else:                      is_list, imgs = False, [imgs]

    degree = (random.random() - 0.5) * 2 * self.max_rotate_degree
    center = (imgs[0].size[0] / 2, imgs[0].size[1] / 2)
    if PIL.__version__[0] == '4':
      imgs = [ img.rotate(degree, center=center) for img in imgs ]
    else:
      imgs = [ img.rotate(degree) for img in imgs ]

    point_meta.apply_rotate(center, degree)
    point_meta.apply_bound(imgs[0].size[0], imgs[0].size[1])

    if is_list == False: imgs = imgs[0]

    return imgs, point_meta
