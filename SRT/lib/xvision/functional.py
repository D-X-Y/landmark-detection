# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
from __future__ import division
import torch
import sys
import math
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
try:
  import accimage
except ImportError:
  accimage = None
import numpy as np


def _is_pil_image(img):
  if accimage is not None:
    return isinstance(img, (Image.Image, accimage.Image))
  else:
    return isinstance(img, Image.Image)


def _is_tensor_image(img):
  return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
  return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
  """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
  See ``ToTensor`` for more details.
  Args:
    pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
  Returns:
    Tensor: Converted image.
  """
  if not(_is_pil_image(pic) or _is_numpy_image(pic) or _is_tensor_image(pic)):
    raise TypeError('pic should be PIL Image or ndarray or Tensor. Got {}'.format(type(pic)))

  if _is_tensor_image(pic): return pic

  if isinstance(pic, np.ndarray):
    # handle numpy array
    if pic.ndim == 2:
      pic = pic[:, :, None]

    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    # backward compatibility
    if isinstance(img, torch.ByteTensor):
      return img.float().div(255)
    else:
      return img

  if accimage is not None and isinstance(pic, accimage.Image):
    nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
    pic.copyto(nppic)
    return torch.from_numpy(nppic)

  # handle PIL Image
  if pic.mode == 'I':
    img = torch.from_numpy(np.array(pic, np.int32, copy=False))
  elif pic.mode == 'I;16':
    img = torch.from_numpy(np.array(pic, np.int16, copy=False))
  elif pic.mode == 'F':
    img = torch.from_numpy(np.array(pic, np.float32, copy=False))
  elif pic.mode == '1':
    img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
  else:
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
  # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
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
    return img.float().div(255)
  else:
    return img
