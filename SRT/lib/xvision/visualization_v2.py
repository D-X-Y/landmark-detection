# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
from os import path as osp
import numpy as np
from numpy import linspace
from matplotlib import cm

def merge_images(images, gap, direction='y'):
  assert len(images) > 0 and isinstance(gap, int), 'The gap must be interge : {}'.format(gap)
  assert direction == 'x' or direction == 'y', 'The direction must be x or y, not {}'.format(direction)
  for index, image in enumerate(images):
    assert isinstance(image, Image.Image), 'The {}-th image is PIL.Image'.format(index)
  size = images[0].size
  images = [np.array(image) for image in images]
  imagelist = []
  for index, image in enumerate(images):
    if direction == 'y': gap_shape = [gap, size[0], 3]
    else:                gap_shape = [size[1], gap, 3]
    if index > 0: imagelist.append(np.zeros(gap_shape).astype('uint8'))
    imagelist.append(image)
  if direction == 'y': stack = np.vstack( imagelist )
  else:                stack = np.hstack( imagelist )
  return Image.fromarray( stack )
    
"""
def merge_images_matrix(images, gap):
  assert len(images) > 0 and isinstance(gap, int), 'The gap must be interge : {}'.format(gap)
  assert direction == 'x' or direction == 'y', 'The direction must be x or y, not {}'.format(direction)
  for index, image in enumerate(images):
    assert isinstance(image, Image.Image), 'The {}-th image is PIL.Image'.format(index)
  images = [np.array(image) for image in images]
  num_w = int( np.sqrt( len(images) ) )
  num_h = int( np.ceil( len(images)/num_w ) )
  assert False, 'Un finished'
  size = images[0].size
  imagelist = []
  for index, image in enumerate(images):
    if direction == 'y': gap_shape = [gap, size[0], 3]
    else:                gap_shape = [size[1], gap, 3]
    if index > 0: imagelist.append(np.zeros(gap_shape).astype('uint8'))
    imagelist.append(image)
  if direction == 'y': stack = np.vstack( imagelist )
  else:                stack = np.hstack( imagelist )
  return Image.fromarray( stack )
"""

def overlap_two_pil_image(imageA, imageB):
  assert isinstance(imageA, Image.Image), 'The 1-th image type is not PIL.Image.Image'
  assert isinstance(imageB, Image.Image), 'The 2-th image type is not PIL.Image.Image'
  width = max(imageA.size[0], imageB.size[0])
  height = max(imageA.size[1], imageB.size[1])
  imageA = imageA.resize((width, height), Image.BICUBIC)
  imageB = imageB.resize((width, height), Image.BICUBIC)
  imageA, imageB = imageA.convert('RGB'), imageB.convert('RGB')
  image = (np.array(imageA) + np.array(imageB) * 1.0) / 2.0
  return Image.fromarray(np.uint8(image))


def mat2im(mat, cmap, limits):

  assert len(mat.shape) == 2
  if len(limits) == 2:
    minVal = limits[0]
    tempss = np.zeros(mat.shape) + minVal
    mat    = np.maximum(tempss, mat)
    maxVal = limits[1]
    tempss = np.zeros(mat.shape) + maxVal
    mat    = np.minimum(tempss, mat)
  else:
    minVal = mat.min()
    maxVal = mat.max()
  L = len(cmap)
  if maxVal <= minVal:
    mat = mat-minVal
  else:
    mat = (mat-minVal) / (maxVal-minVal) * (L-1)
  mat = mat.astype(np.int32)
  
  image = np.reshape(cmap[ np.reshape(mat, (mat.size)), : ], mat.shape + (3,))
  return image

def jet(m):
  cm_subsection = linspace(0, 1, m)
  colors = [ cm.jet(x) for x in cm_subsection ]
  J = np.array(colors)
  J = J[:, :3]
  return J

def generate_color_from_heatmap(maps, num_of_color=100, index=None):
  assert isinstance(maps, np.ndarray)
  if len(maps.shape) == 3:
    return generate_color_from_heatmaps(maps, num_of_color, index)
  elif len(maps.shape) == 2:
    return mat2im( maps, jet(num_of_color), [maps.min(), maps.max()] )
  else:
    assert False, 'generate_color_from_heatmap wrong shape : {}'.format(maps.shape)
    

def generate_color_from_heatmaps(maps, num_of_color=100, index=None):
  assert isinstance(maps, np.ndarray) and len(maps.shape) == 3, 'maps type : {}'.format(type(maps))
  __jet = jet(num_of_color)

  if index is None:
    answer = []
    for i in range(maps.shape[2]):
      temp = mat2im( maps[:,:,i], __jet, [maps[:,:,i].min(), maps[:,:,i].max()] )
      answer.append( temp )
    return answer
  else:
    return mat2im( maps[:,:,index], __jet, [maps[:,:,index].min(), maps[:,:,index].max()] )
