##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
from os import path as osp
import numpy as np
from numpy import linspace
from matplotlib import cm
import datasets

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

def overlap_two_pil_image(imageA, imageB):
  assert isinstance(imageA, Image.Image), 'The 1-th image type is not PIL.Image.Image'
  assert isinstance(imageB, Image.Image), 'The 2-th image type is not PIL.Image.Image'
  width = max(imageA.size[0], imageB.size[0])
  height = max(imageA.size[1], imageB.size[1])
  imageA = imageA.resize((width, height), Image.BICUBIC)
  imageB = imageB.resize((width, height), Image.BICUBIC)
  image = (np.array(imageA) + np.array(imageB) * 1.0) / 2.0
  return Image.fromarray(np.uint8(image))

def draw_image_with_pts(_image, pts, radius = 10, color = (193,255,193), fontScale = 16, text_color = (255,255,255), linewidth = 3, window = None):
  '''
  visualize image and plot keypoints on top of it
  parameter:
    image:          PIL.Image.Image object
    pts:            (2 or 3) x num_pts numpy array : [x, y, visiable]
  '''
  if isinstance(_image, str):
    _image = datasets.pil_loader(_image)
  assert isinstance(_image, Image.Image), 'image type is not PIL.Image.Image'
  assert isinstance(pts, np.ndarray) and (pts.shape[0] == 2 or pts.shape[0] == 3), 'input points are not correct'
  image = _image.copy()
  draw  = ImageDraw.Draw(image)

  try:
    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', fontScale)
  except:
    font_path = osp.join(os.environ['HOME'], '.fonts', 'freefont', 'FreeMono.ttf')
    font = ImageFont.truetype(font_path, fontScale)

  line_radius = abs(radius)
  for idx in range(pts.shape[1]):
    #if (pts.shape[0] == 2 or pts[2, idx] >= 1): # visiable
    if (pts.shape[0] == 2 or (pts[0, idx] > 0 and pts[1, idx] > 0)): # visiable
      # draw hollow circle
      point = (int(pts[0,idx])-radius, int(pts[1,idx])-radius, int(pts[0,idx])+radius, int(pts[1,idx])+radius)
      if radius > 0:
        draw.ellipse(point, fill=None, outline=color)

      point = (int(pts[0,idx]-line_radius), int(pts[1,idx]-line_radius), int(pts[0,idx]+line_radius), int(pts[1,idx]+line_radius))
      draw.line(point, fill=color, width = linewidth)

      point = (int(pts[0,idx]-line_radius), int(pts[1,idx]+line_radius), int(pts[0,idx]+line_radius), int(pts[1,idx]-line_radius))
      draw.line(point, fill=color, width = linewidth)
      
      point = (int(pts[0,idx]+line_radius), int(pts[1,idx]-line_radius))
      draw.text(point, '{}'.format(idx+1), fill=text_color, font=font)
  
      if window is not None:
        assert isinstance(window, int), 'The window is not ok : {}'.format(window)
        point = (int(pts[0,idx]-window), int(pts[1,idx]-window), int(pts[0,idx]-window), int(pts[1,idx]+window))
        draw.line(point, fill=color, width = 1)
        point = (int(pts[0,idx]+window), int(pts[1,idx]-window), int(pts[0,idx]+window), int(pts[1,idx]+window))
        draw.line(point, fill=color, width = 1)
        point = (int(pts[0,idx]-window), int(pts[1,idx]-window), int(pts[0,idx]+window), int(pts[1,idx]-window))
        draw.line(point, fill=color, width = 1)
        point = (int(pts[0,idx]-window), int(pts[1,idx]+window), int(pts[0,idx]+window), int(pts[1,idx]+window))
        draw.line(point, fill=color, width = 1)
        window = None
    
  #point = (int(image.size[0]/2.)-radius*3, int(image.size[1]/2.0)-radius*3, int(image.size[0]/2.)+radius*3, int(image.size[1]/2.0)+radius*3)
  #draw.ellipse(point, fill=(0,0,0), outline=(0,0,0))

  return image


def mat2im(mat, cmap, limits):
  '''
% PURPOSE
% Uses vectorized code to convert matrix "mat" to an m-by-n-by-3
% image matrix which can be handled by the Mathworks image-processing
% functions. The the image is created using a specified color-map
% and, optionally, a specified maximum value. Note that it discards
% negative values!
%
% INPUTS
% mat     - an m-by-n matrix  
% cmap    - an m-by-3 color-map matrix. e.g. hot(100). If the colormap has 
%           few rows (e.g. less than 20 or so) then the image will appear 
%           contour-like.
% limits  - by default the image is normalised to it's max and min values
%           so as to use the full dynamic range of the
%           colormap. Alternatively, it may be normalised to between
%           limits(1) and limits(2). Nan values in limits are ignored. So
%           to clip the max alone you would do, for example, [nan, 2]
%          
%
% OUTPUTS
% im - an m-by-n-by-3 image matrix  
  '''

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
