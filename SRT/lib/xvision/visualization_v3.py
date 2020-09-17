# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os, torch
from pathlib import Path
import torch.nn.functional as F
import numpy as np


def pil_loader(path, use_gray=False):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(str(path), 'rb') as f:
    with Image.open(f) as img:
      if use_gray: return img.convert('L')
      else       : return img.convert('RGB')


def get_font():
  font_path = (Path(__file__).parent / '..' / '..' / '.fonts' / 'freefont' / 'FreeMono.ttf').resolve()
  assert font_path.exists(), 'Can not find : {:}'.format(font_path)
  return str( font_path )


def draw_image_by_points_major(_image, pts_A, pts_B, radius, color_A, color_B, crop, base_texts):
  if isinstance(_image, str):
    image = pil_loader(_image, False)
  else: image = _image
  assert isinstance(image, Image.Image), 'image type is not PIL.Image.Image'
  for pts in [pts_A, pts_B]:
    assert isinstance(pts, torch.Tensor) and (pts.shape[0] == 2 or pts.shape[0] == 3), 'input points are not correct : {:}'.format(pts)
  pts_A, pts_B = pts_A.clone(), pts_B.clone()

  num_points = pts_A.shape[1]
  visiable_points = []
  for idx in range(num_points):
    if (pts_A.shape[0] == 2 or bool(pts_A[2,idx]>0.5)) and \
       (pts_B.shape[0] == 2 or bool(pts_A[2,idx]>0.5)):
      visiable_points.append( True )
    else:
      visiable_points.append( False )

  visiable_points = torch.BoolTensor( visiable_points )

  if crop:
    assert isinstance(crop, list)
    x1, y1, x2, y2 = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])
    image = image.crop((x1, y1, x2, y2))
    pts_A[0, visiable_points] = pts_A[0, visiable_points] - x1
    pts_A[1, visiable_points] = pts_A[1, visiable_points] - y1
    pts_B[0, visiable_points] = pts_B[0, visiable_points] - x1
    pts_B[1, visiable_points] = pts_B[1, visiable_points] - y1

  image_A = image
  image_B = image_A.copy()

  draw_A = ImageDraw.Draw(image_A)
  draw_B = ImageDraw.Draw(image_B)
  #texts  = ['baseline', 'SRT']
  texts  = base_texts
  for idx in range(num_points):
    if visiable_points[ idx ]:
      # draw hollow circle
      for pts, draw, color in zip([pts_A,pts_B], [draw_A, draw_B], [color_A,color_B]):
        px = (pts[0,idx]-radius, pts[1,idx]-radius, pts[0,idx]+radius, pts[1,idx]+radius)
        px = [float(x) for x in px]
        if radius > 0: draw.ellipse(px, fill=color, outline=color)
  if isinstance(texts, (list, tuple)) and len(texts) == 2:
    fontScale = int(min(image_A.size)/10.0)
    font = ImageFont.truetype(get_font(), fontScale)
    draw_A.text((10, 10), texts[0], fill=color_A, font=font)
    draw_B.text((10, 10), texts[1], fill=color_B, font=font)
  return image_A, image_B


def draw_dualimage_by_points(image, pts_A, pts_B, radius, color_A, color_B, base_texts):
  if isinstance(image, str):  # In this case, image is an image path.
    image = pil_loader(image, False)
  assert isinstance(image, Image.Image), 'image type is not PIL.Image.Image'
  for pts in [pts_A, pts_B]:
    assert isinstance(pts, torch.Tensor) and (pts.shape[0] == 2 or pts.shape[0] == 3), 'input points are not correct : {:}'.format(pts)
  pts_A, pts_B = pts_A.clone(), pts_B.clone()

  num_points = pts_A.shape[1]
  visiable_points = []
  for idx in range(num_points):
    if (pts_A.shape[0] == 2 or bool(pts_A[2,idx]>0.5)) and \
       (pts_B.shape[0] == 2 or bool(pts_A[2,idx]>0.5)):
      visiable_points.append( True )
    else:
      visiable_points.append( False )

  visiable_points = torch.BoolTensor( visiable_points )

  finegrain = True
  if finegrain:
    owidth, oheight = image.size
    image = image.resize((owidth*8,oheight*8), Image.BICUBIC)
    pts_A[:2, visiable_points] = pts_A[:2, visiable_points] * 8.0
    pts_B[:2, visiable_points] = pts_B[:2, visiable_points] * 8.0
    radius = radius * 8

  image_A = image
  image_B = image_A.copy()

  draw_A = ImageDraw.Draw(image_A)
  draw_B = ImageDraw.Draw(image_B)
  #texts  = ['baseline', 'SRT']
  texts  = base_texts
  for idx in range(num_points):
    if visiable_points[ idx ]:
      # draw hollow circle
      for pts, draw, color in zip([pts_A,pts_B], [draw_A, draw_B], [color_A,color_B]):
        px = (pts[0,idx]-radius, pts[1,idx]-radius, pts[0,idx]+radius, pts[1,idx]+radius)
        px = [float(x) for x in px]
        if radius > 0: draw.ellipse(px, fill=color, outline=color)
  fontScale = int(min(image_A.size)/10.0)
  font = ImageFont.truetype(get_font(), fontScale)
  if texts is not None and isinstance(texts, (list,tuple)) and len(texts) == 2:
    draw_A.text((10, 10), texts[0], fill=color_A, font=font)
    draw_B.text((10, 10), texts[1], fill=color_B, font=font)
  
  if finegrain:
    image_A = image_A.resize((owidth,oheight), Image.BICUBIC)
    image_B = image_B.resize((owidth,oheight), Image.BICUBIC)
  return image_A, image_B


def draw_image_by_points_minor(_image, pts_A, pts_B, radius, color_A, color_B, resz):
  image = pil_loader(_image, False)
  assert isinstance(image, Image.Image), 'image type is not PIL.Image.Image'
  for pts in [pts_A, pts_B]:
    assert isinstance(pts, np.ndarray) and (pts.shape[0] == 2 or pts.shape[0] == 3), 'input points are not correct : {:}'.format(pts)
  pts_A, pts_B = pts_A.copy(), pts_B.copy()

  num_points = pts_A.shape[1]
  visiable_points, ctr_xs, ctr_ys = [], [], []
  for idx in range(num_points):
    if (pts_A.shape[0] == 2 or bool(pts_A[2,idx]>0.5)) and \
       (pts_B.shape[0] == 2 or bool(pts_A[2,idx]>0.5)):
      visiable_points.append( True )
      ctr_xs.append(pts_B[0,idx])
      ctr_ys.append(pts_B[0,idx])
    else:
      visiable_points.append( False )

  visiable_points = torch.BoolTensor( visiable_points )

  ctr_x, ctr_y = float(np.mean(ctr_xs)), float(np.mean(ctr_ys))
  H, W = pts_B[1,:].max() - pts_B[1,:].min(), pts_B[0,:].max() - pts_B[0,:].min()
  H, W = float(H) * 1.1, float(W) * 1.1
  x1, y1, x2, y2 = int(ctr_x-W/2), int(ctr_y-H/2), int(ctr_x+W/2), int(ctr_y+H/2)
  image = image.crop((x1, y1, x2, y2))
  pts_A[0, visiable_points] = pts_A[0, visiable_points] - x1
  pts_A[1, visiable_points] = pts_A[1, visiable_points] - y1
  pts_B[0, visiable_points] = pts_B[0, visiable_points] - x1
  pts_B[1, visiable_points] = pts_B[1, visiable_points] - y1

  width, height = image.size
  image = image.resize((resz,resz), Image.BICUBIC)
  pts_A[0, visiable_points] = pts_A[0, visiable_points] * 1.0 / width * resz
  pts_A[1, visiable_points] = pts_A[1, visiable_points] * 1.0 / height * resz
  pts_B[0, visiable_points] = pts_B[0, visiable_points] * 1.0 / width * resz
  pts_B[1, visiable_points] = pts_B[1, visiable_points] * 1.0 / height * resz


  draw = ImageDraw.Draw(image)
  for idx in range(num_points):
    if visiable_points[ idx ]:
      # draw hollow circle
      for pts, draw, color in zip([pts_A,pts_B], [draw, draw], [color_A,color_B]):
        px = (pts[0,idx]-radius, pts[1,idx]-radius, pts[0,idx]+radius, pts[1,idx]+radius)
        px = [float(x) for x in px]
        if radius > 0: draw.ellipse(px, fill=color, outline=color)
  return image
