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
from .affine_utils import denormalize_points


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


# if normalize, then each point should be in [-1, 1] following PyTorch
def draw_image_by_points(_image, pts, radius, color, crop, resize, finegrain=False, normalize=False, draw_idx=False):
  if isinstance(_image, str):
    _image = pil_loader(_image, False)
  assert isinstance(_image, Image.Image), 'image type is not PIL.Image.Image'
  assert (isinstance(pts, np.ndarray) or isinstance(pts, torch.Tensor)) \
          and (pts.shape[0] == 2 or pts.shape[0] == 3), 'input points are not correct : {:}'.format(pts)
  image = _image.copy()
  if isinstance(pts, np.ndarray): pts = pts.copy()
  else                          : pts = pts.clone()

  if normalize:
    W, H = image.size
    pts[:2, ] = denormalize_points((H,W), pts[:2,])

  num_points = pts.shape[1]
  visiable_points = []
  for idx in range(num_points):
    if pts.shape[0] == 2 or bool(pts[2,idx]>0.5):
      visiable_points.append( True )
    else:
      visiable_points.append( False )

  if isinstance(pts, np.ndarray):
    visiable_points = np.array( visiable_points )
  else:
    visiable_points = torch.BoolTensor( visiable_points )

  if crop:
    if isinstance(crop, list):
      x1, y1, x2, y2 = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])
    else:
      x1, x2 = pts[0, visiable_points].min(), pts[0, visiable_points].max()
      y1, y2 = pts[1, visiable_points].min(), pts[1, visiable_points].max()
      face_h, face_w = (y2-y1)*0.1, (x2-x1)*0.1
      x1, x2 = int(x1 - face_w), int(x2 + face_w)
      y1, y2 = int(y1 - face_h), int(y2 + face_h)
    image = image.crop((x1, y1, x2, y2))
    pts[0, visiable_points] = pts[0, visiable_points] - x1
    pts[1, visiable_points] = pts[1, visiable_points] - y1

  if resize:
    if isinstance(resize, int): resize = (resize, resize)
    width, height = image.size
    image = image.resize((resize[0],resize[1]), Image.BICUBIC)
    pts[0, visiable_points] = pts[0, visiable_points] * 1.0 / width * resize[0]
    pts[1, visiable_points] = pts[1, visiable_points] * 1.0 / height * resize[1]

  #finegrain = True
  if finegrain:
    owidth, oheight = image.size
    image = image.resize((owidth*8,oheight*8), Image.BICUBIC)
    pts[0, visiable_points] = pts[0, visiable_points] * 8.0
    pts[1, visiable_points] = pts[1, visiable_points] * 8.0
    radius = radius * 8

  draw = ImageDraw.Draw(image)
  for idx in range(num_points):
    if visiable_points[ idx ]:
      # draw hollow circle
      px = (pts[0,idx]-radius, pts[1,idx]-radius, pts[0,idx]+radius, pts[1,idx]+radius)
      px = [float(x) for x in px]
      if radius > 0:
        draw.ellipse(px, fill=color, outline=color)
      if draw_idx:
        fontScale, text_color = int(min(image.size)/10.0), (200,200,200) if image.mode == 'RGB' else (200,0)
        font = ImageFont.truetype(get_font(), fontScale)
        x, y = int(px[0]), int(px[1])
        draw.text((x,y), '{:}'.format(idx+1), fill=text_color, font=font)

  if finegrain:
    image = image.resize((owidth,oheight), Image.BICUBIC)
  return image


def draw_image_by_points_failure_case(_image, pts, radius, colors, crop, resize):
  if isinstance(_image, str):
    _image = pil_loader(_image, False)
  assert isinstance(_image, Image.Image), 'image type is not PIL.Image.Image'
  assert (isinstance(pts, np.ndarray) or isinstance(pts, torch.Tensor)) \
          and (pts.shape[0] == 2 or pts.shape[0] == 3), 'input points are not correct : {:}'.format(pts)
  image = _image.copy()
  if isinstance(pts, np.ndarray): pts = pts.copy()
  else                          : pts = pts.clone()

  num_points = pts.shape[1]
  assert len(colors) == num_points, 'invalid length : {:} vs {:}'.format(len(colors), num_points)

  if crop:
    if isinstance(crop, list):
      x1, y1, x2, y2 = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])
    else:
      x1, x2 = pts[0].min(), pts[0].max()
      y1, y2 = pts[1].min(), pts[1].max()
      face_h, face_w = (y2-y1)*0.1, (x2-x1)*0.1
      x1, x2 = int(x1 - face_w), int(x2 + face_w)
      y1, y2 = int(y1 - face_h), int(y2 + face_h)
    image = image.crop((x1, y1, x2, y2))
    pts[0] = pts[0] - x1
    pts[1] = pts[1] - y1

  if resize:
    if isinstance(resize, int): resize = (resize, resize)
    width, height = image.size
    image = image.resize((resize[0],resize[1]), Image.BICUBIC)
    pts[0] = pts[0] * 1.0 / width * resize[0]
    pts[1] = pts[1] * 1.0 / height * resize[1]

  finegrain = True
  if finegrain:
    owidth, oheight = image.size
    exp_ratio = 6
    image = image.resize((owidth*exp_ratio,oheight*exp_ratio), Image.BICUBIC)
    pts[0] = pts[0] * 1.0 * exp_ratio
    pts[1] = pts[1] * 1.0 * exp_ratio
    radius = radius * 1.0 * exp_ratio

  draw = ImageDraw.Draw(image)
  for idx in range(num_points):
    # draw hollow circle
    px = (pts[0,idx]-radius, pts[1,idx]-radius, pts[0,idx]+radius, pts[1,idx]+radius)
    px = [float(x) for x in px]
    if radius > 0:
       draw.ellipse(px, fill=colors[idx], outline=colors[idx])
  if finegrain:
    image = image.resize((owidth,oheight), Image.BICUBIC)

  return image

# points is normalized into [-1, 1]
def draw_points(points, H, W, draw_idx=False):
  points = points.copy()
  points[:,0] = points[:,0] * 0.5 + 0.5
  points[:,1] = points[:,1] * 0.5 + 0.5
  shape = points.shape
  assert len(shape) == 2 and shape[1] == 2
  num_pts = shape[0]
  distance = np.fromfunction( lambda y, x, pid : (x*1.0/W-points[pid,0])**2 + (y*1.0/H-points[pid,1])**2, (H,W,num_pts), dtype=int)
  distance = np.sqrt( np.min(distance, axis=2) )
  threshold = 0.005
  image = np.zeros((H, W), dtype=np.uint8) + 255
  image[ distance < threshold ] = 0
  image = Image.fromarray( image )
  if draw_idx:
    fontScale, text_color = int(min(H,W)/50), (200,)
    font = ImageFont.truetype(get_font(), fontScale)
    draw  = ImageDraw.Draw(image)
    real_points = points.copy()
    real_points[:, 0] = real_points[:, 0] * W
    real_points[:, 1] = real_points[:, 1] * H
    for idx in range(real_points.shape[0]):
      x = int(real_points[idx,0] + W*threshold)
      y = int(real_points[idx,1] + H*threshold)
      draw.text((x,y), '{:}'.format(idx+1), fill=text_color, font=font)
  return image


def get_image_from_affine(I, theta, shape):
  # I = [C, H, W] ; shape = [H, W]
  assert len(I.size()) == 3, 'invalid size : {:}'.format(I.size())
  assert isinstance(shape, tuple) or isinstance(shape, list), 'invalid shape : {:}'.format(shape)
  theta = theta[:2, :].unsqueeze(0)
  grid_size = torch.Size([1, I.size(0), shape[0], shape[1]])
  grid  = F.affine_grid(theta, grid_size, align_corners=True)
  image = F.grid_sample(I.unsqueeze(0), grid, padding_mode='border', align_corners=True)
  image = image.numpy().squeeze(0)
  image = image.transpose(1, 2, 0)
  if image.shape[2] == 1:
    image = image.squeeze(2)
    image = Image.fromarray(np.uint8(image*255), mode='L')
  else:
    image = Image.fromarray(np.uint8(image*255))
  return image
