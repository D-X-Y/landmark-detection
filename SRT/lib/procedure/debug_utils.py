# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
import os, sys, PIL, torch, random, numpy as np
from PIL import Image
from pathlib import Path
from xvision import denormalize_points
from xvision import affine2image
from xvision import draw_image_by_points
from xvision import generate_color_from_heatmaps
from xvision import overlap_two_pil_image

def pil_loader(path):
  with open(path, 'rb') as f:
    with Image.open(f) as img:
      return img.convert('RGB')

def _save_heatmap(oriimage, heatmap, save_dir, name, ext, front):

  xheatmap = heatmap.cpu().numpy().transpose(1,2,0)
  backimage = generate_color_from_heatmaps(xheatmap, index=-1)
  bimage = PIL.Image.fromarray(np.uint8(backimage*255))
  bimage = overlap_two_pil_image(oriimage, bimage)
  bimage.save( str(save_dir / '{:}-{:}-back.{:}'.format(name, front, ext)) )

  randomidx = random.randint(0, heatmap.size(0)-1)
  rimage = generate_color_from_heatmaps(xheatmap, index=randomidx)
  rimage = PIL.Image.fromarray(np.uint8(rimage*255))
  rimage = overlap_two_pil_image(oriimage, rimage)
  rimage.save( str(save_dir / '{:}-{:}-P{:03d}.{:}'.format(name, front, randomidx, ext)) )


def pro_debug_save(save_dir, name, image, heatmap, normpoint, meantheta, predmap, recover):
  name, ext = name.split('.')
  save_dir.mkdir(parents=True, exist_ok=True)
  C, H, W = image.size()
  oriimage = recover(image)
  oriimage.save( str(save_dir / '{:}-ori.{:}'.format(name, ext)) )
  if C == 1: color = (255,)
  else     : color = (102,255,102)
  ptsimage = draw_image_by_points(oriimage, normpoint, 2, color, False, False, True)
  ptsimage.save( str(save_dir / '{:}-pts.{:}'.format(name, ext)) )
  meanI    = affine2image(image, meantheta, (H,W))
  meanimg  = recover(meanI)
  meanimg.save( str(save_dir / '{:}-tomean.{:}'.format(name, ext)) )
  
  _save_heatmap(oriimage, heatmap, save_dir, name, ext, 'GT')
  _save_heatmap(oriimage, predmap, save_dir, name, ext, 'PD')


def multiview_debug_save(save_dir, base, image_paths, points, rev_points):
  save_dir.mkdir(parents=True, exist_ok=True)
  images = [pil_loader(x) for x in image_paths]
  names  = [Path(x).name for x in image_paths]
  for index, (name, image) in enumerate( zip(names, images) ):
    _points = points[index].transpose(1,0)
    pil_img = draw_image_by_points(image, _points, 2, (102,255,102), False, False, False)
    pil_img.save( str(save_dir / '{:}-ori-x-{:}'.format(base, name)) )
    _points = rev_points[index].transpose(1,0)
    pil_img = draw_image_by_points(image, _points, 2, ( 30,144,255), False, False, False)
    pil_img.save( str(save_dir / '{:}-ori-p-{:}'.format(base, name)) )

def multiview_debug_save_v2(save_dir, base, names, images, points, rev_points):
  save_dir.mkdir(parents=True, exist_ok=True)
  for index, (name, image) in enumerate( zip(names, images) ):
    _points = points[index].transpose(1,0)
    pil_img = draw_image_by_points(image, _points, 2, (255,), False, False, False)
    pil_img.save( str(save_dir / '{:}-trans-x-{:}'.format(base, name)) )
    _points = rev_points[index].transpose(1,0)
    pil_img = draw_image_by_points(image, _points, 2, (0  ,), False, False, False)
    pil_img.save( str(save_dir / '{:}-trans-p-{:}'.format(base, name)) )
