# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import datasets

def draw_image_by_points(_image, pts, radius, color, crop, resize):
  if isinstance(_image, str):
    _image = datasets.pil_loader(_image)
  assert isinstance(_image, Image.Image), 'image type is not PIL.Image.Image'
  assert isinstance(pts, np.ndarray) and (pts.shape[0] == 2 or pts.shape[0] == 3), 'input points are not correct'
  image, pts = _image.copy(), pts.copy()

  num_points = pts.shape[1]
  visiable_points = []
  for idx in range(num_points):
    if pts.shape[0] == 2 or bool(pts[2,idx]):
      visiable_points.append( True )
    else:
      visiable_points.append( False )
  visiable_points = np.array( visiable_points )
  #print ('visiable points : {}'.format( np.sum(visiable_points) ))

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
    width, height = image.size
    image = image.resize((resize,resize), Image.BICUBIC)
    pts[0, visiable_points] = pts[0, visiable_points] * 1.0 / width * resize
    pts[1, visiable_points] = pts[1, visiable_points] * 1.0 / height * resize

  finegrain = True
  if finegrain:
    owidth, oheight = image.size
    image = image.resize((owidth*8,oheight*8), Image.BICUBIC)
    pts[0, visiable_points] = pts[0, visiable_points] * 8.0
    pts[1, visiable_points] = pts[1, visiable_points] * 8.0
    radius = radius * 8

  draw  = ImageDraw.Draw(image)
  for idx in range(num_points):
    if visiable_points[ idx ]:
      # draw hollow circle
      point = (pts[0,idx]-radius, pts[1,idx]-radius, pts[0,idx]+radius, pts[1,idx]+radius)
      if radius > 0:
        draw.ellipse(point, fill=color, outline=color)

  if finegrain:
    image = image.resize((owidth,oheight), Image.BICUBIC)

  return image
