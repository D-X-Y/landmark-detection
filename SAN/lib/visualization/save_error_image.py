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

def save_error_image(image, points, locations, error_bar, save_path, radius = 10, color = (193,255,193), rev_color = (193,255,193), fontScale = 16, text_color = (255,255,255)):
  '''
  visualize image and plot keypoints on top of it
  parameter:
    image:          PIL.Image.Image object
    points:         (2 or 3) x num_pts numpy array : [x, y, visiable] Ground-Truth
    locations:      predictions
  '''
  if isinstance(image, str):
    image = datasets.pil_loader(image)
  assert isinstance(image, Image.Image), 'image type is not PIL.Image.Image'
  assert isinstance(points, np.ndarray) and points.shape[0] == 3, 'input points are not correct : {}'.format(points)
  assert isinstance(locations, np.ndarray) and locations.shape[0] == 3 and locations.shape == points.shape, 'input locations are not correct : {}'.format(locations)
  points, locations = points.copy(), locations.copy()
  distance  = np.zeros((points.shape[1]))
  for idx in range(points.shape[1]):
    if bool(points[2,idx]):
      dis = points[:2,idx] - locations[:2, idx]
      distance[idx] = np.sqrt(np.sum(dis*dis))
  if np.sum(distance > error_bar) == 0: return

  x_min, y_min = int(points[0,:].min())-30, int(points[1,:].min())-50
  x_max, y_max = int(points[0,:].max())+30, int(points[1,:].max())+10
  image = image.crop((x_min, y_min, x_max, y_max))
  locations[0,:] = locations[0,:] - x_min
  locations[1,:] = locations[1,:] - y_min
  points[0,:] = points[0,:] - x_min
  points[1,:] = points[1,:] - y_min
  locations = ( locations + points ) / 2
 
  save_path = save_path[:-4]

  zoom_in(image, locations, save_path+'.zoom.pdf', color)
  # save zoom-in image
  draw  = ImageDraw.Draw(image)
  w, h = image.size

  for idx in range(points.shape[1]):
    if distance[idx] < error_bar: continue
    #point = (int(points[0,idx])-radius, int(points[1,idx])-radius, int(points[0,idx])+radius, int(points[1,idx])+radius)
    #if point[0] > 0 and point[1] > 0 and point[2] < w and point[3] < h:
    #  draw.ellipse(point, fill=rev_color, outline=rev_color)
    # draw hollow circle
    point = (int(locations[0,idx])-radius, int(locations[1,idx])-radius, int(locations[0,idx])+radius, int(locations[1,idx])+radius)
    draw.ellipse(point, fill=color, outline=color)

  image.save(save_path+'.pdf')

def zoom_in(image, points, save_path, rev_color):
  image = image.copy()
  w, h = image.size
  scale = 3
  tw, th = w * scale, h * scale
  image = image.resize((tw,th), Image.BICUBIC)
  points = points * scale
  draw  = ImageDraw.Draw(image)
  radius = 1
  for idx in range(points.shape[1]):
    point = (int(points[0,idx])-radius, int(points[1,idx])-radius, int(points[0,idx])+radius, int(points[1,idx])+radius)
    if point[0] > 0 and point[1] > 0 and point[2] < tw and point[3] < th:
      draw.ellipse(point, fill=rev_color, outline=rev_color)
  image = image.crop((240, 120, 420, 250))
  image.save(save_path)
