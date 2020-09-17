# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from os import path as osp
from copy import deepcopy as copy
import cv2, time, random, numpy as np

import torch
import torch.utils.data as data
from .dataset_utils import pil_loader

from xvision import draw_image_by_points_major, merge_images
from xvision import draw_image_by_points


class WrapParallel(data.Dataset):

  def __init__(self, save_dir, all_image_ps, all_results, all_points, crop_size, color):

    self.save_dir = save_dir
    self.all_image_ps = all_image_ps
    self.all_results  = all_results
    self.all_points   = all_points
    self.centers      = []
    self.length       = len(all_points)
    for idx in range(self.length):
      gt_points = self.all_points[idx]
      x1, y1, x2, y2 = gt_points[0].min(), gt_points[1].min(), gt_points[0].max(), gt_points[1].max()
      ctr_x, ctr_y = (x1.item()+x2.item())/2, (y1.item()+y2.item())/2
      self.centers.append( (ctr_x, ctr_y) )
    self.crop_size = crop_size
    self.color     = color

  def __len__(self):
    return self.length

  def get_center(self, index):
    xs, ys, xrg = [], [], 15
    for i in range(index-xrg, index+xrg):
      if i>=0 and i<self.length:
        xs.append(self.centers[i][0])
        ys.append(self.centers[i][1])
    return float(np.mean(xs)), float(np.mean(ys))

  def __getitem__(self, index):
    assert index >= 0 and index < self.length, 'Invalid index : {:}'.format(index)
    image_path, points = self.all_image_ps[index], self.all_results[index]
    ctr_x, ctr_y = self.get_center(index)
    W, H = self.crop_size, self.crop_size
    image = draw_image_by_points(image_path, points, 2, self.color, [ctr_x-W, ctr_y-H, ctr_x+W, ctr_y+H], False)
    image.save(str(self.save_dir / image_path.split('/')[-1]))
    return index



class WrapParallelV2(data.Dataset):

  def __init__(self, save_dir, all_image_ps, pgts, old_results, new_results, all_points, crop_size, base_texts=['baseline', 'SRT']):

    self.save_dir = save_dir
    self.all_image_ps = all_image_ps
    self.pgts         = pgts
    self.old_results  = old_results
    self.new_results  = new_results
    self.all_points   = all_points
    self.centers      = []
    self.length       = len(all_points)
    for idx in range(self.length):
      gt_points = self.all_points[idx]
      x1, y1, x2, y2 = gt_points[0].min(), gt_points[1].min(), gt_points[0].max(), gt_points[1].max()
      ctr_x, ctr_y = (x1.item()+x2.item())/2, (y1.item()+y2.item())/2
      self.centers.append( (ctr_x, ctr_y) )
    self.centers   = np.array(self.centers)
    self.crop_size = crop_size
    self.base_texts = base_texts

  def __len__(self):
    return self.length

  def get_center(self, index):
    xs, ys, xrg = [], [], 60
    xstart = max(0, index-xrg)
    xend   = min(self.length, index+xrg)
    #for i in range(index-xrg, index+xrg):
    #  if i>=0 and i<self.length:
    #    xs.append(self.centers[i][0])
    #    ys.append(self.centers[i][1])
    #return float(np.mean(xs)), float(np.mean(ys))
    return float(self.centers[xstart:xend,0].mean()), float(self.centers[xstart:xend,1].mean())

  def __getitem__(self, index):
    assert index >= 0 and index < self.length, 'Invalid index : {:}'.format(index)
    image_path, points = self.all_image_ps[index], self.old_results[index]
    ctr_x, ctr_y = self.get_center(index)
    W, H  = self.crop_size, self.crop_size
    new_points = self.new_results[index]
    old_image, new_image = draw_image_by_points_major(image_path, points, new_points, 2, (255,0,0), (0,0,255), [ctr_x-W, ctr_y-H, ctr_x+W, ctr_y+H], self.base_texts)
    image = merge_images([old_image, new_image], 2, 'x')
    image.save(str(self.save_dir / image_path.split('/')[-1]))
    return index


class WrapParallelIMG(data.Dataset):

  def __init__(self, save_dir, all_image_ps, pgts, crop_size):

    self.save_dir = save_dir
    self.all_image_ps = all_image_ps
    self.pgts         = pgts
    self.centers      = []
    self.length       = len(pgts)
    for idx in range(self.length):
      gt_points = self.pgts[idx]
      x1, y1, x2, y2 = gt_points[0].min(), gt_points[1].min(), gt_points[0].max(), gt_points[1].max()
      ctr_x, ctr_y = (x1.item()+x2.item())/2, (y1.item()+y2.item())/2
      self.centers.append( (ctr_x, ctr_y) )
    self.centers   = np.array(self.centers)
    self.boxes     = [None for i in range(self.length)]
    self.images    = [None for i in range(self.length)]
    self.crop_size = crop_size

  def __len__(self):
    return self.length

  def get_center(self, index):
    xs, ys, xrg = [], [], 30
    xstart = max(0, index-xrg)
    xend   = min(self.length, index+xrg)
    return float(self.centers[xstart:xend,0].mean()), float(self.centers[xstart:xend,1].mean())

  def __getitem__(self, index):
    assert index >= 0 and index < self.length, 'Invalid index : {:}'.format(index)
    image_path = self.all_image_ps[index]
    if self.boxes[index] is None:
      ctr_x, ctr_y = self.get_center(index)
      if isinstance(self.crop_size, int):
        W, H = self.crop_size, self.crop_size
      elif isinstance(self.crop_size, (list,tuple)) and len(self.crop_size) == 2:
        W, H = self.crop_size
      else: raise ValueError('invalid crop - size : {:}'.format(self.crop_size))
      self.boxes[index] = (int(ctr_x-W), int(ctr_y-H), int(ctr_x+W), int(ctr_y+H))
    xbox = self.boxes[index]
    if self.images[index] is None:
      ximage = pil_loader(image_path, False)
      self.images[index] = ximage.crop(xbox)
    image = self.images[index]
    if self.save_dir is not None:
      image.save(str(self.save_dir / image_path.split('/')[-1]))
    return index, self.centers[index], np.array(xbox), np.array(image)
