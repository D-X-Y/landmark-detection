# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from os import path as osp
from copy import deepcopy as copy
from tqdm import tqdm
import warnings, time, math, random, numpy as np

from pts_utils import generate_label_map
from xvision import denormalize_points
from xvision import identity2affine, solve2theta, affine2image
from .dataset_utils import pil_loader
from .point_meta_v2 import PointMeta2V
from .point_meta_v2 import apply_affine2point
from .point_meta_v2 import apply_boundary
import torch
import torch.utils.data as data


class RobustDataset(data.Dataset):

  def __init__(self, transform, sigma, downsample, heatmap_type, \
                      shape, use_gray, data_indicator):

    self.transform    = transform
    self.sigma        = sigma
    self.downsample   = downsample
    self.heatmap_type = heatmap_type
    self.dataset_name = data_indicator
    self.shape        = shape # [H,W]
    self.use_gray     = use_gray
    assert transform is not None, 'transform : {:}'.format(transform)
    self.reset()
    print ('The general dataset initialization done : {:}'.format(self))


  def __repr__(self):
    return ('{name}(point-num={NUM_PTS}, shape={shape}, sigma={sigma}, use_gray={use_gray}, heatmap_type={heatmap_type}, length={length}, dataset={dataset_name})'.format(name=self.__class__.__name__, **self.__dict__))


  def reset(self, num_pts=-1, boxid='default', only_pts=False):
    self.NUM_PTS = num_pts
    if only_pts: return
    self.length  = 0
    self.datas   = []
    self.labels  = []
    self.BOXID = boxid
    self.norm_distances = []


  def __len__(self):
    assert len(self.datas) == self.length, 'The length is not correct : {}'.format(self.length)
    return self.length


  def append(self, data, label):
    assert osp.isfile(data), 'The image path is not a file : {:}'.format(data)
    self.datas.append( data )
    self.labels.append( label )
    self.length = self.length + 1

  
  def get_normalization_distance(self, index, init=False):
    if init:
      print ('initialize norm_distances')
      self.norm_distances = []
      for idx in tqdm( range(self.length) ):
        target = self.labels[idx].copy()
        if target.get_box() is None:
          image = pil_loader(self.datas[idx], self.use_gray)
          W, H  = image.size
          box = [0, 0, W, H]
        else:
          box = target.get_box().tolist()
        #if idx > 100: break
        diagonal = math.sqrt( (box[3]-box[1]) * (box[2]-box[0]) )
        self.norm_distances.append( diagonal )
    else:
      assert index >= 0 and index < self.length, 'Invalid index : {:}'.format(index)
      assert len(self.norm_distances) == self.length, '{:} vs. {:}'.format(self.norm_distances, self.length)
      return self.norm_distances[ index ]


  def load_list(self, file_lists, num_pts, boxindicator, reset):
    if reset: self.reset(num_pts, boxindicator)
    else    : assert self.NUM_PTS == num_pts and self.BOXID == boxindicator, 'The number of point is inconsistance : {:} vs {:}'.format(self.NUM_PTS, num_pts)
    if isinstance(file_lists, str): file_lists = [file_lists]
    samples = []
    for idx, file_path in enumerate(file_lists):
      print (':::: load list {:}/{:} : {:}'.format(idx, len(file_lists), file_path))
      xdata = torch.load(file_path)
      if isinstance(xdata, list)  : data = xdata          # image or video dataset list
      elif isinstance(xdata, dict): data = xdata['datas'] # multi-view dataset list
      else: raise ValueError('Invalid Type Error : {:}'.format( type(xdata) ))
      samples = samples + data
    # samples is a dict, where the key is the image-path and the value is the annotation
    # each annotation is a dict, contains 'points' (3,num_pts), and various box
    print ('GeneralDataset-V2 : {:} samples'.format(len(samples)))

    #for index, annotation in enumerate(samples):
    for index in tqdm( range( len(samples) ) ):
      annotation = samples[index]
      image_path  = annotation['current_frame']
      points, box = annotation['points'], annotation['box-{:}'.format(boxindicator)]
      label = PointMeta2V(self.NUM_PTS, points, box, image_path, self.dataset_name)
      self.append(image_path, label)

    assert len(self.datas)         == self.length, 'The length and the datas  is not right {:} vs {:}.'.format(self.length, len(self.datas))
    assert len(self.labels)        == self.length, 'The length and the labels is not right {:} vs {:}.'.format(self.length, len(self.labels))
    print ('Load data done for RobustDataset, which has {:} images.'.format(self.length))


  def __getitem__(self, index):
    assert index >= 0 and index < self.length, 'Invalid index : {:}'.format(index)
    image  = pil_loader(self.datas[index], self.use_gray)
    target = self.labels[index].copy()
    return self._process_(image, target, index)


  def _process_(self, image, target, index):

    # transform the image and points
    image, target, thetas = self.transform(image, target, index)
    (C, H, W), (height, width) = image.size(), self.shape

    # obtain the visiable indicator vector
    if target.is_none(): nopoints = True
    else               : nopoints = False

    assert isinstance(thetas, list) or isinstance(thetas, tuple), 'invalid thetas type : {:}'.format( type(thetas) )
    affineImage, heatmaps, masks, norm_points, THETA = [], [], [], [], []
    for _theta in thetas:
      _affineImage, _heatmap, _mask, _norm_points, _theta = self.__process_affine(image, target, _theta, nopoints)
      affineImage.append(_affineImage)
      heatmaps.append(_heatmap)
      masks.append(_mask)
      norm_points.append(_norm_points)
      THETA.append(_theta)
    affineImage, heatmaps, masks, norm_points, THETA = \
          torch.stack(affineImage), torch.stack(heatmaps), torch.stack(masks), torch.stack(norm_points), torch.stack(THETA)

    torch_index = torch.IntTensor([index])
    torch_nopoints = torch.ByteTensor( [ nopoints ] )
    torch_shape = torch.IntTensor([H,W])

    return affineImage, heatmaps, masks, norm_points, THETA, torch_index, torch_nopoints, torch_shape


  def __process_affine(self, image, target, theta, nopoints):
    image, target, theta = image.clone(), target.copy(), theta.clone()
    (C, H, W), (height, width) = image.size(), self.shape
    if nopoints: # do not have label
      norm_trans_points = torch.zeros((3, self.NUM_PTS))
      heatmaps          = torch.zeros((self.NUM_PTS+1, height//self.downsample, width//self.downsample))
      masks             = torch.ones((self.NUM_PTS+1, 1, 1), dtype=torch.uint8)
    else:
      norm_trans_points = apply_affine2point(target.get_points(), theta, (H,W))
      norm_trans_points = apply_boundary(norm_trans_points)
      real_trans_points = norm_trans_points.clone()
      real_trans_points[:2, :] = denormalize_points(self.shape, real_trans_points[:2,:])
      heatmaps, mask = generate_label_map(real_trans_points.numpy(), height//self.downsample, width//self.downsample, self.sigma, self.downsample, nopoints, self.heatmap_type) # H*W*C
      heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1))).type(torch.FloatTensor)
      masks    = torch.from_numpy(mask.transpose((2, 0, 1))).type(torch.ByteTensor)

    affineImage = affine2image(image, theta, self.shape)

    return affineImage, heatmaps, masks, norm_trans_points, theta
