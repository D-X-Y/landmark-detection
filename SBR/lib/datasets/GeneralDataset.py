# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import print_function
from PIL import Image
from os import path as osp
import numpy as np
import math

from pts_utils import generate_label_map
from .file_utils import load_file_lists
from .dataset_utils import pil_loader
from .dataset_utils import anno_parser
from .point_meta import Point_Meta
import torch
import torch.utils.data as data

class GeneralDataset(data.Dataset):

  def __init__(self, transform, sigma, downsample, heatmap_type, data_indicator):

    self.transform = transform
    self.sigma = sigma
    self.downsample = downsample
    self.heatmap_type = heatmap_type
    self.dataset_name = data_indicator

    self.reset()
    print ('The general dataset initialization done : {:}'.format(self))

  def __repr__(self):
    return ('{name}(point-num={NUM_PTS}, sigma={sigma}, heatmap_type={heatmap_type}, length={length}, dataset={dataset_name})'.format(name=self.__class__.__name__, **self.__dict__))

  def reset(self, num_pts=-1):
    self.length = 0
    self.NUM_PTS = num_pts
    self.datas = []
    self.labels = []
    self.face_sizes = []
    assert self.dataset_name is not None, 'The dataset name is None'

  def __len__(self):
    assert len(self.datas) == self.length, 'The length is not correct : {}'.format(self.length)
    return self.length

  def append(self, data, label, box, face_size):
    assert osp.isfile(data), 'The image path is not a file : {}'.format(data)
    self.datas.append( data )
    if (label is not None) and (label.lower() != 'none'):
      if isinstance(label, str):
        assert osp.isfile(label), 'The annotation path is not a file : {}'.format(label)
        np_points, _ = anno_parser(label, self.NUM_PTS)
        meta = Point_Meta(self.NUM_PTS, np_points, box, data, self.dataset_name)
      elif isinstance(label, Point_Meta):
        meta = label.copy()
      else:
        raise NameError('Do not know this label : {}'.format(label))
    else:
      meta = Point_Meta(self.NUM_PTS, None, box, data, self.dataset_name)
    self.labels.append( meta )
    self.face_sizes.append( face_size )
    self.length = self.length + 1

  def prepare_input(self, image, box):
    meta = Point_Meta(self.NUM_PTS, None, np.array(box), image, self.dataset_name)
    image = pil_loader( image )
    return self._process_(image, meta, -1), meta

  def load_data(self, datas, labels, boxes, face_sizes, num_pts, reset):
    # each data is a png file name
    # each label is a Point_Meta class or the general pts format file (anno_parser_v1)
    assert isinstance(datas, list), 'The type of the datas is not correct : {}'.format( type(datas) )
    assert isinstance(labels, list) and len(datas) == len(labels), 'The type of the labels is not correct : {}'.format( type(labels) )
    assert isinstance(boxes, list) and len(datas) == len(boxes), 'The type of the boxes is not correct : {}'.format( type(boxes) )
    assert isinstance(face_sizes, list) and len(datas) == len(face_sizes), 'The type of the face_sizes is not correct : {}'.format( type(face_sizes) )
    if reset: self.reset(num_pts)
    else:     assert self.NUM_PTS == num_pts, 'The number of point is inconsistance : {} vs {}'.format(self.NUM_PTS, num_pts)

    print ('[GeneralDataset] load-data {:} datas begin'.format(len(datas)))

    for idx, data in enumerate(datas):
      assert isinstance(data, str), 'The type of data is not correct : {}'.format(data)
      assert osp.isfile(datas[idx]), '{} is not a file'.format(datas[idx])
      self.append(datas[idx], labels[idx], boxes[idx], face_sizes[idx])

    assert len(self.datas) == self.length, 'The length and the data is not right {} vs {}'.format(self.length, len(self.datas))
    assert len(self.labels) == self.length, 'The length and the labels is not right {} vs {}'.format(self.length, len(self.labels))
    assert len(self.face_sizes) == self.length, 'The length and the face_sizes is not right {} vs {}'.format(self.length, len(self.face_sizes))
    print ('Load data done for the general dataset, which has {} images.'.format(self.length))

  def load_list(self, file_lists, num_pts, reset):
    lists = load_file_lists(file_lists)
    print ('GeneralDataset : load-list : load {:} lines'.format(len(lists)))

    datas, labels, boxes, face_sizes = [], [], [], []

    for idx, data in enumerate(lists):
      alls = [x for x in data.split(' ') if x != '']
      
      assert len(alls) == 6 or len(alls) == 7, 'The {:04d}-th line in {:} is wrong : {:}'.format(idx, data)
      datas.append( alls[0] )
      if alls[1] == 'None':
        labels.append( None )
      else:
        labels.append( alls[1] )
      box = np.array( [ float(alls[2]), float(alls[3]), float(alls[4]), float(alls[5]) ] )
      boxes.append( box )
      if len(alls) == 6:
        face_sizes.append( None )
      else:
        face_sizes.append( float(alls[6]) )
    self.load_data(datas, labels, boxes, face_sizes, num_pts, reset)

  def __getitem__(self, index):
    assert index >= 0 and index < self.length, 'Invalid index : {:}'.format(index)
    image = pil_loader( self.datas[index] )
    target = self.labels[index].copy()
    return self._process_(image, target, index)

  def _process_(self, image, target, index):

    # transform the image and points
    if self.transform is not None:
      image, target = self.transform(image, target)

    # obtain the visiable indicator vector
    if target.is_none(): nopoints = True
    else               : nopoints = False

    # If for evaluation not load label, keeps the original data
    temp_save_wh = target.temp_save_wh
    ori_size = torch.IntTensor( [temp_save_wh[1], temp_save_wh[0], temp_save_wh[2], temp_save_wh[3]] ) # H, W, Cropped_[x1,y1]
        
    if isinstance(image, Image.Image):
      height, width = image.size[1], image.size[0]
    elif isinstance(image, torch.FloatTensor):
      height, width = image.size(1),  image.size(2)
    else:
      raise Exception('Unknown type of image : {}'.format( type(image) ))

    if target.is_none() == False:
      target.apply_bound(width, height)
      points = target.points.copy()
      points = torch.from_numpy(points.transpose((1,0))).type(torch.FloatTensor)
      Hpoint = target.points.copy()
    else:
      points = torch.from_numpy(np.zeros((self.NUM_PTS,3))).type(torch.FloatTensor)
      Hpoint = np.zeros((3, self.NUM_PTS))

    heatmaps, mask = generate_label_map(Hpoint, height//self.downsample, width//self.downsample, self.sigma, self.downsample, nopoints, self.heatmap_type) # H*W*C

    heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1))).type(torch.FloatTensor)
    mask     = torch.from_numpy(mask.transpose((2, 0, 1))).type(torch.ByteTensor)
  
    torch_index = torch.IntTensor([index])
    torch_nopoints = torch.ByteTensor( [ nopoints ] )

    return image, heatmaps, mask, points, torch_index, torch_nopoints, ori_size
