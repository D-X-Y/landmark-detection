##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
from __future__ import print_function
from PIL import Image
import os
from os import path as osp
import numpy as np
import warnings
import math

from utils import load_list_from_folders, load_txt_file
from utils import generate_label_map_laplacian
from utils import generate_label_map_gaussian
from .dataset_utils import pil_loader
from .dataset_utils import anno_parser
from .point_meta import Point_Meta
import torch
import torch.utils.data as data

class GeneralDataset(data.Dataset):

  def __init__(self, transform, sigma, downsample, heatmap_type, dataset_name):

    self.transform = transform
    self.sigma = sigma
    self.downsample = downsample
    self.heatmap_type = heatmap_type
    self.dataset_name = dataset_name
    self.reset()
    print ('The general dataset initialization done, sigma is {}, downsample is {}, dataset-name : {}, self is : {}'.format(sigma, downsample, dataset_name, self))

  def __repr__(self):
    return ('{name}(number of point={NUM_PTS}, heatmap_type={heatmap_type})'.format(name=self.__class__.__name__, **self.__dict__))

  def convert68to51(self):
    # following 300-VW to remove the contour
    assert self.NUM_PTS == 68, 'Can only support the initial points is 68 vs {}'.format(self.NUM_PTS)
    print ('Start convert 68 points to 51 points for {} images'.format( len(self) ))
    for label in self.labels:
      label.convert68to51()
    self.NUM_PTS = 51

  def convert68to49(self):
    # following 300-VW to remove the contour
    assert self.NUM_PTS == 68, 'Can only support the initial points is 68 vs {}'.format(self.NUM_PTS)
    print ('Start convert 68 points to 49 points for {} images'.format( len(self) ))
    for label in self.labels:
      label.convert68to49()
    self.NUM_PTS = 49

  def reset(self, num_pts=-1):
    self.length = 0
    self.NUM_PTS = num_pts
    self.datas = []
    self.labels = []
    self.face_sizes = []
    assert self.dataset_name is not None, 'The dataset name is None'

  def append(self, data, label, box, face_size):
    assert osp.isfile(data), 'The image path is not a file : {}'.format(data)
    self.datas.append( data )
    if label is not None:
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

  def load_data(self, datas, labels, boxes, face_sizes, num_pts, reset):
    # each data is a png file name
    # each label is a Point_Meta class or the general pts format file (anno_parser_v1)
    print ('Start load data for the general datas')
    assert isinstance(datas, list), 'The type of the datas is not correct : {}'.format( type(datas) )
    assert isinstance(labels, list) and len(datas) == len(labels), 'The type of the labels is not correct : {}'.format( type(labels) )
    assert isinstance(boxes, list) and len(datas) == len(boxes), 'The type of the boxes is not correct : {}'.format( type(boxes) )
    assert isinstance(face_sizes, list) and len(datas) == len(face_sizes), 'The type of the face_sizes is not correct : {}'.format( type(face_sizes) )
    if reset: self.reset(num_pts)
    else:     assert self.NUM_PTS == num_pts, 'The number of point is inconsistance : {} vs {}'.format(self.NUM_PTS, num_pts)

    for idx, data in enumerate(datas):
      assert isinstance(data, str), 'The type of data is not correct : {}'.format(data)
      assert osp.isfile(datas[idx]), '{} is not a file'.format(datas[idx])
      self.append(datas[idx], labels[idx], boxes[idx], face_sizes[idx])

    assert len(self.datas) == self.length, 'The length and the data is not right {} vs {}'.format(self.length, len(self.datas))
    assert len(self.labels) == self.length, 'The length and the labels is not right {} vs {}'.format(self.length, len(self.labels))
    assert len(self.face_sizes) == self.length, 'The length and the face_sizes is not right {} vs {}'.format(self.length, len(self.face_sizes))
    print ('Load data done for the general dataset, which has {} images.'.format(self.length))

  def load_list(self, file_paths, num_pts, reset):
    if file_paths is None:
      print ('Input the None list file, skip load data.')
      return
    else:
      print ('Load list from {}'.format(file_paths))
    if isinstance(file_paths, str):
      file_paths = [ file_paths ]

    datas, labels, boxes, face_sizes = [], [], [], []
    for file_idx, file_path in enumerate(file_paths):
      assert osp.isfile(file_path), 'The path : {} is not a file.'.format(file_path)
      listfile = open(file_path, 'r')
      listdata = listfile.read().splitlines()
      listfile.close()
      print ('Load [{:d}/{:d}]-th list : {:} with {:} images'.format(file_idx, len(file_paths), file_path, len(listdata)))
      for idx, data in enumerate(listdata):
        alls = data.split(' ')
        if '' in alls: alls.remove('')
        assert len(alls) == 6 or len(alls) == 7, 'The {:04d}-th line is wrong : {:}'.format(idx, data)
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

  def __len__(self):
    assert len(self.datas) == self.length, 'The length is not correct : {}'.format(self.length)
    return self.length

  def prepare_input(self, image, box):
    meta = Point_Meta(self.NUM_PTS, None, np.array(box), image, self.dataset_name)
    image = pil_loader( image )
    return self._process_(image, meta, -1), meta

  def __getitem__(self, index):
    image = pil_loader( self.datas[index] )
    xtarget = self.labels[index].copy()
    return self._process_(image, xtarget, index)

  def _process_(self, image, xtarget, index):

    # Get the label
    if xtarget.is_none():
      visiable = None
    else:
      visiable = xtarget.points[2, :].astype('bool')

    # transform the image and points
    if self.transform is not None:
      image, xtarget = self.transform(image, xtarget)

    # If for evaluation not load label, keeps the original data
    temp_save_wh = xtarget.temp_save_wh
    ori_size = torch.IntTensor( [temp_save_wh[1], temp_save_wh[0], temp_save_wh[2], temp_save_wh[3]] ) # H, W, Cropped_[x1,y1]
        
    if isinstance(image, Image.Image):
      height, width = image.size[1], image.size[0]
    elif isinstance(image, torch.FloatTensor):
      height, width = image.size(1),  image.size(2)
    else:
      raise Exception('Unknown type of image : {}'.format( type(image) ))

    if xtarget.is_none() == False:
      xtarget.apply_bound(width, height)
      points = xtarget.points.copy()
      points = torch.from_numpy(points.transpose((1,0))).type(torch.FloatTensor)
      Hpoint = xtarget.points.copy()
    else:
      points = torch.from_numpy(np.zeros((self.NUM_PTS,3))).type(torch.FloatTensor)
      Hpoint = self.NUM_PTS

    if self.heatmap_type == 'laplacian':
      target, mask = generate_label_map_laplacian(Hpoint, height//self.downsample, width//self.downsample, self.sigma, self.downsample, visiable) # H*W*C
    elif self.heatmap_type == 'gaussian':
      target, mask = generate_label_map_gaussian(Hpoint, height//self.downsample, width//self.downsample, self.sigma, self.downsample, visiable) # H*W*C
    else:
      raise Exception('Unknown type of image : {}'.format( type(image) ))
      

    target = torch.from_numpy(target.transpose((2, 0, 1))).type(torch.FloatTensor)
    mask   = torch.from_numpy(mask.transpose((2, 0, 1))).type(torch.ByteTensor)
  
    torch_index = torch.IntTensor([index])
    torch_indicate = torch.ByteTensor( [ xtarget.is_none() == False ] )

    return image, target, mask, points, torch_index, torch_indicate, ori_size
