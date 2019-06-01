##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Modified from PyTorch Cycle-GAN                        ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
from __future__ import print_function
from PIL import Image
import os, random
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

class CycleDataset(data.Dataset):

  def __init__(self, transform, dataset_name):

    self.transform = transform
    self.dataset_name = dataset_name
    self.reset()
    #print ('The general dataset initialization done, dataset-name : {}, self is : {}'.format(dataset_name, self))

  def __repr__(self):
    return ('{name}(dataset={dataset_name}, A.size={A_size}, B.size={B_size})'.format(name=self.__class__.__name__, **self.__dict__))

  def reset(self):
    self.A_size = 0
    self.A_datas = []
    self.A_labels = []
    self.B_size = 0
    self.B_datas = []
    self.B_labels = []
    assert self.dataset_name is not None, 'The dataset name is None'


  def __obtain(self, file_paths):
    datas, boxes = [], []
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
        box = np.array( [ float(alls[2]), float(alls[3]), float(alls[4]), float(alls[5]) ] )
        boxes.append( box )
    labels = []
    for idx, data in enumerate(datas):
      assert isinstance(data, str), 'The type of data is not correct : {}'.format(data)
      meta = Point_Meta(1, None, boxes[idx], data, self.dataset_name)
      labels.append( meta )
    return datas, labels

  def set_a(self, file_paths):
    self.A_datas, self.A_labels = self.__obtain(file_paths)
    self.A_size = len(self.A_datas)
    assert len(self.A_labels) == self.A_size and self.A_size > 0, 'The length is not right : {} vs {}'.format(len(self.A_datas), len(self.A_labels))
    print ('Set the A-dataset from {} lists and obtain {} faces'.format(len(file_paths), self.A_size))

  def set_b(self, file_paths):
    self.B_datas, self.B_labels = self.__obtain(file_paths)
    self.B_size = len(self.B_datas)
    assert len(self.B_labels) == self.B_size and self.B_size > 0, 'The length is not right : {} vs {}'.format(len(self.B_datas), len(self.B_labels))
    print ('Set the B-dataset from {} lists and obtain {} faces'.format(len(file_paths), self.B_size))

  def append_a(self, dataset, indexes):
    for index in indexes:
      self.A_datas.append( dataset.datas[index] )
      self.A_labels.append( dataset.labels[index].copy() )
    self.A_size = len(self.A_datas)

  def append_b(self, dataset, indexes):
    for index in indexes:
      self.B_datas.append( dataset.datas[index] )
      self.B_labels.append( dataset.labels[index].copy() )
    self.B_size = len(self.B_datas)

  def __len__(self):
    return max(self.A_size, self.B_size)

  def __getitem__(self, index):
    index_A = index % self.A_size
    index_B = random.randint(0, self.B_size - 1)

    A_img = pil_loader( self.A_datas[index_A] )
    B_img = pil_loader( self.B_datas[index_B] )
    A_target = self.A_labels[index_A].copy()
    B_target = self.B_labels[index_B].copy()

    # transform the image and points
    if self.transform is not None:
      A_image, A_target = self.transform(A_img, A_target)
      B_image, B_target = self.transform(B_img, B_target)

    return {'A': A_image, 'B': B_image, 'A_index': index_A, 'B_index': index_B}
