##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import os, random, time
import pdb
import torch
from PIL import Image
import init_path
import numpy as np
from os import path as osp
import datasets
from san_vision import transforms

PRINT_GAP = 1000

def crop_style(list_file, num_pts, save_dir):
  #style = 'Original'
  #save_dir = 'cache/{}'.format(style)
  print ('crop face images into {}'.format(save_dir))
  if not osp.isdir(save_dir): os.makedirs(save_dir)
  transform  = transforms.Compose([transforms.PreCrop(0.2), transforms.TrainScale2WH((256, 256))])
  data = datasets.GeneralDataset(transform, 1, 8, 'gaussian', 'test')
  data.load_list(list_file, num_pts, True)
  #loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
  for i, tempx in enumerate(data):
    image = tempx[0]
    #points = tempx[3]
    basename = osp.basename(data.datas[i])
    save_name = osp.join(save_dir, basename)
    image.save(save_name)
    if i % PRINT_GAP == 0:
      print ('--->>> process the {:4d}/{:4d}-th image'.format(i, len(data)))


if __name__ == '__main__':

  this_dir = osp.dirname(osp.abspath(__file__))
  print ('The root dir is : {}'.format(this_dir))

  styles = ['Original', 'Gray', 'Light', 'Sketch']

  for style in styles:
    list_file = ['./cache_data/lists/300W/{:}/300w.train.GTB'.format(style),
                 './cache_data/lists/300W/{:}/300w.test.full.GTB'.format(style)]
    crop_style(list_file, 68, osp.join(this_dir, 'cache_data', 'cache', '300W', style))

  for style in styles:
    list_file = ['./cache_data/lists/AFLW/{:}/all.GTB'.format(style)]
    crop_style(list_file, 19, osp.join(this_dir, 'cache_data', 'cache', 'AFLW', style))

  #crop_style(['./snapshots/CLUSTER-300W_GTB-3/cluster-00-03.lst'], 68, osp.join('..', 'cache_data', 'cache', 'clusters', '300W-0'))
  #crop_style(['./snapshots/CLUSTER-300W_GTB-3/cluster-01-03.lst'], 68, osp.join('..', 'cache_data', 'cache', 'clusters', '300W-1'))
  #crop_style(['./snapshots/CLUSTER-300W_GTB-3/cluster-02-03.lst'], 68, osp.join('..', 'cache_data', 'cache', 'clusters', '300W-2'))
