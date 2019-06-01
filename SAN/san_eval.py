##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
from __future__ import division

import os, sys, time, random, argparse, PIL
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # please use Pillow 4.0.0 or it may fail for some images
from os import path as osp
import numbers, numpy as np
import init_path
import torch
import models
import datasets
from visualization import draw_image_by_points
from san_vision import transforms
from utils import time_string, time_for_file

def evaluate(args):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = True

  print ('The image is {:}'.format(args.image))
  print ('The model is {:}'.format(args.model))
  snapshot = Path(args.model)
  assert snapshot.exists(), 'The model path {:} does not exist'
  print ('The face bounding box is {:}'.format(args.face))
  assert len(args.face) == 4, 'Invalid face input : {:}'.format(args.face)
  snapshot = torch.load(snapshot)

  mean_fill   = tuple( [int(x*255) for x in [0.5, 0.5, 0.5] ] )
  normalize   = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])
  param = snapshot['args']
  eval_transform  = transforms.Compose([transforms.PreCrop(param.pre_crop_expand), transforms.TrainScale2WH((param.crop_width, param.crop_height)),  transforms.ToTensor(), normalize])

  net = models.__dict__[param.arch](param.modelconfig, None)

  net = net.cuda()
  weights = models.remove_module_dict(snapshot['state_dict'])
  net.load_state_dict(weights)

  dataset = datasets.GeneralDataset(eval_transform, param.sigma, param.downsample, param.heatmap_type, param.dataset_name)
  dataset.reset(param.num_pts)

  print ('[{:}] prepare the input data'.format(time_string()))
  [image, _, _, _, _, _, cropped_size], meta = dataset.prepare_input(args.image, args.face)
  inputs = image.unsqueeze(0).cuda()
  print ('[{:}] prepare the input data done'.format(time_string()))
  # network forward
  with torch.no_grad():
    batch_heatmaps, batch_locs, batch_scos, _ = net(inputs)
  print ('[{:}] the network forward done'.format(time_string()))

  # obtain the locations on the image in the orignial size
  cpu = torch.device('cpu')
  np_batch_locs, np_batch_scos, cropped_size = batch_locs.to(cpu).numpy(), batch_scos.to(cpu).numpy(), cropped_size.numpy()
  locations, scores = np_batch_locs[0,:-1,:], np.expand_dims(np_batch_scos[0,:-1], -1)

  scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2) , cropped_size[1] * 1. / inputs.size(-1)

  locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + cropped_size[3]
  prediction = np.concatenate((locations, scores), axis=1).transpose(1,0)
  for i in range(param.num_pts):
    point = prediction[:, i]
    print ('The coordinate of {:02d}/{:02d}-th points : ({:.1f}, {:.1f}), score = {:.3f}'.format(i, param.num_pts, float(point[0]), float(point[1]), float(point[2])))

  if args.save_path:
    image = draw_image_by_points(args.image, prediction, 1, (255,0,0), False, False)
    image.save( args.save_path )
    print ('save image with landmarks into {:}'.format(args.save_path))
  print('finish san evaluation on a single image : {:}'.format(args.image))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluate a single image by the trained model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--image',            type=str,   help='The evaluation image path.')
  parser.add_argument('--model',            type=str,   help='The snapshot to the saved detector.')
  parser.add_argument('--face',  nargs='+', type=float, help='The coordinate [x1,y1,x2,y2] of a face')
  parser.add_argument('--save_path',        type=str,   help='The path to save the visualization results')
  args = parser.parse_args()
  evaluate(args)
