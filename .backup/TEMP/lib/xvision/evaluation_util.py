# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, time
import numpy as np
import torch
import json
from log_utils import print_log
from collections import OrderedDict
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from .common_eval import evaluate_normalized_mean_error

class Eval_Meta():

  def __init__(self):
    self.reset()

  def __repr__(self):
    return ('{name}'.format(name=self.__class__.__name__)+'(number of data = {:})'.format(len(self)))

  def reset(self):
    self.predictions = []
    self.groundtruth = []
    self.image_lists = []
    self.face_sizes  = []

  def __len__(self):
    return len(self.image_lists)

  def append(self, _pred, _ground, image_path, face_size):
    assert _pred.shape[0] == 3 and len(_pred.shape) == 2, 'Prediction\'s shape is {:} vs [should be (3,pts) or (2,pts)]'.format(_pred.shape)
    if _ground is not None:
      assert _pred.shape == _ground.shape, 'shapes must be the same : {} vs {}'.format(_pred.shape, _ground.shape)
    if (not self.predictions) == False:
      assert _pred.shape == self.predictions[-1].shape, 'shapes must be the same : {} vs {}'.format(_pred.shape, self.predictions[-1].shape)
    self.predictions.append(_pred)
    self.groundtruth.append(_ground)
    self.image_lists.append(image_path)
    self.face_sizes.append(face_size)

  def save(self, filename):
    meta = {'predictions': self.predictions, 
            'groundtruth': self.groundtruth,
            'image_lists': self.image_lists,
            'face_sizes' : self.face_sizes}
    torch.save(meta, filename)
    print ('save eval-meta into {}'.format(filename))

  def load(self, filename):
    assert os.path.isfile(filename), '{} is not a file'.format(filename)
    checkpoint       = torch.load(filename)
    self.predictions = checkpoint['predictions']
    self.groundtruth = checkpoint['groundtruth']
    self.image_lists = checkpoint['image_lists']
    self.face_sizes  = checkpoint['face_sizes']

  def compute_mse(self, log):
    predictions, groundtruth, face_sizes, num = [], [], [], 0
    for x, gt, face in zip(self.predictions, self.groundtruth, self.face_sizes):
      if gt is None: continue
      predictions.append(x)
      groundtruth.append(gt)
      face_sizes.append(face)
      num += 1
    print_log('Filter the unlabeled data from {:} into {:} data'.format(len(self), num), log)
    if num == 0:
      nme, auc, pck_curves = -1, None, None
    else:
      nme, auc, pck_curves = evaluate_normalized_mean_error(self.predictions, self.groundtruth, log, self.face_sizes)
    return nme, auc, pck_curves
