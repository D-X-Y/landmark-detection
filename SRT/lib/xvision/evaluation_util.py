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
    self.normalizers  = []

  def __len__(self) -> int:
    return len(self.image_lists)

  def __getitem__(self, index):
    assert index>=0 and index<len(self.image_lists), 'invalid index : {:}'.format(index)
    return self.image_lists[index], self.predictions[index], self.groundtruth[index]

  def path(self, index: int) -> str:
    assert index>=0 and index<len(self.image_lists), 'invalid index : {:}'.format(index)
    return str(self.image_lists[index])

  def error(self, index: int) -> float:
    assert index>=0 and index<len(self.image_lists), 'invalid index : {:}'.format(index)
    preds, labels = self.predictions[index], self.groundtruth[index]
    seen = labels[2, :].astype(bool)
    if int(np.sum(seen)) == 0: return -1.0
    else:
      preds, labels = preds[:2, seen], labels[:2, seen]
      return float(np.linalg.norm(preds - labels, axis=0).mean())

  def append(self, _pred, _ground, image_path, face_size):
    assert _pred.shape[0] == 3 and len(_pred.shape) == 2, 'Prediction\'s shape is {:} vs [should be (3,pts) or (2,pts)]'.format(_pred.shape)
    if _ground is not None:
      assert _pred.shape == _ground.shape, 'shapes must be the same : {} vs {}'.format(_pred.shape, _ground.shape)
    if (not self.predictions) == False:
      assert _pred.shape == self.predictions[-1].shape, 'shapes must be the same : {} vs {}'.format(_pred.shape, self.predictions[-1].shape)
    self.predictions.append(_pred)
    self.groundtruth.append(_ground)
    self.image_lists.append(image_path)
    self.normalizers.append(face_size)

  def save(self, filename):
    meta = {'predictions': self.predictions, 
            'groundtruth': self.groundtruth,
            'image_lists': self.image_lists,
            'normalizers': self.normalizers}
    torch.save(meta, filename)
    print ('save eval-meta into {}'.format(filename))

  def load(self, filename, index=None):
    assert os.path.isfile(filename), '{:} is not a file'.format(filename)
    checkpoint       = torch.load(filename)
    if index == None: assert isinstance(checkpoint, dict), 'invalid type of checkpoint : {:}'.format(type(checkpoint))
    else            : checkpoint = checkpoint[index]
    self.predictions = checkpoint['predictions']
    self.groundtruth = checkpoint['groundtruth']
    self.image_lists = checkpoint['image_lists']
    self.normalizers = checkpoint['normalizers']

  def compute_mse(self, indicator, log, return_all_errors=False):
    predictions, groundtruth, normalizers, num = [], [], [], 0
    for x, gt, face in zip(self.predictions, self.groundtruth, self.normalizers):
      if gt is None: continue
      predictions.append(x)
      groundtruth.append(gt)
      normalizers.append(face)
      num += 1
    print_log('Filter the unlabeled data from {:} into {:} data'.format(len(self), num), log)
    if num == 0:
      nme, auc, pck_curves, _ = -1, None, None, None
    else:
      nme, auc, pck_curves, _ = evaluate_normalized_mean_error(self.predictions, self.groundtruth, self.normalizers, indicator, log)
    if return_all_errors: return _
    else : return nme, auc, pck_curves
