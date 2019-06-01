import os, time
import numpy as np
from utils.time_utils import print_log
import torch
import json
from collections import OrderedDict
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from .common_eval import evaluate_normalized_mean_error

class Eval_Meta():

  def __init__(self):
    self.reset()

  def __repr__(self):
    return ('{name}'.format(name=self.__class__.__name__)+'(number of data = {})'.format(len(self)))

  def reset(self):
    self.predictions = []
    self.groundtruth = []
    self.image_lists = []
    self.face_sizes  = []
    self.mae_bars = [1, 2, 4, 8, 16, 32, 80]

  def __len__(self):
    return len(self.image_lists)

  def append(self, _pred, _ground, image_path, face_size):
    assert _pred.shape[0] == 3 and len(_pred.shape) == 2, 'Prediction\'s shape is {} vs [should be (3,pts) or (2,pts)'.format(_pred.shape)
    assert _pred.shape == _ground.shape, 'shapes must be the same : {} vs {}'.format(_pred.shape, _ground.shape)
    if (not self.predictions) == False:
      assert _pred.shape == self.predictions[-1].shape, 'shapes must be the same : {} vs {}'.format(_pred.shape, _ground.shape)
    self.predictions.append(_pred)
    self.groundtruth.append(_ground)
    self.image_lists.append(image_path)
    self.face_sizes.append(face_size)

  def save(self, filename):
    meta = { 'predictions': self.predictions, 
             'groundtruth': self.groundtruth,
             'image_lists': self.image_lists,
             'face_sizes': self.face_sizes,
             'mae_bars':    self.mae_bars}
    torch.save(meta, filename)
    print ('save Eval_Meta into {}'.format(filename))

  def load(self, filename):
    assert os.path.isfile(filename), '{} is not a file'.format(filename)
    checkpoint       = torch.load(filename)
    self.predictions = checkpoint['predictions']
    self.groundtruth = checkpoint['groundtruth']
    self.image_lists = checkpoint['image_lists']
    self.face_sizes  = checkpoint['face_sizes']
    self.mae_bars    = checkpoint['mae_bars']

  def compute_mse(self, log, return_curve=False):
    nme, auc, for_pck_curve = evaluate_normalized_mean_error(self.predictions, self.groundtruth, log, self.face_sizes)
    if return_curve: return nme, auc, for_pck_curve
    else:            return nme, auc
