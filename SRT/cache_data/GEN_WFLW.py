# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# usage:
# python GEN_WFLW.py
# dataset details in https://wywu.github.io/projects/LAB/WFLW.html
####################################################################
import os, sys, glob, torch, copy
from os import path as osp
from collections import OrderedDict, defaultdict
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
print ('lib-dir : {:}'.format(lib_dir))
import datasets
from utils.file_utils import load_list_from_folders, load_txt_file
from xvision import to_tensor
from xvision import draw_points, normalize_L


class FACE:
  def __init__(self, string):
    values = string.split(' ')
    assert len(values) == 196 + 4 + 6 + 1, 'invalid length : {:}'.format(len(values))
    coordinates = [float(x) for x in values[:196]]
    coordinates = np.array( coordinates )
    X, Y = coordinates[::2], coordinates[1::2]
    xbox = [float(x) for x in values[196:200]]
    [pose, expression, illumination, makeup, occlusion, blur] = [int(x) for x in values[200:206]]
    name = values[-1]
    self.landmarks = np.concatenate((np.expand_dims(X,axis=1), np.expand_dims(Y,axis=1), np.ones((98,1))), axis=1)
    self.xbox = xbox
    self.name = name


def return_box(landmarks, np_boxes, USE_BOX):
  if USE_BOX == 'GTL':
    box = datasets.dataset_utils.PTSconvert2box(landmarks.copy().T)
    box = (max(0, box[0]-1), max(0, box[1]-1), box[2]+1, box[3]+1)
  elif USE_BOX == 'GTB':
    box = (float(np_boxes[0]), float(np_boxes[1]), float(np_boxes[2]), float(np_boxes[3]))
  else:
    raise ValueError('The box indicator not find : {:}'.format(USE_BOX))
  return box


def Generate_WFLW_LIST(root, save_dir):
  assert osp.isdir(root), '{:} is not dir'.format(root)
  #assert osp.isdir(save_dir), '{} is not dir'.format(save_dir)
  if not osp.isdir(save_dir): os.makedirs(save_dir)
  image_dir = osp.join(root, 'WFLW_images')

  train_list = osp.join(root, 'WFLW_annotations', 'list_98pt_rect_attr_train_test', 'list_98pt_rect_attr_train.txt')
  with open(train_list) as f:
    content = f.readlines()
    train_faces = [FACE( x.strip() ) for x in content]
  assert len(train_faces) == 7500, 'invalid number of training faces : {:}'.format( len(train_faces) )
  
  SUFFIXS = ['GTB', 'GTL']
  mean_landmark = {SUFFIX : [[]for i in range(98)] for SUFFIX in SUFFIXS}

  trainData = [] # OrderedDict()
  for face in train_faces:
    landmarks = face.landmarks
    data = {'points': landmarks.T, 'name'  : face.name}
    for SUFFIX in SUFFIXS:
      box = return_box(landmarks, face.xbox, SUFFIX)
      data['box-{:}'.format(SUFFIX)] = box
      for idx in range(98):
        if int(landmarks[idx,2] + 0.5) == 0: continue
        x, y = float(landmarks[idx,0]-box[0]), float(landmarks[idx,1]-box[1])
        x, y = normalize_L(x, box[2]-box[0]), normalize_L(y, box[3]-box[1])
        mean_landmark[SUFFIX][idx].append( (x,y) )
    data['box-default']    = data['box-GTB']
    data['previous_frame'] = None
    data['current_frame']  = osp.join(image_dir, face.name)
    data['next_frame']     = None
    assert osp.isfile( data['current_frame'] )
    trainData.append( data )
  torch.save(trainData, osp.join(save_dir, 'train.pth'))
  for SUFFIX in SUFFIXS:
    allp = mean_landmark[SUFFIX]
    allp = np.array(allp)
    mean_landmark[SUFFIX] = np.mean(allp, axis=1)
    mean_landmark[SUFFIX] = mean_landmark[SUFFIX] * 0.9
    image = draw_points(mean_landmark[SUFFIX], 600, 500, True)
    image.save(osp.join(save_dir, 'train-{:}.png'.format(SUFFIX)))
  mean_landmark['default'] = mean_landmark['GTB']
  torch.save(mean_landmark, osp.join(save_dir, 'train-mean.pth'))
  print ('Training Set   : {:5d} facial images.'.format(len(trainData)))

  test_list = osp.join(root, 'WFLW_annotations', 'list_98pt_rect_attr_train_test', 'list_98pt_rect_attr_test.txt')
  with open(test_list) as f:
    content = f.readlines()
    test_faces = [FACE( x.strip() ) for x in content]
  assert len(test_faces) == 2500, 'invalid number of training faces : {:}'.format( len(train_faces) )
  testData = []
  for face in test_faces:
    landmarks = face.landmarks
    data = {'points': landmarks.T, 'name'  : face.name}
    for SUFFIX in SUFFIXS:
      box = return_box(landmarks, face.xbox, SUFFIX)
      data['box-{:}'.format(SUFFIX)] = box
    data['box-default']    = data['box-GTB']
    data['previous_frame'] = None
    data['current_frame']  = osp.join(image_dir, face.name)
    data['next_frame']     = None
    assert osp.isfile( data['current_frame'] )
    testData.append( data )
  torch.save(testData, osp.join(save_dir, 'test.pth'))
  print ('Test Set   : {:5d} facial images.'.format(len(testData)))
  print ('Save all dataset files into {:}'.format(save_dir))


if __name__ == '__main__':
  HOME_STR = 'DOME_HOME'
  if HOME_STR not in os.environ: HOME_STR = 'HOME'
  assert HOME_STR in os.environ, 'Doest not find the HOME dir : {}'.format(HOME_STR)
  this_dir = osp.dirname(os.path.abspath(__file__))
  SAVE_DIR = osp.join(this_dir, 'lists', 'WFLW')
  print ('This dir : {:}, HOME : [{:}] : {:}'.format(this_dir, HOME_STR, os.environ[HOME_STR]))
  root_path = osp.join( os.environ[HOME_STR], 'datasets', 'landmark-datasets', 'WFLW')

  Generate_WFLW_LIST(root_path, SAVE_DIR)
