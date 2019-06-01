# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, pdb, sys, glob
from os import path as osp
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
print ('lib-dir : {:}'.format(lib_dir))
import datasets
from scipy.io import loadmat
from utils.file_utils import load_list_from_folders, load_txt_file

def load_box(mat_path, cdir):
  mat = loadmat(mat_path)
  mat = mat['bounding_boxes']
  mat = mat[0]
  assert len(mat) > 0, 'The length of this mat file should be greater than 0 vs {}'.format(len(mat))
  all_object = []
  for cobject in mat:
    name = cobject[0][0][0][0]
    bb_detector = cobject[0][0][1][0]
    bb_ground_t = cobject[0][0][2][0]
    image_path = osp.join(cdir, name)
    image_path = image_path[:-4]
    all_object.append( (image_path, bb_detector, bb_ground_t) )
  return all_object

def load_mats(lists):
  all_objects = []
  for dataset in lists:
    cobjects = load_box(dataset[0], dataset[1])
    all_objects = all_objects + cobjects
  return all_objects

def return_box(image_path, pts_path, all_dict, USE_BOX):
  image_path = image_path[:-4]
  assert image_path in all_dict, '{} not find'.format(image_path)
  np_boxes = all_dict[ image_path ]
  if USE_BOX == 'GTL':
    box_str = datasets.dataset_utils.for_generate_box_str(pts_path, 68, 0)
  elif USE_BOX == 'GTB':
    box_str = '{:.4f} {:.4f} {:.4f} {:.4f}'.format(np_boxes[1][0], np_boxes[1][1], np_boxes[1][2], np_boxes[1][3])
  elif USE_BOX == 'DET':
    box_str = '{:.4f} {:.4f} {:.4f} {:.4f}'.format(np_boxes[0][0], np_boxes[0][1], np_boxes[0][2], np_boxes[0][3])
  else:
    assert False, 'The box indicator not find : {}'.format(USE_BOX)
  return box_str

def load_all_300w(root_dir):
  print ('300W Root Dir : {}'.format(root_dir))
  mat_dir = osp.join(root_dir, 'Bounding_Boxes')
  pairs = [(osp.join(mat_dir, 'bounding_boxes_lfpw_testset.mat'),   osp.join(root_dir, 'lfpw', 'testset')),
           (osp.join(mat_dir, 'bounding_boxes_lfpw_trainset.mat'),  osp.join(root_dir, 'lfpw', 'trainset')),
           (osp.join(mat_dir, 'bounding_boxes_ibug.mat'),           osp.join(root_dir, 'ibug')),
           (osp.join(mat_dir, 'bounding_boxes_afw.mat'),            osp.join(root_dir, 'afw')),
           (osp.join(mat_dir, 'bounding_boxes_helen_testset.mat'),  osp.join(root_dir, 'helen', 'testset')),
           (osp.join(mat_dir, 'bounding_boxes_helen_trainset.mat'), osp.join(root_dir, 'helen', 'trainset')),]

  all_datas = load_mats(pairs)
  data_dict = {}
  for i, cpair in enumerate(all_datas):
    image_path = cpair[0].replace(' ', '')
    data_dict[ image_path ] = (cpair[1], cpair[2])
  return data_dict

def generate_300w_list(root, save_dir, box_data, SUFFIX):
  assert osp.isdir(root), '{} is not dir'.format(root)
  #assert osp.isdir(save_dir), '{} is not dir'.format(save_dir)
  if not osp.isdir(save_dir): os.makedirs(save_dir)
  train_length, common_length, challenge_length = 3148, 554, 135
  subsets = ['afw', 'helen', 'ibug', 'lfpw']
  dir_lists = [osp.join(root, subset) for subset in subsets]
  imagelist, num_image = load_list_from_folders(dir_lists, ext_filter=['png', 'jpg', 'jpeg'], depth=3)

  train_set, common_set, challenge_set = [], [], []
  for image_path in imagelist:
    name, ext = osp.splitext(image_path)
    anno_path = name + '.pts'
    assert osp.isfile(anno_path), 'annotation for : {} does not exist'.format(image_path)
    if name.find('ibug') > 0:
      challenge_set.append( (image_path, anno_path) )
    elif name.find('afw') > 0:
      train_set.append( (image_path, anno_path) )
    elif name.find('helen') > 0 or name.find('lfpw') > 0:
      if name.find('trainset') > 0:
        train_set.append( (image_path, anno_path) )
      elif name.find('testset') > 0:
        common_set.append( (image_path, anno_path) )
      else:
        raise Exception('Unknow name : {}'.format(name))
    else:
      raise Exception('Unknow name : {}'.format(name))
  assert len(train_set) == train_length, 'The length is not right for train : {} vs {}'.format(len(train_set), train_length)
  assert len(common_set) == common_length, 'The length is not right for common : {} vs {}'.format(len(common_set), common_length)
  assert len(challenge_set) == challenge_length, 'The length is not right for challeng : {} vs {}'.format(len(common_set), common_length)

  with open(osp.join(save_dir, '300w.train.' + SUFFIX), 'w') as txtfile:
    for cpair in train_set:
      #box_str = datasets.dataset_utils.for_generate_box_str(cpair[1], 68, EXPAND_RATIO)
      box_str = return_box(cpair[0], cpair[1], box_data, SUFFIX)
      txtfile.write('{} {} {}\n'.format(cpair[0], cpair[1], box_str))
  txtfile.close()

  with open(osp.join(save_dir, '300w.test.common.' + SUFFIX), 'w') as txtfile:
    for cpair in common_set:
      #box_str = datasets.dataset_utils.for_generate_box_str(cpair[1], 68, EXPAND_RATIO)
      box_str = return_box(cpair[0], cpair[1], box_data, SUFFIX)
      txtfile.write('{} {} {}\n'.format(cpair[0], cpair[1], box_str))
  txtfile.close()

  with open(osp.join(save_dir, '300w.test.challenge.' + SUFFIX), 'w') as txtfile:
    for cpair in challenge_set:
      #box_str = datasets.dataset_utils.for_generate_box_str(cpair[1], 68, EXPAND_RATIO)
      box_str = return_box(cpair[0], cpair[1], box_data, SUFFIX)
      txtfile.write('{} {} {}\n'.format(cpair[0], cpair[1], box_str))
  txtfile.close()

  with open(osp.join(save_dir, '300w.test.full.' + SUFFIX), 'w') as txtfile:
    fullset = common_set + challenge_set
    for cpair in fullset:
      #box_str = datasets.dataset_utils.for_generate_box_str(cpair[1], 68, EXPAND_RATIO)
      box_str = return_box(cpair[0], cpair[1], box_data, SUFFIX)
      txtfile.write('{} {} {}\n'.format(cpair[0], cpair[1], box_str))
  txtfile.close()

if __name__ == '__main__':
  HOME_STR = 'DOME_HOME'
  if HOME_STR not in os.environ: HOME_STR = 'HOME'
  assert HOME_STR in os.environ, 'Doest not find the HOME dir : {}'.format(HOME_STR)
  this_dir = osp.dirname(os.path.abspath(__file__))
  SAVE_DIR = osp.join(this_dir, 'lists', '300W')
  print ('This dir : {}, HOME : [{}] : {}'.format(this_dir, HOME_STR, os.environ[HOME_STR]))
  path_300w = osp.join( os.environ[HOME_STR], 'datasets', 'landmark-datasets', '300W')
  USE_BOXES = ['GTB', 'DET']
  box_datas = load_all_300w(path_300w)

  for USE_BOX in USE_BOXES:
    generate_300w_list(path_300w, SAVE_DIR, box_datas, USE_BOX)
