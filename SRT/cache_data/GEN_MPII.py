# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# python GEN_MPII.py
###############################################################
import os, sys, math, copy, json, torch
import os.path as osp
import numpy as np
from pathlib import Path
from collections import OrderedDict, defaultdict
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from scipy.io import loadmat
from xvision import draw_points, normalize_L
import datasets

# Change this paths according to your directories
this_dir = osp.dirname(os.path.abspath(__file__))
SAVE_DIR = osp.join(this_dir, 'lists', 'MPII')
HOME_STR = 'DOME_HOME'
if HOME_STR not in os.environ: HOME_STR = 'HOME'
assert HOME_STR in os.environ, 'Doest not find the HOME dir : {}'.format(HOME_STR)
print ('This dir : {}, HOME : [{}] : {}'.format(this_dir, HOME_STR, os.environ[HOME_STR]))
if not osp.isdir(SAVE_DIR): os.makedirs(SAVE_DIR)
image_dir = osp.join(os.environ[HOME_STR], 'datasets', 'landmark-datasets', 'MPII', 'images')
mat_path  = osp.join(os.environ[HOME_STR], 'datasets', 'landmark-datasets', 'MPII', 'mpii_human_pose_v1_u12_2', 'mpii_human_pose_v1_u12_1.mat')
print ('The MPII image dir : {:}'.format(image_dir))
print ('The MPII annotation file : {:}'.format(mat_path))
assert osp.isdir(image_dir), 'The image dir : {:} does not exist'.format(image_dir)
assert osp.isfile(mat_path), 'The annotation file : {:} does not exist'.format(mat_path)


def get_bounding_box(center, scale):
  x1 = center[0] - scale * 136
  y1 = center[1] - scale * 144
  x2 = center[0] + scale * 136
  y2 = center[1] + scale * 144
  return (x1, y1, x2, y2)


def get_person(person, image_path):
  pts = np.zeros((3, 16), dtype='float32')
  for point in person['points']:
    idx = point['id'] - 1
    pts[0, idx] = point['x']
    pts[1, idx] = point['y']
    pts[2, idx] = 1
  box = get_bounding_box(person['center'], person['scale'])
  head = person['head']
  headsize = 0.6 * np.sqrt( (head[2]-head[0])**2 + (head[3]-head[1])**2 )
  data = {'points': pts,
          'box-default': box,
          'normalizeL-head': headsize,
          'current_frame': image_path,
          'previous_frame': None,
          'next_frame': None}
  return data


def check_in_image(data, aux_info):
  box = data['box-default']
  points = data['points']
  oks = []
  for idx in range(points.shape[1]):
    if int(points[2,idx]+0.5) == 1:
      ok = points[0,idx] >= box[0] and points[0,idx] <= box[2] and points[1,idx] >= box[1] and points[1,idx] <= box[3]
      if not ok: print('{:} has {:02}-th point is out of box ({:}) : {:}'.format(aux_info, idx, box, points[:,idx]))
      oks.append(ok)
  return len(oks) == sum(oks)


def save_to_file_trainval(save_dir, trains, valids):

  save_dir = Path(save_dir)
  if not save_dir.exists(): save_dir.mkdir(parents=True, exist_ok=True)

  ## Train
  mean_landmark = [[] for i in range(16)]
  TDatas, OKs = [], []
  for index, DX in enumerate(trains):
    image_path = osp.join(image_dir, DX['name'])
    assert osp.isfile(image_path), '{:} does not exist'.format(image_path)
    for person in DX['persons']:
      data = get_person(person, image_path)
      TDatas.append( data )
      ok = check_in_image(data, 'TRAIN-{:}'.format(index))
      OKs.append( ok )
      # calculate means
      box, landmarks = data['box-default'], data['points']
      for idx in range(landmarks.shape[1]):
        if int(landmarks[2, idx] + 0.5) == 0: continue
        x, y = float(landmarks[0,idx]-box[0]), float(landmarks[1,idx]-box[1])
        x, y = normalize_L(x, box[2]-box[0]), normalize_L(y, box[3]-box[1])
        mean_landmark[idx].append( (x,y) )
  torch.save(TDatas, save_dir / 'train.pth')
  print ('Training has {:} persons with {:} % having out-of-box person.'.format(len(TDatas), 100 - np.array(OKs).mean() * 100))

  # Validation
  VDatas, OKs = [], []
  for index, DX in enumerate(valids):
    image_path = osp.join(image_dir, DX['name'])
    assert osp.isfile(image_path), '{:} does not exist'.format(image_path)
    for person in DX['persons']:
      data = get_person(person, image_path)
      VDatas.append( data )
      ok = check_in_image(data, 'VALID-{:}'.format(index))
      OKs.append( ok )
      # calculate means
      box, landmarks = data['box-default'], data['points']
      for idx in range(landmarks.shape[1]):
        if int(landmarks[2, idx] + 0.5) == 0: continue
        x, y = float(landmarks[0,idx]-box[0]), float(landmarks[1,idx]-box[1])
        x, y = normalize_L(x, box[2]-box[0]), normalize_L(y, box[3]-box[1])
        mean_landmark[idx].append( (x,y) )
  print ('Validation has {:} persons with {:} % having out-of-box person.'.format(len(VDatas), 100 - np.array(OKs).mean() * 100))

  torch.save(VDatas, save_dir / 'valid.pth')
  
  torch.save(TDatas + VDatas, save_dir / 'trainval.pth')

  mean_landmark = [np.array(x) for x in mean_landmark]
  mean_landmark = [np.mean(x, axis=0)  for x in mean_landmark]
  mean_landmark = np.array(mean_landmark)
  image = draw_points(mean_landmark, 600, 500, True)
  image.save(osp.join(save_dir, 'MPII-trainval.png'))
  torch.save({'default': mean_landmark}, osp.join(save_dir, 'MPII-trainval-mean.pth'))




def parse_anno_simple(anno, selects):
  image = anno['image']['name'][0,0][0]
  annorects = np.reshape(anno['annorect'], (anno['annorect'].size,))
  annorects = [annorects[i-1] for i in selects]
  persons = []
  # different persons
  for anno in annorects:
    # head pose
    #x1, y1, x2, y2 = float(anno['x1']), float(anno['y1']), float(anno['x2']), float(anno['y2'])
    center_x, center_y, scale = float(anno['objpos'][0,0]['x']), float(anno['objpos'][0,0]['y']), float(anno['scale'])
    person = {
              'center': [center_x, center_y],
              'scale' : scale}
    persons.append( person )
  return {'name': image,
          'persons': persons}


def parse_anno(anno, selects):
  image = anno['image']['name'][0,0][0]
  vidx, frame_sec = anno['vididx'], anno['frame_sec']
  assert vidx.size == 1 and frame_sec.size == 1
  vidx = vidx[0,0]
  frame_sec = frame_sec[0,0]
  annorects = np.reshape(anno['annorect'], (anno['annorect'].size,))
  annorects = [annorects[i-1] for i in selects]
  persons = []
  # different persons
  for anno in annorects:
    # head pose
    x1, y1, x2, y2 = float(anno['x1']), float(anno['y1']), float(anno['x2']), float(anno['y2'])
    center_x, center_y, scale = float(anno['objpos'][0,0]['x']), float(anno['objpos'][0,0]['y']), float(anno['scale'])
    if anno['annopoints'].size == 0:
      _points = []
    else:
      _points = np.squeeze(anno['annopoints']['point'][0,0], axis=0)
    points = []
    for x in _points:
      data = {'x': float(x['x']),
              'y': float(x['y']),
              'id': int(x['id'])}
      if 'is_visible' not in x or x['is_visible'].size == 0: # visible
        is_visible = True
      elif x['is_visible'].size == 1: #
        is_visible = bool( int(x['is_visible']) )
      else:
        raise ValueError('invalid visible: {:}'.format(x['is_visible']))
      data['is_visible'] = is_visible
      points.append( data )
    person = {'head'  : [x1, y1, x2, y2],
              'center': [center_x, center_y],
              'scale' : scale,
              'points': points}
    persons.append( person )
  return {'name': image,
          'vidx': vidx,
          'frame_sec': frame_sec,
          'persons': persons}


def load_splits(split_dir):
  assert osp.isdir(split_dir), '{:} is not a dir'.format(split_dir)
  cfile = open(osp.join(split_dir, 'train.txt'), 'r')
  Ltrain = cfile.readlines()
  Ltrain = [x.strip() for x in Ltrain]
  cfile.close()
  # validation
  cfile = open(osp.join(split_dir, 'valid.txt'), 'r')
  Lvalid = cfile.readlines()
  Lvalid = [x.strip() for x in Lvalid]
  cfile.close()
  # validation
  cfile = open(osp.join(split_dir, 'test.txt'), 'r')
  Ltest  = cfile.readlines()
  Ltest  = [x.strip() for x in Ltest ]
  cfile.close()
  return Ltrain, Lvalid, Ltest  


if __name__ == "__main__":
  mat = loadmat(mat_path)
  matdata = mat['RELEASE']
  print ('{:}'.format( matdata.dtype ))
  annolist      = np.squeeze( matdata['annolist'][0,0] )
  img_train     = np.squeeze( matdata['img_train'][0,0] )
  single_person = np.squeeze( matdata['single_person'][0,0] )
  act           = np.squeeze( matdata['act'][0,0] )
  video_list    = np.squeeze( matdata['video_list'][0,0] )
  video_list    = [x[0] for x in video_list]

  ACT           = []
  for xx in act:
    if xx['act_name'].shape == (1,):
      act_name = xx['act_name'][0]
    elif xx['act_name'].shape == (0,):
      act_name = None
    else: raise ValueError('{:}'.format( xx['act_name'] ))
    if xx['cat_name'].shape == (1,):
      cat_name = xx['cat_name'][0]
    elif xx['cat_name'].shape == (0,):
      cat_name = None
    else: raise ValueError('{:}'.format( xx['cat_name'] ))
    x = {'act_name': act_name,
         'cat_name': cat_name,
         'act_id'  : xx['act_id'][0,0]}
    ACT.append( x )

  Ltrain, Lvalid, Ltest = load_splits( osp.join(this_dir, 'cache', 'MPII-Split') )
  # get training
  trains, valids, tests, corrupts = [], [], [], []
  for idx, is_train in enumerate(img_train):
    image = annolist[idx]['image']['name'][0,0][0]
    print ('handle {:5d}/{:}-th data : {:}'.format(idx+1, len(img_train), image))
    if is_train:
      if single_person[idx].size == 0: continue
      select = np.reshape(single_person[idx], (single_person[idx].size,))
      if image in Ltrain or image in Lvalid:
        #data = {'anno': parse_anno( annolist[idx], select ),
        #        'act' : ACT[idx]
        #       }
        data = parse_anno( annolist[idx], select )
      else:
        corrupts.append( image )
        continue
      if image in Ltrain  : trains.append( data )
      elif image in Lvalid: valids.append( data )
      else: raise ValueError('Invalid Data : {:}'.format( image ))
    else:
      #assert image in Ltest, '{:} has something wrong'.format( image )
      select = np.reshape(single_person[idx], (single_person[idx].size,))
      data = parse_anno_simple( annolist[idx], select )
  print ('save data into {:}'.format(SAVE_DIR))
  save_to_file_trainval(SAVE_DIR, trains, valids)
  #save_to_file_test    (SAVE_DIR, tests)
