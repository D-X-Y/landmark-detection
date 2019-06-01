# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from os import path as osp
from PIL import Image
from scipy.ndimage.interpolation import zoom
from utils.file_utils import load_txt_file
import numpy as np
import copy, math, pdb

def pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    with Image.open(f) as img:
      return img.convert('RGB')

def remove_item_from_list(list_to_remove, item):
  '''
  remove a single item from a list
  '''
  assert isinstance(list_to_remove, list), 'input list is not a list'
    
  try:
    list_to_remove.remove(item)
  except ValueError:
    print('Warning!!!!!! Item to remove is not in the list. Remove operation is not done.')

  return list_to_remove

def anno_parser(anno_path, num_pts):  
  data, num_lines = load_txt_file(anno_path)                          
  if data[0].find('version: ') == 0: # 300-W
    return anno_parser_v0(anno_path, num_pts)
  else:
    return anno_parser_v1(anno_path, num_pts)

def anno_parser_v0(anno_path, num_pts):  
  '''                        
  parse the annotation for 300W dataset, which has a fixed format for .pts file                                
  return:                    
    pts: 3 x num_pts (x, y, oculusion)                                
  '''                        
  data, num_lines = load_txt_file(anno_path)                          
  assert data[0].find('version: ') == 0, 'version is not correct'     
  assert data[1].find('n_points: ') == 0, 'number of points in second line is not correct'                     
  assert data[2] == '{' and data[-1] == '}', 'starting and end symbol is not correct'                          
                             
  assert data[0] == 'version: 1' or data[0] == 'version: 1.0', 'The version is wrong : {}'.format(data[0])
  n_points = int(data[1][len('n_points: '):])                         
                             
  assert num_lines == n_points + 4, 'number of lines is not correct'    # 4 lines for general information: version, n_points, start and end symbol      
  assert num_pts == n_points, 'number of points is not correct'
                             
  # read points coordinate   
  pts = np.zeros((3, n_points), dtype='float32')                      
  line_offset = 3    # first point starts at fourth line              
  point_set = set()
  for point_index in range(n_points):                                
    try:                     
      pts_list = data[point_index + line_offset].split(' ')       # x y format                                 
      if len(pts_list) > 2:    # handle edge case where additional whitespace exists after point coordinates   
        pts_list = remove_item_from_list(pts_list, '')              
      pts[0, point_index] = float(pts_list[0])                        
      pts[1, point_index] = float(pts_list[1])                        
      pts[2, point_index] = float(1)      # oculusion flag, 0: oculuded, 1: visible. We use 1 for all points since no visibility is provided by 300-W   
      point_set.add( point_index )
    except ValueError:       
      print('error in loading points in %s' % anno_path)              
  return pts, point_set

def anno_parser_v1(anno_path, NUM_PTS, one_base=True):
  '''
  parse the annotation for MUGSY-Full-Face dataset, which has a fixed format for .pts file
  return: pts: 3 x num_pts (x, y, oculusion)
  '''
  data, n_points = load_txt_file(anno_path)
  assert n_points <= NUM_PTS, '{} has {} points'.format(anno_path, n_points)
  # read points coordinate
  pts = np.zeros((3, NUM_PTS), dtype='float32')
  point_set = set()
  for line in data:
    try:
      idx, point_x, point_y, oculusion = line.split(' ')
      idx, point_x, point_y, oculusion = int(idx), float(point_x), float(point_y), oculusion == 'True'
      if one_base==False: idx = idx+1
      assert idx >= 1 and idx <= NUM_PTS, 'Wrong idx of points : {:02d}-th in {:s}'.format(idx, anno_path)
      pts[0, idx-1] = point_x
      pts[1, idx-1] = point_y
      pts[2, idx-1] = float( oculusion )
      point_set.add(idx)
    except ValueError:
      raise Exception('error in loading points in {}'.format(anno_path))
  return pts, point_set

def PTSconvert2str(points):
  assert isinstance(points, np.ndarray) and len(points.shape) == 2, 'The points is not right : {}'.format(points)
  assert points.shape[0] == 2 or points.shape[0] == 3, 'The shape of points is not right : {}'.format(points.shape)
  string = ''
  num_pts = points.shape[1]
  for i in range(num_pts):
    ok = False
    if points.shape[0] == 3 and bool(points[2, i]) == True: 
      ok = True
    elif points.shape[0] == 2:
      ok = True

    if ok:
      string = string + '{:02d} {:.2f} {:.2f} True\n'.format(i+1, points[0, i], points[1, i])
  string = string[:-1]
  return string

def PTSconvert2box(points, expand_ratio=None):
  assert isinstance(points, np.ndarray) and len(points.shape) == 2, 'The points is not right : {}'.format(points)
  assert points.shape[0] == 2 or points.shape[0] == 3, 'The shape of points is not right : {}'.format(points.shape)
  if points.shape[0] == 3:
    points = points[:2, points[-1,:].astype('bool') ]
  elif points.shape[0] == 2:
    points = points[:2, :]
  else:
    raise Exception('The shape of points is not right : {}'.format(points.shape))
  assert points.shape[1] >= 2, 'To get the box of points, there should be at least 2 vs {}'.format(points.shape)
  box = np.array([ points[0,:].min(), points[1,:].min(), points[0,:].max(), points[1,:].max() ])
  W = box[2] - box[0]
  H = box[3] - box[1]
  assert W > 0 and H > 0, 'The size of box should be greater than 0 vs {}'.format(box)
  if expand_ratio is not None:
    box[0] = int( math.floor(box[0] - W * expand_ratio) )
    box[1] = int( math.floor(box[1] - H * expand_ratio) )
    box[2] = int( math.ceil(box[2] + W * expand_ratio) )
    box[3] = int( math.ceil(box[3] + H * expand_ratio) )
  return box

def for_generate_box_str(anno_path, num_pts, extend):
  if isinstance(anno_path, str):
    points, _ = anno_parser(anno_path, num_pts)
  else:
    points = anno_path.copy()
  box = PTSconvert2box(points, extend)
  return '{:.2f} {:.2f} {:.2f} {:.2f}'.format(box[0], box[1], box[2], box[3])
    
def resize_heatmap(maps, height, width, order=3):
  # maps  = np.ndarray with shape [height, width, channels]
  # order = 0 Nearest
  # order = 1 Bilinear
  # order = 2 Cubic
  assert isinstance(maps, np.ndarray) and len(maps.shape) == 3, 'maps type : {}'.format(type(maps))
  
  scale = tuple(np.array([height,width], dtype=float) / np.array(maps.shape[:2]))
  return zoom(maps, scale + (1,), order=order)

def analysis_dataset(dataset):
  all_values = np.zeros((3,len(dataset.datas)), dtype=np.float64)
  hs = np.zeros((len(dataset.datas),), dtype=np.float64)
  ws = np.zeros((len(dataset.datas),), dtype=np.float64)

  for index, image_path in enumerate(dataset.datas):
    img = pil_loader(image_path)
    ws[index] = img.size[0]
    hs[index] = img.size[1]
    img = np.array(img)
    all_values[:, index] = np.mean(np.mean(img, axis=0), axis=0).astype('float64')
  mean = np.mean(all_values, axis=1)
  std  = np.std (all_values, axis=1)
  return mean, std, ws, hs

def split_datasets(dataset, point_ids):
  sub_dataset = copy.deepcopy(dataset)
  assert len(point_ids) > 0
  assert False, 'un finished'

def convert68to49(points):
  points = points.copy()
  assert len(points.shape) == 2 and (points.shape[0] == 3 or points.shape[0] == 2) and points.shape[1] == 68, 'The shape of points is not right : {}'.format(points.shape)
  out = np.ones((68,)).astype('bool')
  out[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,60,64]] = False
  cpoints = points[:, out]
  assert len(cpoints.shape) == 2 and cpoints.shape[1] == 49
  return cpoints

def convert68to51(points):
  points = points.copy()
  assert len(points.shape) == 2 and (points.shape[0] == 3 or points.shape[0] == 2) and points.shape[1] == 68, 'The shape of points is not right : {}'.format(points.shape)
  out = np.ones((68,)).astype('bool')
  out[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]] = False
  cpoints = points[:, out]
  assert len(cpoints.shape) == 2 and cpoints.shape[1] == 51
  return cpoints

def merge_lists_from_file(file_paths, seed=None):
  assert file_paths is not None, 'The input can not be None'
  if isinstance(file_paths, str):
    file_paths = [ file_paths ]
  print ('merge lists from {} files with seed={} for random shuffle'.format(len(file_paths), seed))
  # load the data
  all_data = []
  for file_path in file_paths:
    assert osp.isfile(file_path), '{} does not exist'.format(file_path)
    listfile = open(file_path, 'r')
    listdata = listfile.read().splitlines()
    listfile.close()
    all_data = all_data + listdata
  total = len(all_data)
  print ('merge all the lists done, total : {}'.format(total))
  # random shuffle
  if seed is not None:
    np.random.seed(seed)
    order = np.random.permutation(total).tolist()
    new_data = [ all_data[idx] for idx in order ]
    all_data = new_data
  return all_data
