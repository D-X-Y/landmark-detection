# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, sys, math, copy, torch, sqlite3
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

#Change this paths according to your directories
this_dir = osp.dirname(os.path.abspath(__file__))
SAVE_DIR = osp.join(this_dir, 'lists', 'AFLW')
HOME_STR = 'DOME_HOME'
if HOME_STR not in os.environ: HOME_STR = 'HOME'
assert HOME_STR in os.environ, 'Doest not find the HOME dir : {}'.format(HOME_STR)
print ('This dir : {:}, HOME : [{:}] : {:}'.format(this_dir, HOME_STR, os.environ[HOME_STR]))
if not osp.isdir(SAVE_DIR): os.makedirs(SAVE_DIR)
image_dir = osp.join(os.environ[HOME_STR], 'datasets', 'landmark-datasets', 'AFLW', 'images')
print ('The AFLW image dir : {:}'.format(image_dir))
assert osp.isdir(image_dir), 'The image dir : {:} does not exist'.format(image_dir)


def get_name(image_path):
  X = image_path.split('AFLW/images')
  assert len(X) == 2, 'Invalid image path : {:}'.format(image_path)
  name = 'AFLW' + X[1].replace('/', '-')
  return name
  

class AFLWFace():
  def __init__(self, index, name, mask, landmark, box):
    self.image_path = name
    self.face_id = index
    self.face_box = [float(box[0]), float(box[2]), float(box[1]), float(box[3])]
    mask = np.expand_dims(mask, axis=1)
    landmark = landmark.copy()
    self.landmarks = np.concatenate((landmark, mask), axis=1)
    self.yaw = None
    self.pitch = None
    self.roll = None
  
  def set_pose(self, yaw, pitch, roll):
    self.yaw   = yaw
    self.pitch = pitch
    self.roll  = roll

  def get_face_size(self, use_box):
    box = []
    if use_box == 'GTL':
      box = datasets.dataset_utils.PTSconvert2box(self.landmarks.copy().T)
      box = (max(0, box[0]-1), max(0, box[1]-1), box[2]+1, box[3]+1)
    elif use_box == 'GTB':
      box = (self.face_box[0], self.face_box[1], self.face_box[2], self.face_box[3])
    else:
      assert False, 'The box indicator not find : {:}'.format(use_box)
    assert box[2] > box[0], 'The size of box is not right [{:}] : {:}'.format(self.face_id, box)
    assert box[3] > box[1], 'The size of box is not right [{:}] : {:}'.format(self.face_id, box)
    face_size = math.sqrt( float(box[3]-box[1]) * float(box[2]-box[0]) )
    return box, face_size

  def check_front(self):
    oks = 0
    box = self.face_box
    for idx in range(self.landmarks.shape[0]):
      if bool(self.landmarks[idx,2]):
        x, y = self.landmarks[idx,0], self.landmarks[idx,1]
        if x > self.face_box[0] and x < self.face_box[2]:
          if y > self.face_box[1] and y < self.face_box[3]:
            oks = oks + 1
    return oks == 19
    
  def __repr__(self):
    return ('{name}(path={image_path}, face-id={face_id})'.format(name=self.__class__.__name__, **self.__dict__))


def save_to_list_file(allfaces, lst_file, image_style_dir, face_indexes, use_front, USE_BOXES):
  if face_indexes is not None:
    save_faces = []
    for index in face_indexes:
      face = allfaces[index]
      if use_front == False or face.check_front():
        save_faces.append( face )
  else:
    save_faces = allfaces
  print ('Prepare to save {:05} face images into {:}'.format(len(save_faces), lst_file))

  mean_landmark = {SUFFIX : [[]for i in range(19)] for SUFFIX in USE_BOXES}

  Datas = []
  for index, face in enumerate(save_faces):
    image_path = face.image_path
    image_path = osp.join(image_style_dir, image_path)
   
    landmarks = face.landmarks.T.copy()
    try:
      assert osp.isfile(image_path), 'The image [{:}/{:}] {:} does not exsit'.format(index, len(save_faces), image_path)
    except AssertionError:
      # some image extensions are incorrect
      image_path = image_path.replace('.png', '.jpg')
      assert osp.isfile(image_path), 'The image [{:}/{:}] {:} does not exsit'.format(index, len(save_faces), image_path)
    data = {'points': landmarks,
            'name'  : get_name(image_path)}
  
    for SUFFIX in USE_BOXES:
      box, face_size = face.get_face_size(SUFFIX)
      data['box-{:}'.format(SUFFIX)] = box
      data['face-size-{:}'.format(SUFFIX)] = face_size
      for idx in range(19):
        if int(landmarks[2, idx] + 0.5) == 0: continue
        x, y = float(landmarks[0,idx]-box[0]), float(landmarks[1,idx]-box[1])
        x, y = normalize_L(x, box[2]-box[0]), normalize_L(y, box[3]-box[1])
        mean_landmark[SUFFIX][idx].append( [x,y] )
      data['normalizeL-{:}'.format(SUFFIX)] = face_size
    data['box-default']    = data['box-GTL']
    data['normalizeL-default'] = data['normalizeL-GTL']
    data['face-size-default'] = data['face-size-GTL']
    data['previous_frame'] = None
    data['current_frame']  = image_path
    data['next_frame']     = None
    Datas.append( data ) #[image_path] = data
  torch.save(Datas, lst_file + '.pth')

  for SUFFIX in USE_BOXES:
    alls = []
    for idx in range(19):
      allp = mean_landmark[SUFFIX][idx]
      allp = np.array(allp)
      pts  = np.mean(allp, axis=0)
      alls.append(pts)
    alls = np.array(alls)
    mean_landmark[SUFFIX] = alls * 0.9
    image = draw_points(mean_landmark[SUFFIX], 600, 500, True)
    image.save( lst_file + '-{:}.png'.format(SUFFIX) )
  mean_landmark['default'] = mean_landmark['GTL']
  torch.save(mean_landmark, lst_file + '-mean.pth')



if __name__ == "__main__":
  mat_path = osp.join(this_dir, 'cache', 'AFLWinfo_release.mat')
  aflwinfo = dict()
  mat = loadmat(mat_path)
  total_image = 24386
  # load train/test splits
  ra = np.squeeze(mat['ra']-1).tolist()
  aflwinfo['train-index'] = ra[:20000]
  aflwinfo['test-index'] = ra[20000:]
  aflwinfo['name-list'] = []
  # load name-list
  for i in range(total_image):
    name = mat['nameList'][i,0][0]
    #name = name[:-4] + '.jpg'
    aflwinfo['name-list'].append( name )
  aflwinfo['mask'] = mat['mask_new'].copy()
  aflwinfo['landmark'] = mat['data'].reshape((total_image, 2, 19))
  aflwinfo['landmark'] = np.transpose(aflwinfo['landmark'], (0,2,1))
  aflwinfo['box'] = mat['bbox'].copy()
  allfaces = []
  for i in range(total_image):
    face = AFLWFace(i, aflwinfo['name-list'][i], aflwinfo['mask'][i], aflwinfo['landmark'][i], aflwinfo['box'][i])
    allfaces.append( face )
  
  #USE_BOXES = ['GTL', 'GTB']
  USE_BOXES = ['GTL', 'GTB']
  save_to_list_file(allfaces, osp.join(SAVE_DIR, 'train')     , image_dir, aflwinfo['train-index'], False, USE_BOXES)
  save_to_list_file(allfaces, osp.join(SAVE_DIR, 'test')      , image_dir, aflwinfo['test-index'],  False, USE_BOXES)
  save_to_list_file(allfaces, osp.join(SAVE_DIR, 'test.front'), image_dir, aflwinfo['test-index'],  True,  USE_BOXES)
  save_to_list_file(allfaces, osp.join(SAVE_DIR, 'all')       , image_dir, aflwinfo['train-index'] + aflwinfo['test-index'], False, USE_BOXES)

  # use aflw 30 yaw > 60 yaw
  mat_path = osp.join(this_dir, 'cache', 'aflw-sqlite.pth')
  sq_info  = torch.load( mat_path )
  test_faces = [allfaces[i] for i in aflwinfo['test-index']]
  faces_less_30 = []
  faces_grtr_60 = []
  for i, face in enumerate(test_faces):
    xpath = face.image_path
    if xpath not in sq_info:
      print ('Does not find this path : {:}'.format(xpath))
      continue
    else:
      xlist, find = sq_info[xpath], False
      for xpose in xlist:
        if xpose['rect'][0] == int(face.face_box[0]) and xpose['rect'][1] == int(face.face_box[1]):
          face.set_pose(*xpose['pose'][0])
          find = True ; break
      if not find: print ('Does not find suitable pose for this path : {:}'.format(xpath))
    if not find: continue
    if abs(face.yaw) < math.pi * 30 / 180:
      faces_less_30.append( face )
    if abs(face.yaw) > math.pi * 60 / 180:
      faces_grtr_60.append( face)
  print ('There are {:} faces with < 30 yaw degree'.format(len(faces_less_30)))
  print ('There are {:} faces with > 60 yaw degree'.format(len(faces_grtr_60)))
  save_to_list_file(faces_less_30, osp.join(SAVE_DIR, 'test-less-30-yaw'), image_dir, None, None, USE_BOXES)
  save_to_list_file(faces_grtr_60, osp.join(SAVE_DIR, 'test-more-60-yaw'), image_dir, None, None, USE_BOXES)
  print ('-'*100)
