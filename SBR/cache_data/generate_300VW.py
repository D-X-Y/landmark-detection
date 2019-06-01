# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import math, os, pdb, sys, glob
from os import path as osp
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
print ('lib-dir : {:}'.format(lib_dir))
import datasets

EXPAND_RATIO = 0.0
afterfix='.10'


def str2size(box_str):
  splits = box_str.split(' ')
  x1, y1, x2, y2 = float(splits[0]), float(splits[1]), float(splits[2]), float(splits[3])
  return math.sqrt( (x2-x1) * (y2-y1) )


def load_video_dir(root, dirs, save_dir, save_name):
  videos, sparse_videos = [], []
  first_videos = []
  for idx, cdir in enumerate(dirs):
    annot_path = osp.join(root, cdir, 'annot')
    frame_path = osp.join(root, cdir, 'extraction')
    all_frames = glob.glob( osp.join(frame_path, '*.png') )
    all_annots = glob.glob( osp.join(annot_path, '*.pts') )
    assert len(all_frames) == len(all_annots), 'The length is not right for {} : {} vs {}'.format(cdir, len(all_frames), len(all_annots))
    all_frames = sorted(all_frames)
    all_annots = sorted(all_annots)
    current_video = []
    txtfile = open(osp.join(save_dir, save_name + cdir), 'w')
    nonefile = open(osp.join(save_dir, save_name + cdir + '.none'), 'w')

    all_sizes = []
    for frame, annot in zip(all_frames, all_annots):
      basename_f = osp.basename(frame)
      basename_a = osp.basename(annot)
      assert basename_a[:6] == basename_f[:6], 'The name of {} is not right with {}'.format(frame, annot)
      current_video.append( (frame, annot) )
      box_str = datasets.dataset_utils.for_generate_box_str(annot, 68, EXPAND_RATIO)
      txtfile.write('{} {} {}\n'.format(frame, annot, box_str))
      nonefile.write('{} None {}\n'.format(frame, box_str))
      all_sizes.append( str2size(box_str) )
      if len(current_video) == 1:
        first_videos.append( (frame, annot) )
    txtfile.close()
    nonefile.close()
    videos.append( current_video )
    all_sizes = np.array( all_sizes )
    print ('--->>> {:} : [{:02d}/{:02d}] : {:} has {:} frames | face size : mean={:.2f}, std={:.2f}'.format(save_name, idx, len(dirs), cdir, len(all_frames), all_sizes.mean(), all_sizes.std()))

    for jxj, video in enumerate(current_video):
      if jxj <= 3 or jxj + 3 >= len(current_video): continue
      if jxj % 10 == 3:
        sparse_videos.append( video )

  txtfile = open(osp.join(save_dir, save_name), 'w')
  nonefile = open(osp.join(save_dir, save_name + '.none'), 'w')
  for video in videos:
    for cpair in video:
      box_str = datasets.dataset_utils.for_generate_box_str(cpair[1], 68, EXPAND_RATIO)
      txtfile.write('{} {} {}\n'.format(cpair[0], cpair[1], box_str))
      nonefile.write('{} {} {}\n'.format(cpair[0], 'None', box_str))
      txtfile.flush()
      nonefile.flush()
  txtfile.close()
  nonefile.close()

  txtfile = open(osp.join(save_dir, save_name + '.sparse' + afterfix), 'w')
  nonefile = open(osp.join(save_dir, save_name + '.sparse.none' + afterfix), 'w')
  for cpair in sparse_videos:
    box_str = datasets.dataset_utils.for_generate_box_str(cpair[1], 68, EXPAND_RATIO)
    txtfile.write('{} {} {}\n'.format(cpair[0], cpair[1], box_str))
    nonefile.write('{} {} {}\n'.format(cpair[0], 'None', box_str))
  txtfile.close()
  nonefile.close()

  txtfile = open(osp.join(save_dir, save_name + '.first'), 'w')
  for cpair in first_videos:
    box_str = datasets.dataset_utils.for_generate_box_str(cpair[1], 68, EXPAND_RATIO)
    txtfile.write('{} {} {}\n'.format(cpair[0], cpair[1], box_str))
  txtfile.close()

  print ('{} finish save into {}'.format(save_name, save_dir))
  return videos

def generate_300vw_list(root, save_dir):
  assert osp.isdir(root), '{} is not dir'.format(root)
  if not osp.isdir(save_dir): os.makedirs(save_dir)
  test_1_dirs = [114, 124, 125, 126, 150, 158, 401, 402, 505, 506, 507, 508, 509, 510, 511, 514, 515, 518, 519, 520, 521, 522, 524, 525, 537, 538, 540, 541, 546, 547, 548]
  test_2_dirs = [203, 208, 211, 212, 213, 214, 218, 224, 403, 404, 405, 406, 407, 408, 409, 412, 550, 551, 553]
  test_3_dirs = [410, 411, 516, 517, 526, 528, 529, 530, 531, 533, 557, 558, 559, 562]
  train_dirs  = ['009', '059', '002', '033', '020', '035', '018', '119', '120', '025', '205', '047', '007', '013', '004', '143',
                 '034', '028', '053', '225', '041', '010', '031', '046', '049', '011', '027', '003', '016', '160', '113', '001', '029', '043',
                 '112', '138', '144', '204', '057', '015', '044', '048', '017', '115', '223', '037', '123', '019', '039', '022']

  test_1_dirs, test_2_dirs, test_3_dirs = [ '{}'.format(x) for x in test_1_dirs], [ '{}'.format(x) for x in test_2_dirs], [ '{}'.format(x) for x in test_3_dirs]
  #all_dirs = os.listdir(root)
  #train_dirs = set(all_dirs) - set(test_1_dirs) - set(test_2_dirs) - set(test_3_dirs) - set(['ReadMe.txt', 'extra.zip'])
  #train_dirs = list( train_dirs )
  assert len(train_dirs) == 50, 'The length of train_dirs is not right : {}'.format( len(train_dirs) )
  assert len(test_3_dirs) == 14, 'The length of test_3_dirs is not right : {}'.format( len(test_3_dirs) )

  load_video_dir(root,  train_dirs, save_dir, '300VW.train.lst')
  load_video_dir(root, test_1_dirs, save_dir, '300VW.test-1.lst')
  load_video_dir(root, test_2_dirs, save_dir, '300VW.test-2.lst')
  load_video_dir(root, test_3_dirs, save_dir, '300VW.test-3.lst')

if __name__ == '__main__':
  HOME_STR = 'DOME_HOME'
  if HOME_STR not in os.environ: HOME_STR = 'HOME'
  assert HOME_STR in os.environ, 'Doest not find the HOME dir : {}'.format(HOME_STR)

  this_dir = osp.dirname(os.path.abspath(__file__))
  SAVE_DIR = osp.join(this_dir, 'lists', '300VW')
  print ('This dir : {}, HOME : [{}] : {}'.format(this_dir, HOME_STR, os.environ[HOME_STR]))
  path_300vw = osp.join(os.environ[HOME_STR], 'datasets', 'landmark-datasets', '300VW_Dataset_2015_12_14')
  generate_300vw_list(path_300vw, SAVE_DIR)
