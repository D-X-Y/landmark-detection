# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# (1) Please use the following command to create the script to extract 300VW frames,
#       you can use ./cache/parallel.cpp to accelerate the script `Extract300VW.sh` by parallel
# python GEN_300VW.py create
# (2) Please use the following command to create the 300VW list for training, after extracting all frames
# python GEN_300VW.py genlist
#
import numpy as np
import math, os, pdb, sys, glob, torch
from pathlib import Path
from tqdm import tqdm
from os import path as osp
from collections import OrderedDict, defaultdict
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
print ('lib-dir : {:}'.format(lib_dir))
import datasets
from xvision import draw_points, normalize_L


def generate_extract_300vw(this_dir, P300VW):
  print ('This Dir : {:}'.format( this_dir ))
  allfiles = glob.glob(osp.join(P300VW, '*'))
  alldirs = []
  for xfile in allfiles:
    if osp.isdir( xfile ):
      alldirs.append(xfile)
  assert len(alldirs) == 114, 'The directories of 300VW should be 114 not {}'.format(len(alldirs))
  cmds = []
  for xdir in alldirs:
    video = osp.join(xdir, 'vid.avi')
    exdir = osp.join(xdir, 'extraction')
    if not osp.isdir(exdir): os.makedirs(exdir)
    cmd = 'ffmpeg -i {:} -q:v 1 {:}/%06d.png'.format(video, exdir)
    cmds.append( cmd )

  save_dir = osp.join(this_dir, 'cache')
  if not osp.isdir(save_dir):
    os.makedirs(save_dir)

  script_path = osp.join(save_dir, 'Extract300VW.sh')
  with open(script_path, 'w') as txtfile:
    for cmd in cmds:
      txtfile.write('{}\n'.format(cmd))
  txtfile.close()
  print ('save the script into {:}'.format(script_path))
  print ('please type \'bash {:}\' to extract video frames'.format(script_path))



def get_name(frame):
  X = frame.split('300VW_Dataset_2015_12_14')
  assert len(X) == 2, 'invalid frame path : {:}'.format(frame)
  name = '300VW' + X[1].replace('/', '-')
  return name


def return_box(pts_path):
  box = datasets.dataset_utils.for_generate_box_str(pts_path, 68, 0, False)
  box = (max(0, box[0]-1), max(0, box[1]-1), box[2]+1, box[3]+1)
  return box


def get_offset_frame(path):
  path      = Path(path)
  ParentDir = path.parent
  Name, Ext = path.name.split('.')
  Previous  = ParentDir / '{:06d}.{:}'.format(int(Name)-1, Ext)
  Next      = ParentDir / '{:06d}.{:}'.format(int(Name)+1, Ext)
  if Previous.exists(): Previous = str(Previous)
  else                : Previous = None
  if Next.exists(): Next = str(Next)
  else            : Next = None
  return Previous, Next


def load_video_dir(root, dirs, save_dir, save_name, subsave):

  save_path = osp.join(save_dir, save_name + '.pth')
  nopoints_save_path = osp.join(save_dir, save_name + '-no-points.pth')
  sub_save_dir = Path(save_dir) / subsave
  if not sub_save_dir.exists(): sub_save_dir.mkdir(parents=True, exist_ok=True)

  Datas, total_frames = [], 0
  mean_landmark = {'GTL' : [[]for i in range(68)]}

  #for idx, cdir in enumerate(dirs):
  for idx in tqdm( range(len(dirs)) ):
    cdir       = dirs[idx]
    annot_path = osp.join(root, cdir, 'annot')
    frame_path = osp.join(root, cdir, 'extraction')
    all_frames = glob.glob( osp.join(frame_path, '*.png') )
    all_annots = glob.glob( osp.join(annot_path, '*.pts') )
    assert len(all_frames) == len(all_annots), 'The length is not right for {:} : {:} vs {:}'.format(cdir, len(all_frames), len(all_annots))
    all_frames, all_annots = sorted(all_frames), sorted(all_annots)
    total_frames += len(all_frames)

    XXDatas = []
    for idx, (frame, annot) in enumerate(zip(all_frames, all_annots)):
      basename_f = osp.basename(frame)
      basename_a = osp.basename(annot)
      assert basename_a[:6] == basename_f[:6], 'The name of {:} is not right with {:}'.format(frame, annot)
      landmarks = datasets.dataset_utils.anno_parser(annot, 68)
  
      data = {'points': landmarks[0],
              'name'  : get_name(frame)}
      box  = return_box(annot)
      data['box-GTL'] = box
      data['box-DET'] = box
      data['box-GTB'] = box
      data['box-default'] = box
      data['normalizeL-default'] = None
      for idx in range(68):
        if int(landmarks[0][2, idx] + 0.5) == 0: continue
        x, y = float(landmarks[0][0,idx]-box[0]), float(landmarks[0][1,idx]-box[1])
        x, y = normalize_L(x, box[2]-box[0]), normalize_L(y, box[3]-box[1])
        mean_landmark['GTL'][idx].append( (x,y) )

      Previous, Next = get_offset_frame(frame)
      data['previous_frame'] = Previous
      data['current_frame']  = frame
      data['next_frame']     = Next
      #Datas[frame] = data
      XXDatas.append( data )
    torch.save(XXDatas, str(sub_save_dir / (cdir + '.pth')))
    Datas = Datas + XXDatas
  print ('--->>> {:} : {:} datas with {:} frames'.format(save_path, len(Datas), total_frames))
  torch.save(Datas, save_path)
  for data in Datas:
    data['points'] = None
  torch.save(Datas, nopoints_save_path)
  print ('--->>> save no-point data into : {:}'.format(nopoints_save_path))

  mean_landmark['GTL'] = np.mean( np.array(mean_landmark['GTL']), axis=1)
  mean_landmark['GTL'] = mean_landmark['GTL'] * 0.9
  image = draw_points(mean_landmark['GTL'], 600, 500)
  image.save(osp.join(save_dir, save_name + '-GTL.png'))
  torch.save(mean_landmark, osp.join(save_dir, save_name + '-mean.pth'))


def generate_300vw_list(root, save_dir):
  assert osp.isdir(root), '{:} is not dir'.format(root)
  if not osp.isdir(save_dir): os.makedirs(save_dir)
  test_1_dirs = [114, 124, 125, 126, 150, 158, 401, 402, 505, 506, 507, 508, 509, 510, 511, 514, 515, 518, 519, 520, 521, 522, 524, 525, 537, 538, 540, 541, 546, 547, 548]
  test_2_dirs = [203, 208, 211, 212, 213, 214, 218, 224, 403, 404, 405, 406, 407, 408, 409, 412, 550, 551, 553]
  test_3_dirs = [410, 411, 516, 517, 526, 528, 529, 530, 531, 533, 557, 558, 559, 562]
  train_dirs  = ['009', '059', '002', '033', '020', '035', '018', '119', '120', '025', '205', '047', '007', '013', '004', '143',
                 '034', '028', '053', '225', '041', '010', '031', '046', '049', '011', '027', '003', '016', '160', '113', '001', '029', '043',
                 '112', '138', '144', '204', '057', '015', '044', '048', '017', '115', '223', '037', '123', '019', '039', '022']

  test_1_dirs = [ '{:}'.format(x) for x in test_1_dirs ]
  test_2_dirs = [ '{:}'.format(x) for x in test_2_dirs ]
  test_3_dirs = [ '{:}'.format(x) for x in test_3_dirs ]
  assert len(train_dirs) == 50, 'The length of train_dirs is not right : {}'.format( len(train_dirs) )
  assert len(test_3_dirs) == 14, 'The length of test_3_dirs is not right : {}'.format( len(test_3_dirs) )
  all_videos = train_dirs + test_1_dirs + test_2_dirs + test_3_dirs
  print ('There are {:} videos in total'.format( len(all_videos) ))
  for idx, video in enumerate( all_videos ):
    print ('[{:3d}] ==== {:}'.format(idx+1, video))

  load_video_dir(root,  train_dirs, save_dir, '300VW.train' , 'TRAIN')
  load_video_dir(root, test_1_dirs, save_dir, '300VW.test-1', 'A')
  load_video_dir(root, test_2_dirs, save_dir, '300VW.test-2', 'B')
  load_video_dir(root, test_3_dirs, save_dir, '300VW.test-3', 'C')
  print ('Save all dataset files into {:}'.format(save_dir))


if __name__ == '__main__':
  HOME_STR = 'DOME_HOME'
  if HOME_STR not in os.environ: HOME_STR = 'HOME'
  assert HOME_STR in os.environ, 'Doest not find the HOME dir : {}'.format(HOME_STR)

  this_dir = osp.dirname(os.path.abspath(__file__))
  SAVE_DIR = osp.join(this_dir, 'lists', '300VW')
  print ('This dir : {}, HOME : [{}] : {}'.format(this_dir, HOME_STR, os.environ[HOME_STR]))
  P300VW = osp.join(os.environ[HOME_STR], 'datasets', 'landmark-datasets', '300VW_Dataset_2015_12_14')
  if sys.argv[1] == 'create':
    generate_extract_300vw(this_dir, P300VW)
  elif sys.argv[1] == 'genlist':
    generate_300vw_list(P300VW, SAVE_DIR)
  else:
    raise ValueError('Invalid argv : {:}'.format( sys.argv ))
