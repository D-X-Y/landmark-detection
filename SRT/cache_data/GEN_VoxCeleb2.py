# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# cat vox2_dev* > vox2_mp4.zip
# python GEN_VoxCeleb2.py create ./cache/extract_vox.sh
# ./parallel extract_vox.sh 100
# python GEN_VoxCeleb2.py genlist
# python GEN_VoxCeleb2.py clean > ./cache/delet_Vox.sh
#

import os
from os import path as osp
import cv2, sys, math, torch, subprocess, copy, numpy as np
from tqdm import tqdm
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
print ('lib-dir : {:}'.format(lib_dir))
from log_utils import time_string


def get_num_frame(video):
  cmd_n = 'ffprobe -select_streams v -show_streams {:} 2>/dev/null | grep nb_frames | sed -e \'s/nb_frames=//\''.format( video )
  p = subprocess.Popen(cmd_n, shell=True, stdout=subprocess.PIPE)
  (output, err) = p.communicate()
  nums = int(output.strip())
  assert nums > 0, "{:}".format(video)
  return nums


def create_frame_script(root, script):
  root = Path(root)
  assert root.exists(), 'root : {:} not exist'.format(root)
  dev_dir = root / 'dev' / 'mp4'
  assert dev_dir.exists(), 'dev : {:} not exist'.format(dev_dir)
  IDdirs = [x for x in dev_dir.glob("id*") if x.is_dir()]
  print ('There are {:} identities in {:}'.format( len(IDdirs), dev_dir ))
  assert len(IDdirs) == 5994, '{:} vs {:}'.format(len(IDdirs), 5994)
  IDdirs = sorted( IDdirs )

  def get_videos(xdir):
    videos = []
    for xtemp in xdir.glob("*"):
      if not xtemp.is_dir(): continue
      #first_segment = list( xtemp.glob("*.mp4") )[0]
      all_sub_videos = sorted( list( xtemp.glob("*.mp4") ) )
      for _video in all_sub_videos:
        videos.append( _video )
        break
      #videos.append( first_segment )
      if len(videos) >= 5: break
    videos = sorted(videos, key=lambda x: x.name)
    for idx, video in enumerate(videos):
      if idx > 0: assert videos[idx-1].name != videos[idx].name, '{:} vs {:}'.format(videos[idx-1], videos[idx])
    return videos

  if script is not None: save_file = open(script, 'w')
  else                 : save_file = None
  
  num_videos, num_frames = 0, 0
  #for index in tqdm( range( len(IDdirs) ) ):
  #  ID     = IDdirs[index]
  for index, ID in enumerate( IDdirs ):
    videos = get_videos(ID)
    num_videos += len( videos )
    print ('ID={:05d}/{:} :: {:} has {:} videos'.format(index, len(IDdirs), ID, len(videos)))
    for iv, video in enumerate( videos ):
      #num_frames += get_num_frame(video)
      frame_dir   = root / 'frames' / ID.name / (video.name.split('.')[0])
      if not frame_dir.exists(): frame_dir.mkdir(parents=True, exist_ok=True)
      command = 'ffmpeg -i {:} -start_number 0 -threads 1 -vframes 50 -q:v 1 {:}/%08d.png'.format(video, frame_dir)
      if save_file: save_file.write('{:}\n'.format( command ))
  print ('There are {:} IDs, {:} videos, {:} frames'.format(len(IDdirs), num_videos, num_frames))

  if save_file is not None:
    print ('save into {:}'.format(script))
    save_file.close()


def delect_all(root):
  root = Path(root) / 'frames'
  assert root.exists(), 'root : {:} not exist'.format(root)
  subdirs = [x for x in root.glob("*") if x.is_dir()]
  for ID in subdirs:
    for tdir in ID.glob('*'):
      print('rm -rf {:}'.format(tdir))


def generate_list(root, save_dir):
  root, save_dir = Path(root) / 'frames', Path( save_dir )
  assert root.exists(), 'root : {:} not exist'.format(root)
  if not save_dir.exists(): save_dir.mkdir(parents=True, exist_ok=True)

  def get_frames(cdir, name, xlist):
    ans = [cdir / '{:08d}.png'.format(name+offset) for offset in xlist]
    for i in range(len(ans)):
      if not ans[i].exists(): ans[i] = None
      else                  : ans[i] = str(ans[i])
    return ans

  IDS   = [x for x in root.glob("*") if x.is_dir()]
  Datas, num_videos = [], 0
  for index, ID in enumerate(IDS):
    print('{:} {:5d} / {:} :: {:} || {:} frames'.format(time_string(), index, len(IDS), ID, len(Datas)))
    videos = [x for x in ID.glob("*") if x.is_dir()]
    for video in videos:
      frames = list( video.glob("*.png") )
      if not (len(frames) <= 50 and len(frames) >= 40): continue
      frames = sorted( frames )
      num_videos += 1
      for frame in frames:
        Past, Now, Future = get_frames(video, int(frame.name.split('.')[0]), [-1,0,1])
        assert Now is not None, '{:}'.format( frame )
        data = {'previous_frame': Past,
                'current_frame' : Now,
                'next_frame'    : Future,
                'points'        : None,
                'box-default'   : None,
                'normalizeL-default' : None}
        Datas.append( data )
  print ('There are {:} videos and {:} frames'.format(num_videos, len(Datas)))
  save_path = save_dir / 'Unlabeled-VoxCeleb2-Videos.pth'
  torch.save(Datas, str(save_path), pickle_protocol=4)
  print ('Save into {:}'.format(save_path))
  print ('-' * 100)
  
  


if __name__ == '__main__':
  HOME_STR = 'DOME_HOME'
  if HOME_STR not in os.environ: HOME_STR = 'HOME'
  assert HOME_STR in os.environ, 'Doest not find the HOME dir : {}'.format(HOME_STR)
  this_dir = osp.dirname(os.path.abspath(__file__))
  SAVE_DIR = Path(this_dir) / 'lists' / 'VoxCeleb2'
  print ('This dir : {:}, HOME : [{:}] : {:}'.format(this_dir, HOME_STR, os.environ[HOME_STR]))
  VoxCeleb2 = osp.join( os.environ[HOME_STR], 'datasets', 'landmark-datasets', 'VoxCeleb2')
  assert len(sys.argv) >= 2, 'There must have one arg vs {:}'.format( sys.argv )

  if sys.argv[1] == 'create':
    if len(sys.argv) == 3  : script_path = sys.argv[2]
    elif len(sys.argv) == 2: script_path = None
    else: raise ValueError('Invalid argv : {:}'.format(sys.argv))
    create_frame_script(VoxCeleb2, script_path)
  elif sys.argv[1] == 'genlist':
    generate_list(VoxCeleb2, SAVE_DIR)
  elif sys.argv[1] == 'clean':
    delect_all(VoxCeleb2)
  else:
    raise ValueError('Invalid argv : {:}'.format( sys.argv ))
