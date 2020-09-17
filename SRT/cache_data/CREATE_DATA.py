# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
import os, sys, cv2, torch
from os import path as osp
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
print ('lib-dir : {:}'.format(lib_dir))
this_dir = osp.dirname(os.path.abspath(__file__))
from datasets import cv2_loader
from log_utils import time_string


def GET_300W_LIST():
  PATHS = [('300-TRAIN', './lists/300W/300w.train.pth'),
           ('300-T-COM', './lists/300W/300w.test-common.pth'),
           ('300-T-CHL', './lists/300W/300w.test-challenge.pth'),
           ('300-T-FUL', './lists/300W/300w.test-full.pth')]
  return PATHS


def load_save2dir(PATHS, save_dir):
  if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)

  for indicator, PATH in PATHS:
    datas = torch.load( PATH )
    for idx, (key, value) in enumerate(datas.items()):
      image = cv2_loader(key, cv2, False)
      value['image'] = image
      value['previous'] = None
      value['after'] = None

      if idx % 500 == 0:
        print('{:} {:10s} {:04d}/{:04d}'.format(time_string(), indicator, idx, len(datas)))
    torch.save(datas, save_dir / (indicator+'.pth'))
  

if __name__ == '__main__':
  HOME_STR = 'DOME_HOME'
  if HOME_STR not in os.environ: HOME_STR = 'HOME'
  #save_dir = osp.join( os.environ[HOME_STR], 'datasets', 'landmark-datasets', 'cv2-data')
  save_dir = Path( osp.join(this_dir, 'cv2-data') )

  paths = GET_300W_LIST()
  load_save2dir(paths, save_dir)
