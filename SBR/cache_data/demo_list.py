# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, pdb, sys, glob, cv2
from os import path as osp
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
print ('lib-dir : {:}'.format(lib_dir))
from datasets import pil_loader
from utils.file_utils import load_list_from_folders, load_txt_file
# ffmpeg -i shooui.mp4 -filter:v "crop=450:680:10:120" -c:a copy ~/Desktop/demo.mp4


def generate(demo_dir, list_dir, savename, check):
  imagelist, num_image = load_list_from_folders(demo_dir, ext_filter=['png'], depth=1)
  assert num_image == check, 'The number of images is not right vs. {:}'.format(num_image)
  if not osp.isdir(list_dir): os.makedirs(list_dir)
  
  gap, x1, y1, x2, y2 = 5, 5, 5, 450, 680

  imagelist.sort()

  txtfile = open(osp.join(list_dir, savename), 'w')
  for idx, image in enumerate(imagelist):
    if idx < 2 or idx + 2 >= len(imagelist): continue
    box_str = '{:.1f} {:.1f} {:.1f} {:.1f}'.format(gap, gap, x2-x1-gap, y2-y1-gap)
    txtfile.write('{:} {:} {:}\n'.format(image, 'none', box_str))
    txtfile.flush()
  txtfile.close()
  print('there are {:} images for the demo video sequence'.format(num_image))

if __name__ == '__main__':
  HOME_STR = 'DOME_HOME'
  if HOME_STR not in os.environ: HOME_STR = 'HOME'
  assert HOME_STR in os.environ, 'Doest not find the HOME dir : {}'.format(HOME_STR)

  this_dir = osp.dirname(os.path.abspath(__file__))
  demo_dir = osp.join(this_dir, 'cache', 'demo-sbrs')
  list_dir = osp.join(this_dir, 'lists', 'demo')
  print ('This dir : {}, HOME : [{}] : {}'.format(this_dir, HOME_STR, os.environ[HOME_STR]))
  generate(demo_dir, list_dir, 'demo-sbr.lst', 275)

  #demo_dir = osp.join(this_dir, 'cache', 'demo-pams')
  #list_dir = osp.join(this_dir, 'lists', 'demo')
  #print ('This dir : {}, HOME : [{}] : {}'.format(this_dir, HOME_STR, os.environ[HOME_STR]))
  #generate(demo_dir, list_dir, 'demo-pam.lst', 253)
