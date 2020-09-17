# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#############################################################
# python GEN_list.py /mnt/home/dongxuanyi/datasets/landmark-datasets/for_xuanyi/ears.train.lst 19 lists/demo/ears.train.pth
import os, pdb, sys, glob, torch
from os import path as osp
from collections import OrderedDict, defaultdict
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
print ('lib-dir : {:}'.format(lib_dir))
import datasets
from utils.file_utils import load_list_from_folders
from PIL import Image
import re


def generate(file_list, NUM_PTS, savepath):
  with open(file_list) as fl:  
    imagelist = [l.strip().split() for l in fl]

  samples = []
  for idx, image_info in enumerate(imagelist):
    print ('load {:}-th :: {:}'.format(idx, image_info))
    landmarks = None
    if len(image_info) == 6:
      img_path, label_path, x1, y1, x2, y2 = image_info
      landmarks = datasets.dataset_utils.anno_parser_v1(label_path, NUM_PTS, one_base=True)[0]
    elif len(image_info) == 2:
      img_path, label_path = image_info
      x1, y1 = 0, 0
      img = Image.open(img_path)
      x2, y2 = img.size
      landmarks = datasets.dataset_utils.anno_parser_v1(label_path, NUM_PTS, one_base=True)[0]
    else:
      img_path = image_info[0]
      x1, y1 = 0, 0
      img = Image.open(img_path)
      x2, y2 = img.size

    box = [x1, y1, x2-x1, y2-y1]
  

    data = {'points' : landmarks,
            'box'    : box,
            'box-DET': box,
            'box-default': box,
            'name'   : re.sub('/', '_', file_list)}
    data['previous_frame'] = None
    data['current_frame']  = img_path
    data['next_frame']     = None
    samples.append( data )
  torch.save(samples, savepath)
  print('there are {:} images, and save them into {:}.'.format(len(imagelist), savepath))

if __name__ == '__main__':

  if len(sys.argv) != 4:
    print("input_file_list num_pts output_pth")
    print("input_file_list can have two columns (img_path, anno_path) or six columns (img_path, anno_path, tlx, tly, brx, bry)")
    sys.exit(-1)
  else:
    print (sys.argv)

  generate(sys.argv[1], int(sys.argv[2]), sys.argv[3])
