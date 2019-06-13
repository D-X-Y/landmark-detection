# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division

import os, sys, time, random, argparse, PIL
from os import path as osp
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
import numbers, numpy as np
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from xvision import draw_image_by_points
from xvision import Eval_Meta

def visualize(args):

  print ('The result file is {:}'.format(args.meta))
  print ('The save path is {:}'.format(args.save))
  meta = Path(args.meta)
  save = Path(args.save)
  assert meta.exists(), 'The model path {:} does not exist'
  xmeta = Eval_Meta()
  xmeta.load(meta)
  print ('this meta file has {:} predictions'.format(len(xmeta)))
  if not save.exists(): os.makedirs( args.save )
  for i in range(len(xmeta)):
    image, prediction = xmeta.image_lists[i], xmeta.predictions[i]
    name = osp.basename(image)
    image = draw_image_by_points(image, prediction, 2, (255, 0, 0), False, False)
    path = save / name
    image.save(path)
    print ('{:03d}-th image is saved into {:}'.format(i, path))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='visualize the results on a single ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--meta',            type=str,   help='The evaluation image path.')
  parser.add_argument('--save',            type=str,   help='The path to save the visualized results.')
  args = parser.parse_args()
  visualize(args)
