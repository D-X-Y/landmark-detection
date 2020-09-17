# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
##
# python MPII.py --dir ~/dongxuanyi/datasets/landmark-datasets/MPII --image --video | bash
import os, sys, time, argparse
from pathlib import Path


parser = argparse.ArgumentParser(description='Download Dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir',     type=str,            help='Path to save dataset')
parser.add_argument('--image',   action='store_true', help='download video data or not.')
parser.add_argument('--video',   action='store_true', help='download image data or not.')
args = parser.parse_args()


def get_image(cdir):
  image_url = 'https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz'
  annot_url = 'https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip'
  cmds = ['wget {:} -P {:}'.format(image_url, cdir),
          'wget {:} -P {:}'.format(annot_url, cdir)]
  return cmds


def get_video(cdir):
  cmds = []
  for i in range(1, 26):
    xurl = 'https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_sequences_batch{:}.tar.gz'.format(i)
    cmds.append('wget {:} -P {:}'.format(xurl, cdir))
  xurl = 'https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_sequences_keyframes.mat'
  cmds.append( xurl )
  return cmds

if __name__ == '__main__':
  assert args.dir is not None, '--dir can not be None'
  cdir = Path(args.dir)
  if not cdir.exists(): cdir.mkdir(parents=True, exist_ok=True)
  cmds = []
  if args.image: cmds += get_image(cdir)
  if args.video: cmds += get_video(cdir)
  for x in cmds:
    print ( x )
