# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, sys, time, random, argparse

def add_shared_args( parser ):
  # Data Generation
  parser.add_argument('--heatmap_type',     type=str,   choices=['gaussian','laplacian'], help='The method for generating the heatmap.')
  parser.add_argument('--data_indicator',   type=str,                   help='The dataset name indicator.')
  parser.add_argument('--use_gray',         type=int,                   help='Use the gray scale image')
  parser.add_argument('--boxindicator',     type=str,                   help='Use which kinds of face box.')
  parser.add_argument('--normalizeL',       type=str,                   help='Noramlized distance indicator.')
  # Data Transform
  parser.add_argument('--pre_crop_expand',  type=float,                 help='parameters for pre-crop expand ratio')
  parser.add_argument('--sigma',            type=float,                 help='sigma distance.')
  parser.add_argument('--scale_prob',       type=float,                 help='argument scale probability.')
  parser.add_argument('--scale_min',        type=float,                 help='argument scale : minimum scale factor.')
  parser.add_argument('--scale_max',        type=float,                 help='argument scale : maximum scale factor.')
  parser.add_argument('--color_disturb',    type=float,                 help='argument color : maximum color factor.')
  parser.add_argument('--rotate_prob',      type=float,                 help='argument rotate : probability.')
  parser.add_argument('--rotate_max',       type=int,                   help='argument rotate : maximum rotate degree.')
  parser.add_argument('--offset_max',       type=float,                 help='argument the maximum random offset ratio.')
  parser.add_argument('--crop_max',         type=float,                 help='argument crop : the maximum randomly crop ratio.')
  parser.add_argument('--robust_iter',      type=int,                   help='argument to test robust : the maximum iteration.')
  parser.add_argument('--height',           type=int,                   help='argument final shape.')
  parser.add_argument('--width',            type=int,                   help='argument final shape.')
  parser.add_argument('--arg_flip',         action='store_true',        help='Using flip data argumentation or not.')
  parser.add_argument('--cutout_length',    type=int,                   help='The cutout length.')
  # Printing
  parser.add_argument('--print_freq',       type=int,   default=100,    help='print frequency (default: 200)')
  parser.add_argument('--print_freq_eval',  type=int,   default=100,    help='print frequency (default: 200)')
  # Checkpoints
  parser.add_argument('--eval_freq',        type=int,                   help='evaluation frequency (default: 200)')
  parser.add_argument('--save_path',        type=str,                   help='Folder to save checkpoints and log.')
  # Acceleration
  parser.add_argument('--shared_img_cache', type=str,   nargs='+',      help='Cache files for Path to PIL Image.')
  parser.add_argument('--workers',          type=int,   default=8,      help='number of data loading workers (default: 2)')
  # Random Seed
  parser.add_argument('--rand_seed',        type=int,                   help='manual seed')
