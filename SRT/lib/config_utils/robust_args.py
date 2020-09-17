# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, sys, time, random, argparse


def obtain_robust_args():
  parser = argparse.ArgumentParser(description='Train landmark detectors on 300-W or AFLW', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--eval_lists',       type=str,   nargs='+',   help='The list file path to the test dataset.')
  parser.add_argument('--mean_point',       type=str,                help='The mean file path to the image dataset.')
  parser.add_argument('--init_model',       type=str,                help='The path to the pre-trained detection model.')
  parser.add_argument('--save_path',        type=str,                help='The path to save path.')
  parser.add_argument('--robust_cache_dir', type=str,   default='./cache_data/cache/robust-args', help='The path to robustness cache dir.')
  # data augmentation to evaluate the robustness
  parser.add_argument('--robust_scale',     type=float,              help='argument maximum scale range.')
  parser.add_argument('--robust_offset',    type=float,              help='argument maximum offset range.')
  parser.add_argument('--robust_rotate',    type=float,              help='argument maximum rotation degree range.')
  parser.add_argument('--robust_iters',     type=float, default=2,   help='argument maximum .')
  # Optimization options
  parser.add_argument('--print_freq',       type=int,                help='The print frequency.')
  parser.add_argument('--rand_seed',        type=int,                help='The random seed.')
  parser.add_argument('--workers',          type=int,                help='The number of workers.')
  args = parser.parse_args()

  if args.rand_seed is None or args.rand_seed < 0:
    args.rand_seed = random.randint(1, 100000)
  assert args.save_path is not None, 'save-path argument can not be None'
  #state = {k: v for k, v in args._get_kwargs()}
  #Arguments = namedtuple('Arguments', ' '.join(state.keys()))
  #arguments = Arguments(**state)
  return args
