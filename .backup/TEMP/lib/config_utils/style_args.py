# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, sys, time, random, argparse

def obtain_style_args():
  parser = argparse.ArgumentParser(description='Generate style images.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--train_lists',      type=str,   nargs='+',      help='The list file path to the video training dataset.')
  parser.add_argument('--eval_lists',       type=str,   nargs='+',      help='The list file path to the video testing dataset.')
  parser.add_argument('--num_pts',          type=int,                   help='Number of point.')
  parser.add_argument('--epochs',           type=int,                   help='The number of training epoch.')
  parser.add_argument('--critic_iters',     type=int,                   help='How many critic iterations per generator iteration?')
  # GAN Related
  parser.add_argument('--gan_norm',         type=str,                   help='The normalization type in the GAN models.')
  parser.add_argument('--LR_D',             type=float,                 help='The learning rate ratio for the discriminator')
  parser.add_argument('--LR_G',             type=float,                 help='The learning rate ratio for the generator')
  parser.add_argument('--gradient_penalty', action='store_true',        help='The gradient penalty in the improved WGAN.')
  parser.add_argument('--gp_lambda',        type=float,                 help='The coefficient of the gradient penalty in the improved WGAN.')
  parser.add_argument('--wgan_clip',        type=float,                 help='The clip parameter of WGAN.')
  parser.add_argument('--eval_freq',        type=int,                   help='The frequency of evaluation.')
  parser.add_argument('--amsgrad',          type=int,                   help='Whether to use amsgrad or not.')
  parser.add_argument('--debug',            action='store_true',        help='Set debug mode or not')
  parser.add_argument('--use_tf',           action='store_true',        help='Whether to use tensorboard or not.')

  # Data Generation
  parser.add_argument('--heatmap_type',     type=str,   choices=['gaussian','laplacian'], help='The method for generating the heatmap.')
  parser.add_argument('--data_indicator',   type=str, default='300W-68',help='The method for generating the heatmap.')
  # Data Transform
  parser.add_argument('--pre_crop_expand',  type=float,                 help='parameters for pre-crop expand ratio')
  parser.add_argument('--sigma',            type=float,                 help='sigma distance for CPM.')
  parser.add_argument('--scale_prob',       type=float,                 help='argument scale probability.')
  parser.add_argument('--scale_min',        type=float,                 help='argument scale : minimum scale factor.')
  parser.add_argument('--scale_max',        type=float,                 help='argument scale : maximum scale factor.')
  parser.add_argument('--scale_eval',       type=float,                 help='argument scale : maximum scale factor.')
  parser.add_argument('--rotate_max',       type=int,                   help='argument rotate : maximum rotate degree.')
  parser.add_argument('--crop_height',      type=int,   default=256,    help='argument crop : crop height.')
  parser.add_argument('--crop_width',       type=int,   default=256,    help='argument crop : crop width.')
  parser.add_argument('--crop_perturb_max', type=int,                   help='argument crop : center of maximum perturb distance.')
  parser.add_argument('--arg_flip',         action='store_true',        help='Using flip data argumentation or not ')
  # Optimization options
  parser.add_argument('--batch_size',       type=int,   default=2,      help='Batch size for training.')
  # Checkpoints
  parser.add_argument('--print_freq',       type=int,   default=100,    help='print frequency (default: 200)')
  parser.add_argument('--save_path',        type=str,                   help='Folder to save checkpoints and log.')
  # Acceleration
  parser.add_argument('--workers',          type=int,   default=8,      help='number of data loading workers (default: 2)')
  # Random Seed
  parser.add_argument('--rand_seed',        type=int,                   help='manual seed')
  args = parser.parse_args()

  if args.rand_seed is None:
    args.rand_seed = random.randint(1, 100000)
  assert args.save_path is not None, 'save-path argument can not be None'

  #state = {k: v for k, v in args._get_kwargs()}
  #Arguments = namedtuple('Arguments', ' '.join(state.keys()))
  #arguments = Arguments(**state)
  return args
