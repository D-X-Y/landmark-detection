# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# For the regression-based detector:
# python exps/BASE-eval-image.py --image ./cache_data/cache/self.jpeg --face 250 150 900 1100 --model ${check_point_path}
#
from __future__ import division

import sys, time, torch, random, argparse, PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
import numpy as np
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from datasets import GeneralDatasetV2 as Dataset, PointMeta2V as PointMeta, pil_loader
from xvision import transforms2v as transforms, draw_image_by_points
from xvision import normalize_points, denormalize_points
from models import obtain_pro_model, remove_module_dict
from config_utils import load_configure


def evaluate(args):
  if args.cuda:
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True
    print ('Use the GPU mode')
  else:
    print ('Use the CPU mode')

  print ('The image is {:}'.format(args.image))
  print ('The model is {:}'.format(args.model))
  last_info_or_snap = Path(args.model)
  assert last_info_or_snap.exists(), 'The model path {:} does not exist'.format(last_info)
  last_info_or_snap = torch.load(last_info_or_snap, map_location=torch.device('cpu'))
  if 'last_checkpoint' in last_info_or_snap:
    snapshot = last_info_or_snap['last_checkpoint']
    assert snapshot.exists(), 'The model path {:} does not exist'.format(snapshot)
    print ('The face bounding box is {:}'.format(args.face))
    assert len(args.face) == 4, 'Invalid face input : {:}'.format(args.face)
    snapshot = torch.load(snapshot, map_location=torch.device('cpu'))
  else:
    snapshot = last_info_or_snap

  param = snapshot['args']
  # General Data Argumentation
  if param.use_gray == False:
    mean_fill = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])
  else:
    mean_fill = (0.5,)
    normalize = transforms.Normalize(mean=[mean_fill[0]], std=[0.5])
  eval_transform  = transforms.Compose2V([transforms.ToTensor(), normalize, \
                                          transforms.PreCrop(param.pre_crop_expand), \
                                          transforms.CenterCrop(param.crop_max)])

  model_config = load_configure(param.model_config, None)
  # dataset = Dataset(eval_transform, param.sigma, model_config.downsample, param.heatmap_type, (120, 96), param.use_gray, None, param.data_indicator)
  dataset = Dataset(eval_transform, param.sigma, model_config.downsample, param.heatmap_type, (param.height, param.width), param.use_gray, None, param.data_indicator)
  dataset.reset( param.num_pts )
  net = obtain_pro_model(model_config, param.num_pts, param.sigma, param.use_gray)
  net.eval()
  try:
    net.load_state_dict( snapshot['detector'] )
  except:
    net.load_state_dict( remove_module_dict(snapshot['detector']) )
  if args.cuda: net = net.cuda()
  print ('Processing the input face image.')
  face_meta = PointMeta(dataset.NUM_PTS, None, args.face, args.image, 'BASE-EVAL') 
  face_img  = pil_loader(args.image, dataset.use_gray)
  affineImage, heatmaps, mask, norm_trans_points, transthetas, _, _, _, shape = dataset._process_(face_img, face_meta, -1)

  # network forward
  with torch.no_grad():
    if args.cuda: inputs = affineImage.unsqueeze(0).cuda()
    else        : inputs = affineImage.unsqueeze(0)
 
    batch_locs = net(inputs)
    batch_locs = batch_locs.cpu()
    (batch_size, C, H, W), num_pts = inputs.size(), param.num_pts
    norm_locs = torch.cat((batch_locs[0].transpose(1,0), torch.ones(1, num_pts)), dim=0)
    norm_locs = torch.mm(transthetas[:2, :], norm_locs)
    real_locs  = denormalize_points(shape.tolist(), norm_locs)
  print ('the coordinates for {:} facial landmarks:'.format(param.num_pts))
  for i in range(param.num_pts):
    point = real_locs[:, i]
    print ('the {:02d}/{:02d}-th landmark : ({:.1f}, {:.1f})'.format(i, param.num_pts, float(point[0]), float(point[1])))

  if args.save:
    resize = 512
    image = draw_image_by_points(args.image, real_locs, 2, (255, 0, 0), args.face, resize)
    image.save(args.save)
    print ('save the visualization results into {:}'.format(args.save))
  else:
    print ('ignore the visualization procedure')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluate a single image by the trained model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--image',            type=str,   help='The evaluation image path.')
  parser.add_argument('--model',            type=str,   help='The snapshot to the saved detector.')
  parser.add_argument('--face',  nargs='+', type=float, help='The coordinate [x1,y1,x2,y2] of a face')
  parser.add_argument('--save',             type=str,   help='The path to save the visualized results.')
  parser.add_argument('--cuda',             action='store_true', help='Use cuda or not.')
  args = parser.parse_args()
  evaluate(args)
