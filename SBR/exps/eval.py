# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division

import sys, time, torch, random, argparse, PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
import numbers, numpy as np
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from datasets import GeneralDataset as Dataset
from xvision  import transforms, draw_image_by_points
from models   import obtain_model, remove_module_dict
from utils    import get_model_infos
from config_utils import load_configure


def evaluate(args):
  if not args.cpu:
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True

  print ('The image is {:}'.format(args.image))
  print ('The model is {:}'.format(args.model))
  snapshot = Path(args.model)
  assert snapshot.exists(), 'The model path {:} does not exist'
  print ('The face bounding box is {:}'.format(args.face))
  assert len(args.face) == 4, 'Invalid face input : {:}'.format(args.face)
  if args.cpu: snapshot = torch.load(snapshot, map_location='cpu')
  else       : snapshot = torch.load(snapshot)

  # General Data Argumentation
  mean_fill   = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
  normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

  param = snapshot['args']
  eval_transform  = transforms.Compose([transforms.PreCrop(param.pre_crop_expand), transforms.TrainScale2WH((param.crop_width, param.crop_height)), transforms.ToTensor(), normalize])
  model_config = load_configure(param.model_config, None)
  dataset = Dataset(eval_transform, param.sigma, model_config.downsample, param.heatmap_type, param.data_indicator)
  dataset.reset(param.num_pts)
  
  net = obtain_model(model_config, param.num_pts + 1)
  if not args.cpu: net = net.cuda()
  #import pdb; pdb.set_trace()
  try:
    weights = remove_module_dict(snapshot['detector'])
  except:
    weights = remove_module_dict(snapshot['state_dict'])
  net.load_state_dict(weights)
  print ('Prepare input data')
  [image, _, _, _, _, _, cropped_size], meta = dataset.prepare_input(args.image, args.face)
  # network forward
  with torch.no_grad():
    if args.cpu: inputs = image.unsqueeze(0)
    else       : inputs = image.unsqueeze(0).cuda()
    batch_heatmaps, batch_locs, batch_scos = net(inputs)
    flops, params = get_model_infos(net, inputs.shape)
    print ('IN-shape : {:}, FLOPs : {:} MB, Params : {:} MB'.format(list(inputs.shape), flops, params))
  # obtain the locations on the image in the orignial size
  cpu = torch.device('cpu')
  np_batch_locs, np_batch_scos, cropped_size = batch_locs.to(cpu).numpy(), batch_scos.to(cpu).numpy(), cropped_size.numpy()
  locations, scores = np_batch_locs[0,:-1,:], np.expand_dims(np_batch_scos[0,:-1], -1)

  scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2) , cropped_size[1] * 1. / inputs.size(-1)

  locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + cropped_size[3]
  prediction = np.concatenate((locations, scores), axis=1).transpose(1,0)

  print ('the coordinates for {:} facial landmarks:'.format(param.num_pts))
  for i in range(param.num_pts):
    point = prediction[:, i]
    print ('the {:02d}/{:02d}-th point : ({:.1f}, {:.1f}), score = {:.2f}'.format(i, param.num_pts, float(point[0]), float(point[1]), float(point[2])))

  if args.save:
    resize = 512
    image = draw_image_by_points(args.image, prediction, 2, (255, 0, 0), args.face, resize)
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
  parser.add_argument('--cpu',     action='store_true', help='Use CPU or not.')
  args = parser.parse_args()
  evaluate(args)
