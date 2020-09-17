# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Supervision-by-Registration

import sys, time, torch, random, argparse, PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
import numbers, numpy as np
from os import path as osp
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)

from xvision   import Eval_Meta, draw_image_by_points_failure_case as draw_image_by_points
from xvision   import normalize_points, denormalize_points
from log_utils import AverageMeter, time_for_file, convert_secs2time, time_string


def get_box(points):
  if points.shape[0] == 3:
    points = points[:2, points[2,:]==1]
  [x1, y1, x2, y2] = [points[0].min(), points[1].min(), points[0].max(), points[1].max()]
  H, W = y2-y1, x2-x1
  x1 = max(0, x1 - W * 0.1)
  y1 = max(0, y1 - H * 0.1)
  x2, y2 = x2 + W * 0.1, y2 + H * 0.1
  return [x1, y1, x2, y2]


def main(save_dir, meta, mindex, maximum):
  save_dir = Path( save_dir )
  save_dir.mkdir(parents=True, exist_ok=True)
  assert osp.isfile( meta ), 'invalid meta file : {:}'.format(meta)
  checkpoint = torch.load(meta)
  xmeta      = checkpoint[ mindex ]
  RED, GREEN, BLUE = (255, 0,   0), (0, 255,   0), (0,   0, 255)
  
  random.seed( 111 )
  index_list = list(range(len(xmeta)))
  random.shuffle(index_list)

  for i in range(0, min(maximum, len(xmeta))):
    index = index_list[i]
    image, predicts, gts = xmeta[index]
    crop_box = get_box(gts)
    num_pts  = predicts.shape[1]
    predicts, gts = torch.Tensor(predicts), torch.Tensor(gts)
    avaliable = gts[2,:] == 1
    predicts, gts = predicts[:2, avaliable], gts[:2, avaliable]

    colors   = [ BLUE for _ in range( avaliable.sum().item() ) ] + [ GREEN for _ in range( avaliable.sum().item() ) ]
    points   = torch.cat((gts, predicts), dim=1)

    image    = draw_image_by_points(image, points, 3, colors, crop_box, (400, 500))
    image.save('{:}/image-{:05d}.png'.format(save_dir, index))
  print('save into {:}'.format(save_dir))


if __name__ == '__main__':
  main('snapshots/DEBUG-DXY/MPII/basic',             './snapshots/MPII/BASE-HG-S3-B16-1/metas/seed-17086-hourglass.pth'                   , 0, 100)
  main('snapshots/DEBUG-DXY/MPII/SBR',               './snapshots/MPII/BASE-HG-S4-B16-1/metas/seed-82751-hourglass.pth'                   , 0, 100)

  main('snapshots/DEBUG-DXY/300WW/basic-regression', './snapshots/300W/REG-DET-300W-RD02-default-96x96-32/metas/seed-84217-regression.pth', 2, 20)
  main('snapshots/DEBUG-DXY/300WW/sbr-regression',   './snapshots/300W/REG-SBR-300WVW-L1.W05-default-96x96-32.32/metas/regression-epoch-199-200.pth', 2, 20)
  main('snapshots/DEBUG-DXY/300VW/basic-regression', './snapshots/300W/REG-SBR-300WVW-L1.W05-default-96x96-32.32/metas/regression-first.pth', 3, 20)
  main('snapshots/DEBUG-DXY/300VW/sbr-regression',   './snapshots/300W/REG-SBR-300WVW-L1.W05-default-96x96-32.32/metas/regression-epoch-199-200.pth', 3, 20)

  main('snapshots/DEBUG-DXY/AFLWW/basic-regression', './snapshots/AFLW/HG-DET-ADAM-128x128-S2-B8/metas/hourglass-epoch-159-160.pth', 1, 20)
  main('snapshots/DEBUG-DXY/AFLWW/sbr-regression',   './snapshots/AFLW/HG-DET-RMSP-120x96-B12/metas/hourglass-epoch-140-150.pth',    1, 20)

  main('snapshots/DEBUG-DXY/Mugsy/basic-regression', './snapshots/Mugsy/REG-DET.RD09-Mugsy-240x320-32/metas/seed-28475-regression.pth', 1, 200)
  main('snapshots/DEBUG-DXY/Mugsy/sbr-regression',   './snapshots/Mugsy/REG-SRT-240x320-W05.W05-0.32.16/metas/regression-epoch-199-200.pth', 1, 200)

