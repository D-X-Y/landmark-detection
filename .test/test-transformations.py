import os, sys, time, torch, random, argparse, PIL
from PIL import Image, ImageFile
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
import numbers, numpy as np
import torch.nn.functional as F
lib_dir = (Path(__file__).parent / '..' / 'SRT' / 'lib').resolve()
if str(lib_dir) not in sys.path:
  sys.path.insert(0, str(lib_dir))
print(lib_dir)
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
this_dir = Path(__file__).parent.resolve()
import datasets
from datasets.point_meta_v1 import Point_Meta
from xvision import transforms1v as transforms
from xvision import get_image_from_affine
from xvision import normalize_points
from xvision import draw_image_by_points


def get_list():
  with open('{:}/DEBUG.lst'.format(this_dir), 'r') as cfile:
    contents = cfile.readlines()
  images, labels, boxes = [], [], []
  for content in contents:
    image, label, x1,y1,x2,y2 = content.strip().split(' ')
    boxes.append( (float(x1),float(y1),float(x2),float(y2)) )
    images.append( image )
    labels.append( label )
  return images, labels, boxes


transform_dict = {
  'precrop': transforms.PreCrop(0.2),
  'scale'  : transforms.AugScale(1, 0.7, 1.3),
  # 'crop'   : transforms.AugCrop(0.8),
  'rotate' : transforms.AugRotate(30)
}


def get_transforms(transtr):
  transform_funcs = []
  for indicator in transtr.split('-'):
    transform_funcs.append(transform_dict[indicator])
  return transform_funcs


def main(use_gray, transform_strs):
  if not use_gray:
    mean_fill = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0,0,0], std=[1,1,1])
    color     = (102,255,102)
  else:
    mean_fill = (0.5,)
    normalize = transforms.Normalize(mean=[mean_fill[0]], std=[0.5])
    normalize = transforms.Normalize(mean=[0], std=[1])
    color     = (255,)

  debug_dir = '{:}/cache/gray-{:}'.format(this_dir, use_gray)
  if not os.path.isdir(debug_dir):
    os.makedirs( debug_dir )

  transform_funcs = [transforms.ToTensor(), normalize] + get_transforms(transform_strs)
  transform = transforms.Compose(transform_funcs)

  shape = (300, 200)
  images, labels, boxes = get_list()
  for image, label, box in zip(images, labels, boxes):
    imgx = datasets.pil_loader( image, use_gray )
    np_points, _ = datasets.anno_parser(label, 68)
    meta = Point_Meta(68, np_points, box, image, 'face68')
    I, L, theta = transform(imgx, meta)
    points = torch.Tensor(L.get_points(True))
    points = normalize_points((I.size(1), I.size(2)), points)
    name   = Path(image).name
    image  = get_image_from_affine(I, theta, shape)
    points = torch.cat( (points, torch.ones((1, points.shape[1]))), dim=0)
    # new_points, LU = torch.gesv(points, theta)
    new_points, _ = torch.solve(points, theta)
    
    PImage = draw_image_by_points(image, new_points[:2,:], 2, color, False, False, True, draw_idx=True)
    
    save_name = os.path.join(debug_dir, '{:}-{:}'.format(transform_strs, name))
    PImage.save( save_name )


if __name__ == '__main__':
  main(False, 'rotate-precrop')
  # main(False, 'precrop')
