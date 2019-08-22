# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import warnings
from os import path as osp

def parse_basic(ori_filename, length_l, length_r):
  folder = osp.dirname(ori_filename)
  filename = osp.basename(ori_filename)
  # 300-VW
  if folder[-10:] == 'extraction':
    assert filename[-4:] == '.png', 'The filename is not right : {}'.format(filename)
    idx = int(filename[: filename.find('.png')])
    assert idx >= 0, 'The index must be greater than 0'
    images = []
    for i in range(idx-length_l, idx+length_r+1):
      path = osp.join(folder, '{:06d}.png'.format(i))
      if not osp.isfile(path):
        xpath = osp.join(folder, '{:06d}.png'.format(idx))
        warnings.warn('Path [{}] does not exist, maybe it reaches the start or end of the video, use {} instead.'.format(path, xpath), UserWarning)
        path = xpath
      assert osp.isfile(path), '!!WRONG file path : {}, the original frame is {}'.format(path, filename)
      images.append(path)
    return images, True
  # YouTube Cele..
  elif folder.find('YouTube_Celebrities_Annotation') > 0:
    assert filename[-4:] == '.png', 'The filename is not right : {}'.format(filename)
    idx = int(filename[filename.find('_')+1: filename.find('.png')])
    assert idx >= 0, 'The index must be greater than 0'
    images = []
    for i in range(idx-length_l, idx+length_r+1):
      path = osp.join(folder, 'frame_{:05d}.png'.format(i))
      if not osp.isfile(path):
        xpath = osp.join(folder, 'frame_{:05d}.png'.format(idx))
        warnings.warn('Path [{}] does not exist, maybe it reaches the start or end of the video, use {} instead.'.format(path, xpath), UserWarning)
        path = xpath
      assert osp.isfile(path), '!!WRONG file path : {}, the original frame is {}'.format(path, filename)
      images.append(path)
    return images, True
  # Talking Face..
  elif folder.find('talking_face') > 0:
    assert filename[-4:] == '.jpg', 'The filename is not right : {}'.format(filename)
    idx = int(filename[filename.find('_')+1: filename.find('.jpg')])
    assert idx >= 0, 'The index must be greater than 0'
    images = []
    for i in range(idx-length_l, idx+length_r+1):
      path = osp.join(folder, 'franck_{:05d}.png'.format(i))
      if not osp.isfile(path):
        xpath = osp.join(folder, 'franck_{:05d}.png'.format(idx))
        warnings.warn('Path [{}] does not exist, maybe it reaches the start or end of the video, use {} instead.'.format(path, xpath), UserWarning)
        path = xpath
      assert osp.isfile(path), '!!WRONG file path : {}, the original frame is {}'.format(path, filename)
      images.append(path)
    return images, True
  # YouTube Face..
  elif folder.find('YouTube-Face') > 0:
    assert filename[-4:] == '.jpg', 'The filename is not right : {}'.format(filename)
    splits = filename.split('.')
    assert len(splits) == 3, 'The format is not right : {}'.format(filename)
    idx = int(splits[1])
    images = []
    for i in range(idx-length_l, idx+length_r+1):
      path = osp.join(folder, '{}.{}.{}'.format(splits[0], i, splits[2]))
      if not osp.isfile(path):
        xpath = osp.join(folder, '{}.{}.{}'.format(splits[0], idx, splits[2]))
        warnings.warn('Path [{}] does not exist, maybe it reaches the start or end of the video, use {} instead.'.format(path, xpath), UserWarning)
        path = xpath
      assert osp.isfile(path), '!!WRONG file path : {}, the original frame is {}'.format(path, filename)
      images.append(path)
    return images, True
  elif folder.find('demo-pams') > 0 or folder.find('demo-sbrs') > 0:
    assert filename[-4:] == '.png', 'The filename is not right : {}'.format(filename)
    assert filename[:5] == 'image', 'The filename is not right : {}'.format(filename)
    splits = filename.split('.')
    assert len(splits) == 2, 'The format is not right : {}'.format(filename)
    idx = int(splits[0][5:])
    images = []
    for i in range(idx-length_l, idx+length_r+1):
      path = osp.join(folder, 'image{:04d}.{:}'.format(i, splits[1]))
      if not osp.isfile(path):
        xpath = osp.join(folder, 'image{:04d}.{:}'.format(idx, splits[1]))
        warnings.warn('Path [{}] does not exist, maybe it reaches the start or end of the video, use {} instead.'.format(path, xpath), UserWarning)
        path = xpath
      assert osp.isfile(path), '!!WRONG file path : {}, the original frame is {}'.format(path, filename)
      images.append(path)
    return images, True
  else:
    return [ori_filename] * (length_l+length_r+1), False

def parse_video_by_indicator(image_path, parser, return_info=False):
  if parser is None or parser.lower() == 'none':
    method, offset_l, offset_r = 'None', 0, 0
  else:
    parser = parser.split('-')
    assert len(parser) == 3, 'The video parser must be 3 elements : {:}'.format(parser)
    method, offset_l, offset_r = parser[0], int(parser[1]), int(parser[2])
  if return_info:
    return offset_l, offset_r
  else:
    images, is_video_or_not = parse_basic(image_path, offset_l, offset_r)
  return images, is_video_or_not
