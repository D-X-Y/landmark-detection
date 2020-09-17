# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
import numpy as np
from os import path as osp


class VideoWriter():

  def __init__(self, save_name, fps=30):
    self.reset(save_name, fps)

  def reset(self, save_name, fps):
    self.video_save_path = save_name
    self.writer = None
    self.vshape = None
    self.fps = fps

  def __repr__(self):
    return ('{name}(fps={fps}, shape={vshape})'.format(name=self.__class__.__name__, **self.__dict__))

  def init(self, shape):
    import cv2
    self.writer = cv2.VideoWriter(self.video_save_path, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, shape)
    self.vshape = shape

  def write_path(self, image_path):
    import cv2
    assert osp.isfile(image_path), '{} does not exist'.format(image_path)
    image = cv2.imread(image_path)
    self.write(image)
      
  def write(self, frame):
    assert isinstance(frame, np.ndarray) and frame.dtype==np.uint8 and len(frame.shape)==3 and frame.shape[2]==3, 'wrong type of frame : {} {}'.format(type(frame), frame.shape)
    if self.writer is None:
      self.init( (frame.shape[1], frame.shape[0]) )
    else:
      assert self.vshape == (frame.shape[1], frame.shape[0]), 'The video frame size is not right : {} vs {}'.format(self.vshape, (frame.shape[1], frame.shape[0]))
    assert self.writer.isOpened(), '{} not opened'.format(self.video_save_path)
    self.writer.write(frame)

  def write_pil(self, pil):
    image = np.array(pil)
    image = image[:,:,[2,1,0]]
    self.write(image)

  def close(self):
    self.writer.release()
    self.writer = None
