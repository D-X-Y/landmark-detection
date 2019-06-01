# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from PIL import Image
from scipy.ndimage.interpolation import zoom
from utils.file_utils import load_txt_file
from .dataset_utils import convert68to49 as _convert68to49
from .dataset_utils import convert68to51 as _convert68to51
import numpy as np
import copy, math

class Point_Meta():
  # points: 3 x num_pts (x, y, oculusion)
  # image_size: original [width, height]
  def __init__(self, num_point, points, box, image_path, dataset_name):

    self.num_point = num_point
    assert len(box.shape) == 1 and box.shape[0] == 4, 'The shape of box is not right : {}'.format( box )
    self.box = box.copy()
    if points is None:
      self.points = points
    else:
      assert len(points.shape) == 2 and points.shape[0] == 3 and points.shape[1] == self.num_point, 'The shape of point is not right : {}'.format( points )
      self.points = points.copy()
    self.update_center()
    self.image_path = image_path
    self.datasets = dataset_name
    self.temp_save_wh = None

  def __repr__(self):
    return ('{name}(number of point={num_point})'.format(name=self.__class__.__name__, **self.__dict__))

  def convert68to49(self):
    if self.points is not None:
      self.points = _convert68to49(self.points)

  def convert68to51(self):
    if self.points is not None:
      self.points = _convert68to51(self.points)

  def update_center(self):
    if self.points is not None:
      self.center = np.mean(self.points[:2, self.points[2,:]>0], axis=1)
    else:
      self.center = np.array([ (self.box[0]+self.box[2])/2, (self.box[1]+self.box[3])/2 ])

  def apply_bound(self, width, height):
    if self.points is not None:
      oks = np.vstack((self.points[0, :] >= 0, self.points[1, :] >=0, self.points[0, :] <= width, self.points[1, :] <= height, self.points[2, :].astype('bool')))
      oks = oks.transpose((1,0))
      self.points[2, :] = np.sum(oks, axis=1) == 5
    self.box[0], self.box[1] = np.max([self.box[0], 0]),     np.max([self.box[1], 0])
    self.box[2], self.box[3] = np.min([self.box[2], width]), np.min([self.box[3], height])

  def apply_scale(self, scale):
    if len(scale) == 1:   # scale the same size for both x and y
      if self.points is not None:
        self.points[:2, self.points[2,:]>0] = self.points[:2, self.points[2,:]>0] * scale[0]
      self.center                         = self.center   * scale[0]
      self.box[0], self.box[1]            = self.box[0] * scale[0], self.box[1] * scale[0]
      self.box[2], self.box[3]            = self.box[2] * scale[0], self.box[3] * scale[0]
    elif len(scale) == 2: # scale the width and height
      if self.points is not None:
        self.points[0, self.points[2,:]>0] = self.points[0, self.points[2,:]>0] * scale[0]
        self.points[1, self.points[2,:]>0] = self.points[1, self.points[2,:]>0] * scale[1]
      self.center[0]                     = self.center[0] * scale[0]
      self.center[1]                     = self.center[1] * scale[1]
      self.box[0], self.box[1]            = self.box[0] * scale[0], self.box[1] * scale[1]
      self.box[2], self.box[3]            = self.box[2] * scale[0], self.box[3] * scale[1]
    else:
      assert False, 'Does not support this scale : {}'.format(scale)

  def apply_offset(self, ax=None, ay=None):
    if ax is not None:
      if self.points is not None:
        self.points[0, self.points[2,:]>0] = self.points[0, self.points[2,:]>0] + ax
      self.center[0]                     = self.center[0] + ax
      self.box[0], self.box[2]           = self.box[0] + ax, self.box[2] + ax
    if ay is not None:
      if self.points is not None:
        self.points[1, self.points[2,:]>0] = self.points[1, self.points[2,:]>0] + ay
      self.center[1]                     = self.center[1] + ay
      self.box[1], self.box[3]           = self.box[1] + ay, self.box[3] + ay

  def apply_rotate(self, center, degree):
    degree = math.radians(-degree)
    if self.points is not None:
      vis_xs = self.points[0, self.points[2,:]>0]
      vis_ys = self.points[1, self.points[2,:]>0]
      self.points[0, self.points[2,:]>0] = (vis_xs - center[0]) * np.cos(degree) - (vis_ys - center[1]) * np.sin(degree) + center[0]
      self.points[1, self.points[2,:]>0] = (vis_xs - center[0]) * np.sin(degree) + (vis_ys - center[1]) * np.cos(degree) + center[1]
    # rotate the box
    corners = np.zeros((4,2))
    corners[0,0], corners[0,1] = self.box[0], self.box[1]
    corners[1,0], corners[1,1] = self.box[0], self.box[3]
    corners[2,0], corners[2,1] = self.box[2], self.box[1]
    corners[3,0], corners[3,1] = self.box[2], self.box[3]
    corners[:, 0] = (corners[:, 0] - center[0]) * np.cos(degree) - (corners[:, 1] - center[1]) * np.sin(degree) + center[0]
    corners[:, 1] = (corners[:, 0] - center[0]) * np.sin(degree) - (corners[:, 1] - center[1]) * np.cos(degree) + center[1]
    self.box[0], self.box[1] = corners[0,0], corners[0,1]
    self.box[2], self.box[3] = corners[3,0], corners[3,1]
    
  def apply_horizontal_flip(self, width):
    self.points[0, :] = width - self.points[0, :] - 1
    # Mugsy spefic or Synthetic
    if self.datasets == 'Mugsy.full_face_v1':
      ori = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
      pos = np.array([ 3,  4,  1,  2,  9, 10, 11, 12,  5,  6,  7,  8, 14, 13, 15, 16, 17, 18, 19, 20])
      self.points[:, pos-1] = self.points[:, ori-1]
    elif self.datasets == 'Synthetic.v1':
      ori = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
      pos = np.array([ 3,  4,  1,  2,  9, 10, 11, 12,  5,  6,  7,  8, 14, 13, 15, 16, 17, 18, 19, 20])
      self.points[:, pos-1] = self.points[:, ori-1]
    else:
      assert False, 'Does not support {}.{}'.format(self.datasets, self.subsets)
    
  # all points' range [0, w) [0, h)
  def check_nan(self):
    if math.isnan(self.center[0]) or math.isnan(self.center[1]):
      return True
    for i in range(self.num_point):
      if self.points[2, i] > 0:
        if math.isnan(self.points[0, i]) or math.isnan(self.points[1, i]):
          return True
    return False

  def visiable_pts_num(self):
    ans = self.points[2,:]>0
    return np.sum(ans)

  def set_precrop_wh(self, W, H, x1, y1, x2, y2):
    self.temp_save_wh = [W, H, x1, y1, x2, y2]

  def get_box(self):
    return self.box.copy()

  def get_points(self):
    if self.points is not None:
      return self.points.copy()
    else:
      return np.zeros((3, self.num_point), dtype='float32')

  def is_none(self):
    assert self.box is not None, 'The box should not be None'
    return self.points is None

  def copy(self):
    return copy.deepcopy(self)
