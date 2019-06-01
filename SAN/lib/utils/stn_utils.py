##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import numpy as np

def normalize_length(x, width):
  return -1. + 2. * x / (width-1)

def get_abs_location(x, width):
  return (x+1)/2.0 * (width-1)

def crop2affine(spatial_size, crop_box):
  # retuen a 2x3 matrix containing the affine-transformation parameters.
  assert isinstance(spatial_size, np.ndarray) and (spatial_size.dtype == 'int32' or spatial_size.dtype == 'int64') and spatial_size.shape == (2,), 'wrong type of spatial_size : {}'.format(spatial_size)
  assert isinstance(crop_box, np.ndarray) and crop_box.shape == (4,), 'wrong type of crop_box : {}'.format(crop_box)
  parameters = np.zeros((2,3), dtype='float32')
  crop_box = crop_box.astype('float32')
  ## normalize
  crop_box[0] = normalize_length(crop_box[0], spatial_size[0])
  crop_box[1] = normalize_length(crop_box[1], spatial_size[1])
  crop_box[2] = normalize_length(crop_box[2], spatial_size[0])
  crop_box[3] = normalize_length(crop_box[3], spatial_size[1])

  parameters[0, 0] = (crop_box[2] - crop_box[0]) / 2
  parameters[0, 2] = (crop_box[0] + crop_box[2]) / 2
  parameters[1, 1] = (crop_box[3] - crop_box[1]) / 2
  parameters[1, 2] = (crop_box[1] + crop_box[3]) / 2

  return parameters

def identity2affine():
  parameters = np.zeros((2,3), dtype='float32')
  parameters[0, 0] = parameters[1, 1] = 1
  return parameters
  
