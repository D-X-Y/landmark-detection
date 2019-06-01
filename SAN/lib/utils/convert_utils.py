##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import numpy as np
import struct

def isscalar(scalar_test):
    return isinstance(scalar_test, int) or isinstance(scalar_test, float)

def scalar_list2float_list(scalar_list):
  '''
  remove an empty string from a list
  '''
  assert isinstance(scalar_list, list) and all(isscalar(scalar_tmp) for scalar_tmp in scalar_list), 'input list is not a scalar list'
  float_list = list()
  for item in scalar_list:
    float_list.append(float(item))
  return float_list 

def float_list2bytes(float_list):
  '''
  convert a float number to a set of bytes
  '''
  assert isinstance(float_list, float) or (isinstance(float_list, list) and all(isinstance(float_tmp, float) for float_tmp in float_list)), 'input is not a floating number or a list of floating number' 
  # convert a single floating number to a list with one item 
  if isinstance(float_list, float):
    float_list = [float_list]
  try:
    binary = struct.pack('%sf' % len(float_list), *float_list)
  except ValueError:
    print('Warnings!!!! Failed to convert to bytes!!!!!') 
  return binary  
