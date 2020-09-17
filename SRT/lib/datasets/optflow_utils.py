# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
import cv2

def get_optflow_retval(algorithm):
  if algorithm.lower() == 'deepflow':
    retval = cv2.optflow.createOptFlow_DeepFlow()
  elif algorithm.lower() == 'farneback':
    retval = cv2.optflow.createOptFlow_Farneback()
  elif algorithm.lower() == 'tvl1':
    retval = cv2.createOptFlow_DualTVL1()
  elif algorithm.lower() == 'sparse2dense':
    retval = cv2.optflow.createOptFlow_SparseToDense()
  elif algorithm == 'DISflow_ultrafast':
    retval = cv2.optflow.createOptFlow_DIS(0)
  elif algorithm == 'DISflow_fast':
    retval = cv2.optflow.createOptFlow_DIS(1)
  elif algorithm == 'DISflow_medium':
    retval = cv2.optflow.createOptFlow_DIS(2)
  else: raise ValueError('algorithm is not found : {:}'.format( algorithm ))
  return retval
