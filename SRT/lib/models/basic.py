# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from .ProHG    import ProHourGlass
from .ProCPM   import ProCPM
from .ProCU    import ProCUNet
from .ProMSPN  import ProMSPNet
from .ProREG   import ProRegression
from .ProHRNet import ProHRNet
from .ProTemporal_HEAT import TemporalHEAT
from .ProTemporal_REG  import TemporalREG
from .STM_REG  import SpatialTemporalMultiviewREG
from .STM_HEAT import SpatialTemporalMultiviewHEAT


"""
def obtain_model(configure, points):
  from .cpm_vgg16 import cpm_vgg16
  from .hourglass import hourglass
  if configure.arch == 'cpm_vgg16':
    net = cpm_vgg16(configure, points)
  elif configure.arch == 'hourglass':
    net = hourglass(configure, points)
  else:
    raise TypeError('Unkonw type : {:}'.format(configure.arch))
  return net
"""
def obtain_pro_stm(configure, stm_config, points, sigma, use_gray):
  model = obtain_pro_model(configure, points, sigma, use_gray)
  if stm_config.arch == 'heatmap':
    stm_model = SpatialTemporalMultiviewHEAT(model, stm_config, model.pts_num)
  elif stm_config.arch == 'regression':
    stm_model = SpatialTemporalMultiviewREG(model, stm_config, model.pts_num)
  else:
    raise TypeError('Unkonw type : {:}'.format(stm_config))
  return stm_model


def obtain_pro_temporal(configure, sbr_config, points, sigma, use_gray):
  model = obtain_pro_model(configure, points, sigma, use_gray)
  if sbr_config.arch == 'heatmap':
    sbr_model = TemporalHEAT(model, sbr_config, model.pts_num)
  elif sbr_config.arch == 'regression':
    sbr_model = TemporalREG (model, sbr_config, model.pts_num)
  else:
    raise TypeError('Unkonw type : {:}'.format(sbr_config))
  return sbr_model

def obtain_pro_model(configure, points, sigma, use_gray):
  if configure.arch == 'hourglass':
    net = ProHourGlass(configure, points + configure.background, sigma, use_gray)
  elif configure.arch == 'cpm_vgg16':
    net = ProCPM(configure, points + configure.background, sigma, use_gray)
  elif configure.arch == 'cunet':
    net = ProCUNet(configure, points + configure.background, sigma, use_gray)
  elif configure.arch == 'regression':
    net = ProRegression(configure, points + configure.background, use_gray)
  elif configure.arch == 'mspn':
    net = ProMSPNet(configure, points + configure.background, sigma, use_gray)
  elif configure.arch == 'hrnet':
    net = ProHRNet(configure, points + configure.background, sigma, use_gray)
  else:
    raise TypeError('Unkonw type : {:}'.format(configure.arch))
  return net
