##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
from __future__ import division
import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from .model_utils import get_parameters, load_weight_from_dict, count_network_param
from .basic_batch import find_tensor_peak_batch
from .initialization import weights_init_cpm
from .cycle_util import load_network
from .itn import define_G

class ITN_CPM(nn.Module):
  def __init__(self, model_config):
    super(ITN_CPM, self).__init__()

    self.config = model_config.copy()
    self.downsample = 1

    self.netG_A = define_G()
    self.netG_B = define_G()

    self.features = nn.Sequential(
          nn.Conv2d(  3,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d( 64,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d( 64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))
  
    self.downsample = 8
    pts_num = self.config.pts_num

    self.CPM_feature = nn.Sequential(
          nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), #CPM_1
          nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)) #CPM_2

    self.stage1 = nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True),
          nn.Conv2d(512, pts_num, kernel_size=1, padding=0))

    self.stage2 = nn.Sequential(
          nn.Conv2d(128*2+pts_num*2, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
          nn.Conv2d(128,     pts_num, kernel_size=1, padding=0))

    self.stage3 = nn.Sequential(
          nn.Conv2d(128*2+pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
          nn.Conv2d(128,     pts_num, kernel_size=1, padding=0))
  
    assert self.config.num_stages >= 1, 'stages of cpm must >= 1'

  def set_mode(self, mode):
    if mode.lower() == 'train':
      self.train()
      for m in self.netG_A.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
          m.eval()
      for m in self.netG_B.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
          m.eval()
    elif mode.lower() == 'eval':
      self.eval()
    else:
      raise NameError('The wrong mode : {}'.format(mode))

  def specify_parameter(self, base_lr, base_weight_decay):
    params_dict = [ {'params': self.netG_A.parameters()                   , 'lr': 0        , 'weight_decay': 0},
                    {'params': self.netG_B.parameters()                   , 'lr': 0        , 'weight_decay': 0},
                    {'params': get_parameters(self.features,   bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.features,   bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
                    {'params': get_parameters(self.CPM_feature, bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.CPM_feature, bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
                    {'params': get_parameters(self.stage1,      bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.stage1,      bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
                    {'params': get_parameters(self.stage2,      bias=False), 'lr': base_lr*4, 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.stage2,      bias=True ), 'lr': base_lr*8, 'weight_decay': 0},
                    {'params': get_parameters(self.stage3,      bias=False), 'lr': base_lr*4, 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.stage3,      bias=True ), 'lr': base_lr*8, 'weight_decay': 0}
                  ]
    return params_dict

  # return : cpm-stages, locations
  def forward(self, inputs):
    assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
    batch_size = inputs.size(0)
    num_stages, num_pts = self.config.num_stages, self.config.pts_num - 1

    batch_cpms, batch_locs, batch_scos = [], [], []     # [Squence, Points]

    features, stage1s = [], []
    inputs = [inputs, (self.netG_A(inputs)+self.netG_B(inputs))/2]
    for input in inputs:
      feature  = self.features(input)
      feature = self.CPM_feature(feature)
      features.append(feature)
      stage1s.append( self.stage1(feature) )

    xfeature = torch.cat(features, 1)
    cpm_stage2 = self.stage2(torch.cat([xfeature, stage1s[0], stage1s[1]], 1))
    cpm_stage3 = self.stage3(torch.cat([xfeature, cpm_stage2], 1))
    batch_cpms = [stage1s[0], stage1s[1]] + [cpm_stage2, cpm_stage3]

    # The location of the current batch
    for ibatch in range(batch_size):
      batch_location, batch_score = find_tensor_peak_batch(cpm_stage3[ibatch], self.config.argmax, self.downsample)
      batch_locs.append( batch_location )
      batch_scos.append( batch_score )
    batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)

    return batch_cpms, batch_locs, batch_scos, inputs[1:]


# use vgg16 conv1_1 to conv4_4 as feature extracation        
model_urls = 'https://download.pytorch.org/models/vgg16-397923af.pth'

def itn_cpm(model_config, cycle_model_path):
  
  print ('Initialize ITN-CPM with configure : {}'.format(model_config))
  model = ITN_CPM(model_config)
  model.apply(weights_init_cpm)

  if model_config.pretrained:
    print ('vgg16_base use pre-trained model')
    weights = model_zoo.load_url(model_urls)
    load_weight_from_dict(model, weights, None, False)

  if cycle_model_path:
    load_network(cycle_model_path, 'G_A', model.netG_A)
    load_network(cycle_model_path, 'G_B', model.netG_B)

  print ('initialize the generator network by {} with {} parameters'.format(cycle_model_path, count_network_param(model)))
  return model
