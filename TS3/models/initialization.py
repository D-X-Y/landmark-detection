import torch
import torch.nn as nn
from torch.nn import init

def weights_init_cpm(m):
  classname = m.__class__.__name__
  # print(classname)
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0, 0.01)
    if m.bias is not None: m.bias.data.zero_()
  elif classname.find('BatchNorm2d') != -1:
    m.weight.data.fill_(1)
    m.bias.data.zero_()

def weights_init_normal(m):
  classname = m.__class__.__name__
  # print(classname)
  if classname.find('Conv') != -1:
    init.uniform(m.weight.data, 0.0, 0.02)
  elif classname.find('Linear') != -1:
    init.uniform(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm2d') != -1:
    init.uniform(m.weight.data, 1.0, 0.02)
    init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
  classname = m.__class__.__name__
  # print(classname)
  if classname.find('Conv') != -1:
    init.xavier_normal(m.weight.data, gain=1)
  elif classname.find('Linear') != -1:
    init.xavier_normal(m.weight.data, gain=1)
  elif classname.find('BatchNorm2d') != -1:
    init.uniform(m.weight.data, 1.0, 0.02)
    init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
  classname = m.__class__.__name__
  # print(classname)
  if classname.find('Conv') != -1:
    init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
  elif classname.find('Linear') != -1:
    init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
  elif classname.find('BatchNorm2d') != -1:
    init.uniform(m.weight.data, 1.0, 0.02)
    init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
  classname = m.__class__.__name__
  print(classname)
  if classname.find('Conv') != -1:
    init.orthogonal(m.weight.data, gain=1)
  elif classname.find('Linear') != -1:
    init.orthogonal(m.weight.data, gain=1)
  elif classname.find('BatchNorm2d') != -1:
    init.uniform(m.weight.data, 1.0, 0.02)
    init.constant(m.bias.data, 0.0)
