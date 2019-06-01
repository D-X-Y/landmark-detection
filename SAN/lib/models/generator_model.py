import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

# Define a resnet block
class ResnetBlock(nn.Module):
  def __init__(self, dim, padding_type, use_dropout, use_bias):
    super(ResnetBlock, self).__init__()
    self.conv_block = self.build_conv_block(dim, padding_type, use_dropout, use_bias)

  def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
    conv_block = []
    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
             nn.InstanceNorm2d(dim, affine=False),
             nn.ReLU(True)]
    if use_dropout:
      conv_block += [nn.Dropout(0.5)]

    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
             nn.InstanceNorm2d(dim, affine=False)]

    return nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
  def __init__(self, n_blocks=6, gpu_ids=[], padding_type='reflect'):
    assert(n_blocks >= 0)
    super(ResnetGenerator, self).__init__()
    self.gpu_ids = gpu_ids

    model = [nn.ReflectionPad2d(3),
         nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=True),
         nn.InstanceNorm2d(64, affine=False),
         nn.ReLU(True)]

    n_downsampling = 2
    for i in range(n_downsampling):
      mult = 2**i
      model += [nn.Conv2d(64 * mult, 64 * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(64 * mult * 2, affine=False),
            nn.ReLU(True)]

    mult = 2**n_downsampling
    for i in range(n_blocks):
      model += [ResnetBlock(64 * mult, padding_type=padding_type, use_dropout=False, use_bias=True)]

    for i in range(n_downsampling):
      mult = 2**(n_downsampling - i)
      model += [nn.ConvTranspose2d(64 * mult, int(64 * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(int(64 * mult / 2), affine=False),
            nn.ReLU(True)]
    model += [nn.ReflectionPad2d(3)]
    model += [nn.Conv2d(64, 3, kernel_size=7, padding=0)]
    model += [nn.Tanh()]

    self.model = nn.Sequential(*model)

  def forward(self, input):
    if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
      return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
    else:
      return self.model(input)
