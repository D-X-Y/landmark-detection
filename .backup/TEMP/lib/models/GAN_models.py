import torch
import torch.nn as nn
import functools
import numpy as np

def get_norm_layer(norm_type, dim):
  if norm_type == 'batch':
    norm_layer = nn.BatchNorm2d(dim)
  elif norm_type == 'instance':
    norm_layer = nn.InstanceNorm2d(dim)
  else :
    raise NotImplementedError('normalization layer {:} is not found.'.format(norm_type))
  return norm_layer

# Define a resnet block
class ResnetBlock(nn.Module):
  def __init__(self, dim, padding_type, use_dropout, use_bias, norm_type):
    super(ResnetBlock, self).__init__()

    conv_block = []
    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
      p = 1
    else: raise NotImplementedError('padding {:} is not implemented'.format(padding_type))

    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                   get_norm_layer(norm_type, dim), nn.ReLU(True)]

    if use_dropout: conv_block += [nn.Dropout(0.5)]

    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
      p = 1
    else: raise NotImplementedError('padding {:} is not implemented'.format(padding_type))

    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                   get_norm_layer(norm_type, dim)]

    self.conv_block = nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
  def __init__(self, input_dim, norm_type, n_blocks=6, padding_type='reflect'):
    assert(n_blocks >= 0)
    super(ResnetGenerator, self).__init__()

    model = [nn.ReflectionPad2d(3),
             nn.Conv2d(input_dim, 64, kernel_size=7, padding=0, bias=True),
             get_norm_layer(norm_type, 64),
             nn.ReLU(True)]

    n_downsampling = 2
    for i in range(n_downsampling):
      mult = 2**i
      model += [nn.Conv2d(64 * mult, 64 * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                get_norm_layer(norm_type, 64 * mult * 2), nn.ReLU(True)]

    mult = 2**n_downsampling
    for i in range(n_blocks):
      model += [ResnetBlock(64 * mult, padding_type=padding_type, use_dropout=False, use_bias=True, norm_type=norm_type)]

    for i in range(n_downsampling):
      mult = 2**(n_downsampling - i)
      model += [nn.ConvTranspose2d(64 * mult, int(64 * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                get_norm_layer(norm_type, int(64 * mult / 2)), nn.ReLU(True)]

    model += [nn.ReflectionPad2d(3)]
    model += [nn.Conv2d(64, 3, kernel_size=7, padding=0)]
    model += [nn.Tanh()]

    self.model = nn.Sequential(*model)

  def forward(self, input):
    return self.model(input)

class NLayerDiscriminator(nn.Module):
  def __init__(self, input_dim, norm_type, n_layers=3, use_sigmoid=False):
    super(NLayerDiscriminator, self).__init__()

    kw = 4
    padw = 1
    sequence = [
      nn.Conv2d(input_dim, 64, kernel_size=kw, stride=2, padding=padw),
      nn.LeakyReLU(0.2)
    ]

    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):
      nf_mult_prev = nf_mult
      nf_mult = min(2**n, 8)
      sequence += [
        nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True),
        get_norm_layer(norm_type, 64 * nf_mult),
        nn.LeakyReLU(0.2)
      ]

    nf_mult_prev = nf_mult
    nf_mult = min(2**n_layers, 8)
    sequence += [
      nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True),
      get_norm_layer(norm_type, 64 * nf_mult),
      nn.LeakyReLU(0.2)
    ]

    sequence += [nn.Conv2d(64 * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

    if use_sigmoid:
      sequence += [nn.Sigmoid()]

    self.model = nn.Sequential(*sequence)

  def forward(self, input):
    return self.model(input)
