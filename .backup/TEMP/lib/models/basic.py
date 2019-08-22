# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from .cpm_vgg16 import cpm_vgg16
from .hourglass import hourglass
from .GAN_models import ResnetGenerator, NLayerDiscriminator

def obtain_model(configure, points):
  if configure.arch == 'cpm_vgg16':
    net = cpm_vgg16(configure, points)
  elif configure.arch == 'hourglass':
    net = hourglass(configure, points)
  else:
    raise TypeError('Unkonw type : {:}'.format(configure.arch))
  return net


def obtain_GAN(input_dim, norm_type):
  #from .initialization import weights_init_xavier as init
  from .initialization import weights_init_wgan as init
  #norm_layer = get_norm_layer(norm_type)

  netG = ResnetGenerator(input_dim, norm_type, 6)
  netD = NLayerDiscriminator(3, norm_type, 5, False)

  netG.apply( init )
  netD.apply( init )
  return netG, netD


def obtain_stlye(configure, points, input_dim, feature_dim, norm_type):
  from .initialization import weights_init_wgan as init
  from .StyleRobust import StyleLandmarkNet
  DetNet = StyleLandmarkNet(configure, points)
  generator = ResnetGenerator(input_dim, norm_type, 6)
  netD = NLayerDiscriminator(feature_dim, norm_type, 3, False)
  generator.apply( init )
  netD.apply( init )
  return DetNet, generator, netD
