# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
from PIL import Image, ImageOps
import numpy as np
import random
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap

class AugStyle(object):

  def __init__(self):
    self.contrast = iaa.ContrastNormalization((0.5, 1.5), per_channel=0.3)
    self.mulaug   = iaa.Multiply((0.8, 1.2), per_channel=0.3)
    self.grayaug  = iaa.Grayscale(alpha=(0.0, 1.0))
    self.sharpen  = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
    self.coloraug = iaa.Sequential([
                      iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                      iaa.WithChannels(0, iaa.Add((-50, 50))),
                      iaa.WithChannels(1, iaa.Add((-50, 50))),
                      iaa.WithChannels(2, iaa.Add((-50, 50))),
                      iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
                      ])

  def augment(self, pic):
    assert isinstance(pic, Image.Image)
    image = np.array(pic)
    aug = False
    if random.random() > 0.4:
      image, aug = self.contrast.augment_image(image), True
    if random.random() > 0.4:
      image, aug = self.mulaug.augment_image(image), True
    if random.random() > 0.4:
      image, aug = self.grayaug.augment_image(image), True
    if random.random() > 0.4 or aug == False:
      image, aug = self.sharpen.augment_image(image), True
    #if random.random() > 0.4 or aug == False:
    #  image, aug = self.coloraug.augment_image(image), True
    augpic = Image.fromarray(image)
    return augpic

  def __call__(self, imgs, point_meta):
    point_meta = point_meta.copy()
    if isinstance(imgs, list): is_list = True
    else:                      is_list, imgs = False, [imgs]

    augimages = [self.augment(x) for x in imgs]

    return imgs + augimages, point_meta
