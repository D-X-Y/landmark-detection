# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time, os
import numpy as np
import torch
from copy import deepcopy
from log_utils import AverageMeter, time_string, convert_secs2time
from .style_utils import generate_noise

# train function (forward, backward, update)
def style_eval_plain(args, loader, netG, recover, save_path, logger):
  args = deepcopy(args)
  data_time, upG_time = AverageMeter(), AverageMeter()

  if not save_path.exists(): os.makedirs(str(save_path))
  logger.log('[style-evaluation] save into {:}'.format(save_path))

  #netG.eval()
  end = time.time()
  with torch.no_grad():
    for i, ((PlainImages, StyleImages), target, mask, points, image_index, nopoints, cropped_size) in enumerate(loader):
      # inputs : Batch, Channel, Height, Width
      data_time.update(time.time() - end)

      # Fake Images
      Isize = list(PlainImages.size())
      noise_inputs = generate_noise(PlainImages)
      fake_images  = netG(noise_inputs)

      for j, fake_image in enumerate(fake_images):
        index = image_index[j].item()
        name  = loader.dataset.datas[index].split('/')[-1]
        spath = save_path / name
        image = recover( fake_image )
        image.save(str(spath))

        if args.debug:
          plain_image = PlainImages[j]
          spath = save_path / ('plain-'+name)
          image = recover( plain_image )
          image.save(str(spath))
          style_image = StyleImages[j]
          spath = save_path / ('style-'+name)
          image = recover( style_image )
          image.save(str(spath))

      # measure elapsed time
      upG_time.update(time.time() - end)

    if i % (args.print_freq*3) == args.print_freq or i+1 == len(loader):
      logger.log(time_string() + ' [Eval-GAN]: [{:03d}/{:03d}] '
                'updateG {upG_time.val:4.2f} ({upG_time.avg:4.2f}) '
                'Data {data_time.val:4.2f} ({data_time.avg:4.2f}) '.format(
                    i, len(loader), upG_time=upG_time,
                    data_time=data_time) + ' I={:}'.format(Isize))
