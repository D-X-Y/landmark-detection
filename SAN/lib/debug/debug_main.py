import PIL
import torch
import numpy as np
import math, os
import os.path as osp
import models
from san_vision import transforms
from utils import print_log
from visualization import save_error_image, draw_image_with_pts
from visualization import merge_images, generate_color_from_heatmaps, overlap_two_pil_image


def main_debug_save(debug_save_dir, loader, image_index, input_vars, batch_locs, target, points, sign_list, batch_cpms, generations, log):
  mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  batch_size, num_pts = batch_locs.size(0), batch_locs.size(1)
  assert batch_size == len(image_index)
  print_log('Save-dir : {} : check {}'.format(debug_save_dir, image_index), log)
  for idx, imgid in enumerate(image_index):
    basename = osp.basename( loader.dataset.datas[imgid] )
    basename = basename.split('.')[0]
    print_log('Start visualization done for [{:03d}/{:03d}] : {:5} | {:}'.format(idx, len(image_index), imgid, loader.dataset.datas[imgid]), log)
    ## Save all images
    images = [input_vars[idx], generations[0][idx], generations[1][idx]]
    _pil_images = []
    for iG, xinput in enumerate(images):
      xinput = xinput.clone()
      Xinput = []
      for t, m, s in zip(xinput, mean, std):
        t = torch.mul(t, s)
        t = torch.add(t, m)
        Xinput.append( t )
      xinput = torch.stack(Xinput)
      if xinput.is_cuda: xinput = xinput.cpu()
      image = transforms.ToPILImage()(xinput.data)
      debug_save_path = os.path.join(debug_save_dir, '{}-G{}.png'.format(basename, iG))
      image.save(debug_save_path)
      _pil_images.append( image )
      """
      debug_loc = []
      for ipts in range(num_pts):
        temploc = models.variable2np( batch_locs[idx][ipts] )
        debug_loc.append( temploc )
      debug_loc = np.array( debug_loc )
      debug_save_path = os.path.join(debug_save_dir, '{}-ans-points.png'.format(basename))
      pimage = draw_image_with_pts(image, debug_loc.transpose(1,0), radius=1, linewidth=1, fontScale=12, window=None)
      pimage.save(debug_save_path)
      """
    image = _pil_images[0]
    debug_save_path = os.path.join(debug_save_dir, '{}-GG.png'.format(basename))
    overlap_two_pil_image(_pil_images[1], _pil_images[2]).save(debug_save_path)
    # save the back ground heatmap
    for icpm in range(len(batch_cpms)):
      cpms = batch_cpms[icpm][idx]
      xtarget = models.variable2np( cpms )
      xtarget = xtarget.transpose(1,2,0)
      xheatmaps = generate_color_from_heatmaps(xtarget, index=-1)
      cimage = PIL.Image.fromarray(np.uint8(xheatmaps*255))
      cimage = overlap_two_pil_image(image, cimage)
      debug_save_path = os.path.join(debug_save_dir, '{:}-BG-{}.png'.format(basename, icpm))
      cimage.save(debug_save_path)

      xheatmaps = generate_color_from_heatmaps(xtarget, index=0)
      cimage = PIL.Image.fromarray(np.uint8(xheatmaps*255))
      cimage = overlap_two_pil_image(image, cimage)
      debug_save_path = os.path.join(debug_save_dir, '{:}-B0-{}.png'.format(basename, icpm))
      cimage.save(debug_save_path)

    # save the ground truth heatmap
    if sign_list[idx] and False:
      xtarget = models.variable2np( target[idx] )
      xtarget = xtarget.transpose(1,2,0)
      xheatmaps = generate_color_from_heatmaps(xtarget)
      all_images = []
      for pid in range(len(xheatmaps)):
        cimage = PIL.Image.fromarray(np.uint8(xheatmaps[pid]*255))
        cimage = overlap_two_pil_image(center_img, cimage)
        all_images.append( cimage )
      debug_save_path = os.path.join(debug_save_dir, '{}-gt-heatmap.png'.format(basename))
      all_images = merge_images(all_images, 10)
      all_images.save( debug_save_path )
  
      # debug save for points
      cpoints = models.variable2np( points[idx] )
      debug_save_path = os.path.join(debug_save_dir, '{}-gt-points.png'.format(basename))
      point_image = draw_image_with_pts(center_img, cpoints.transpose(1,0), radius=1, linewidth=1, fontScale=12, window=lk_window)
      point_image.save(debug_save_path)
      
      print_log('Calculate gt-visualization done for [{:03d}/{:03d}] : {:5} | {:}'.format(idx, len(image_index), imgid, loader.dataset.datas[imgid]), log)
    else:
      print_log('Skip the ground truth heatmap debug saving', log)
