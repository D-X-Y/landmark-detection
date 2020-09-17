# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Spatial Temporal Multiview Dataset
from os import path as osp
from pathlib import Path
from copy import deepcopy as copy
from tqdm import tqdm
import warnings, random, numpy as np

from pts_utils import generate_label_map
from xvision import denormalize_points
from xvision import identity2affine, solve2theta, affine2image
from .dataset_utils import pil_loader
from .point_meta_v2 import PointMeta2V
from .point_meta_v2 import apply_affine2point
from .point_meta_v2 import apply_boundary
from .optflow_utils import get_optflow_retval
from multiview import GetKRT
import torch
import torch.utils.data as data



def check_is_image(frames):
  assert len(frames) > 0, 'this is an empty frame list'
  is_image = True
  for frame in frames:
    if frame != frames[0]:
      is_image = False
  return is_image



class STMDataset(data.Dataset):

  def __init__(self, transform, sigma, downsample, heatmap_type, \
                      shape, use_gray, data_indicator, config, tensor2img):

    self.transform    = transform
    self.sigma        = sigma
    self.downsample   = downsample
    self.heatmap_type = heatmap_type
    self.dataset_name = data_indicator
    self.shape        = shape # [H,W]
    self.use_gray     = use_gray
    self.st_config    = copy( config )
    self.video_L      = config.video_L
    self.video_R      = config.video_R
    self.max_views    = config.max_views
    self.opt_backend  = config.optflow_backend
    self.optflow      = get_optflow_retval( config.optflow_backend )
    self.tensor2img   = tensor2img
    assert self.video_L >= 0 and self.video_R >= 0, 'invalid video L and video R : {:} and {:}'.format(self.video_L, self.video_R)
    assert transform is not None, 'transform : {:}'.format(transform)
    self.reset()
    print ('The video dataset initialization done : {:}'.format(self))


  def __repr__(self):
    return ('{name}(point-num={NUM_PTS}, shape.HW={shape}, length={length}, sigma={sigma}, heatmap_type={heatmap_type}, range=L.{video_L}~R.{video_R}, backend={opt_backend}, dataset={dataset_name})'.format(name=self.__class__.__name__, **self.__dict__))


  def reset(self, num_pts=-1, boxid='default', only_pts=False):
    self.NUM_PTS = num_pts
    if only_pts: return
    self.length  = 0
    self.datas   = []
    self.labels  = []
    self.camera_infos  = []
    self.NormDistances = []
    self.BOXID   = boxid
    self.all_cameras = {}
    self.multiviews  = []
    #assert self.dataset_name is not None, 'The dataset name is None'


  def __len__(self):
    assert len(self.datas) == self.length, 'The length is not correct : {}'.format(self.length)
    return self.length


  def append(self, frames, label, camera_info, distance):
    for frame in frames: assert osp.isfile(frame), 'can not find the frame path : {:}'.format(frame)
    self.datas.append( frames )
    self.labels.append( label )
    self.camera_infos.append( camera_info )
    self.NormDistances.append( distance )
    self.length = self.length + 1


  def load_list(self, file_lists, num_pts, boxindicator, normalizeL, reset):
    if reset: self.reset(num_pts, boxindicator)
    else    : assert self.NUM_PTS == num_pts and self.BOXID == boxindicator, 'The number of point is inconsistance : {:} vs {:}'.format(self.NUM_PTS, num_pts)
    if isinstance(file_lists, str): file_lists = [file_lists]
    samples = []
    for idx, file_path in enumerate(file_lists):
      data = torch.load(file_path)
      if isinstance(data, list):
        samples = samples + data
        print (':::: load list {:}/{:} : {:70s} || with {:7d} samples'.format(idx, len(file_lists), file_path, len(data)))
      elif isinstance(data, dict):
        dataset_name, tdata, tcameras = data['name'], data['datas'], data['all_cameras']
        for key, cameras in tcameras.items():
          new_key = '{:}-{:}'.format(dataset_name, key)
          assert new_key not in self.all_cameras, '{:} already in all_cameras'.format(new_key)
          self.all_cameras[ new_key ] = copy( cameras )
        for iii, temp_data in enumerate(tdata):
          temp_data['cameras_index'] = '{:}-{:}'.format(dataset_name, temp_data['cameras_index'])
          assert temp_data['cameras_index'] in self.all_cameras, '[{:03d}] {:} does not find'.format(iii, temp_data['cameras_index'])
          camera = self.all_cameras[ temp_data['cameras_index'] ][ temp_data['camera_key'] ]
        samples = samples + tdata
        print (':::: load list {:}/{:} : {:70s} || with {:} samples'.format(idx, len(file_lists), file_path, len(tdata)))
      else: raise ValueError('Invalid data type : {:}'.format( type(data) ))
    for key, value in self.all_cameras.items():
      print ('{:} multiview system has : {:} cameras'.format(key, len(value)))
    print ('Starting load {:} samples for Spatial-Temporal-Multiview Datasets'.format(len(samples)))

    # get the forward-backward frames
    Fprevious, Fnext = {}, {}
    for index, annotation in tqdm( enumerate(samples) ):
      ppath, xpath, npath = annotation['previous_frame'], annotation['current_frame'], annotation['next_frame']
      if xpath in Fprevious and Fprevious[xpath] is not None and ppath is not None:
        assert Fprevious[xpath] == ppath, '{:} :: {:} vs. {:}'.format(index, Fprevious[xpath], ppath)
      else: Fprevious[xpath] = ppath
      if xpath in Fnext and Fnext[xpath] is not None and npath is not None:
        assert Fnext[xpath] == npath, '{:} :: {:} vs. {:}'.format(index, Fnext[xpath], npath)
      else: Fnext[xpath] = npath

    for index in tqdm( range(len(samples)) ):
      annotation  = samples[index]
      image_path  = annotation['current_frame']
      points, box = annotation['points'], annotation['box-{:}'.format(boxindicator)]
      label = PointMeta2V(self.NUM_PTS, points, box, image_path, self.dataset_name)
      if normalizeL is None: normDistance = None
      else                 : normDistance = annotation['normalizeL-{:}'.format(normalizeL)]
      if (annotation['previous_frame'] is None) and (annotation['next_frame'] is None) and (annotation['points'] is None) and ('cameras_index' not in annotation):
        continue # useless data in our framework
      frames = [None] * self.video_L + [image_path] + [None] * self.video_R
      temp = Fprevious[image_path]
      for i in range(self.video_L):
        if temp is None: frames[self.video_L-i-1] = frames[self.video_L-i]
        else:
          frames[self.video_L-i-1] = temp
          if temp in Fprevious: temp = Fprevious[temp]
          else                : temp = None
      temp = Fnext[image_path]
      for i in range(self.video_R):
        if temp is None: frames[self.video_L+i+1] = frames[self.video_L+i]
        else:
          frames[self.video_L+i+1] = temp
          if temp in Fnext: temp = Fnext[temp]
          else            : temp = None
      if 'cameras_index' in annotation:
        camera_info = {'cameras_index': annotation['cameras_index'],
                       'camera_key'   : annotation['camera_key']}
      else: camera_info = None
      self.append(frames, label, camera_info, normDistance)

    assert len(self.datas) == self.length, 'The length and the data is not right {:} vs {:}'.format(self.length, len(self.datas))
    assert len(self.labels) == self.length, 'The length and the labels is not right {:} vs {:}'.format(self.length, len(self.labels))
    assert len(self.camera_infos) == self.length, 'The length and the camera_info is not right {:} vs {:}'.format(self.length, len(self.camera_info))
    assert len(self.NormDistances) == self.length, 'The length and the NormDistances is not right {:} vs {:}'.format(self.length, len(self.NormDistance))
    print ('Load data done for STDataset, which has {:} images.'.format(self.length))


  def check_is_image(self, index):
    if index < 0: index = self.length + index
    assert index >= 0 and index < self.length, 'Invalid index : {:}'.format(index)
    return check_is_image( self.datas[index] )

  def prepare_multiview(self):
    for key, cameras in self.all_cameras.items():
      for ckey, camera in cameras.items():
        KRT = GetKRT(camera['K'], camera['R'], camera['t'])
        camera['KRT'] = KRT
    self.multiviews = []
    frame2labels = {}
    for index in tqdm( range( len(self.datas) ) ):
      label = self.labels[index].copy()
      frame2labels[ self.datas[index][self.video_L] ] = label
    print ('prepare_multiview calculate KRT done, starting collect data')
    bad_num = 0
    for index in tqdm( range( len(self.datas) ) ):
      caminfo = self.camera_infos[index]
      if caminfo is None:
        multiviews = None
      else:
        xcameras = self.all_cameras[ caminfo['cameras_index'] ]
        others   = set(xcameras.keys())
        others.remove( caminfo['camera_key'] )
        cameras = [xcameras[ caminfo['camera_key'] ]] + [ xcameras[x] for x in others ]
        frame_name = Path(self.datas[index][self.video_L]).name
        multiviews = []
        for camera in cameras:
          if 'dataset' in camera and camera['dataset'] == 'Mugsy': # Mugsy
            xfname = str(Path(camera['frame_path']).parent / frame_name)
          else:                                                    # PanopticStudio
            xfname = osp.join(camera['frame_path'], camera['name'] + frame_name[5:])
          if osp.isfile(xfname) and xfname in frame2labels:
            KRT = GetKRT(camera['K'], camera['R'], camera['t'])
            multiviews.append( (xfname, frame2labels[xfname], KRT) )
          elif osp.isfile(xfname) and ('dataset' in camera and camera['dataset'] == 'Mugsy'): # Special for Mugsy
            KRT = GetKRT(camera['K'], camera['R'], camera['t'])
            multiviews.append( (xfname, PointMeta2V(self.NUM_PTS, None, None, xfname, self.dataset_name), KRT) )
        if len(multiviews) < 4:
          print( 'Index : {:} can not find at least 4 views instead of {:} views'.format(index, len(multiviews)) )
          multiviews, bad_num = None, bad_num + 1
      self.multiviews.append( multiviews )
    print ('Filter {:} less-view images from {:}'.format(bad_num, len(self.datas)))

  def __getitem__(self, index):
    assert index >= 0 and index < self.length, 'Invalid index : {:}'.format(index)
    frame_paths = self.datas[index]
    frames      = [pil_loader(f_path, self.use_gray) for f_path in frame_paths]
    target      = self.labels[index].copy()
    multiviews  = self.multiviews[index]
    if multiviews is None:
      view_info = None
      mulview_str = [''] * self.max_views
    else                 :
      view_info = [multiviews[0]]
      view_info += random.sample(multiviews[1:], self.max_views-1)
      mulview_str = [xpath for xpath, label, KRT in view_info]
      view_info = [ (pil_loader(xpath, self.use_gray), label.copy(), KRT.clone()) for xpath, label, KRT in view_info]

    skipopt = check_is_image(frame_paths)

    torch_is_image = torch.ByteTensor( [check_is_image(frame_paths)] )
    affineFrames, forward_flow, backward_flow, heatmaps, mask, norm_trans_points, THETA, \
          MultiViewTensor, MultiViewThetas, MultiViewShapes, MultiViewKRT, torch_is_3D, \
          torch_index, torch_nopoints, torch_shape = self._process_(frames, target, view_info, index, skipopt)
    return affineFrames, forward_flow, backward_flow, heatmaps, mask, norm_trans_points, THETA, \
           MultiViewTensor, MultiViewThetas, MultiViewShapes, MultiViewKRT, torch_is_3D, torch_is_image , \
           torch_index, torch_nopoints, torch_shape, mulview_str


  def _process_(self, frames, target, view_info, index, skipopt):

    # transform the image and points
    frames, target, theta = self.transform(frames, target)
    (C, H, W), (height, width) = frames[0].size(), self.shape

    # obtain the visiable indicator vector
    if target.is_none(): nopoints = True
    else               : nopoints = False

    affineFrames, forward_flow, backward_flow, heatmaps, mask, norm_trans_points, THETA = self.__process_affine(frames, target, theta, nopoints, skipopt)

    torch_index    = torch.IntTensor([index])
    torch_nopoints = torch.ByteTensor( [ nopoints ] )
    torch_shape    = torch.IntTensor([H,W])
    torch_is_3D    = torch.ByteTensor( [ view_info is not None ] )
    if view_info is None:
      MultiViewKRT    = torch.zeros((self.max_views, 3, 4))
      MultiViewTensor = torch.zeros((self.max_views, 1 if self.use_gray else 3, self.shape[0], self.shape[1]))
      MultiViewShapes = torch.zeros((self.max_views, 2), dtype=torch.float32)
      MultiViewThetas = torch.zeros((self.max_views, 3, 3))
    else:
      MultiViewIs, MultiViewLs, MultiViewKRT = [], [], []
      for Mimage, Mlabel, MKRT in view_info:
        MultiViewIs.append(Mimage)
        MultiViewLs.append(Mlabel)
        MultiViewKRT.append(MKRT)
      MultiViewIs, MultiViewLs, MultiViewTheta = self.transform(MultiViewIs, MultiViewLs)
      MultiViewTensor = []
      for ximage, xtheta in zip(MultiViewIs, MultiViewTheta):
        xTensor = affine2image(ximage, xtheta, self.shape)
        MultiViewTensor.append( xTensor )
      MultiViewKRT    = torch.stack( MultiViewKRT )
      MultiViewTensor = torch.stack(MultiViewTensor)
      MultiViewShapes = torch.FloatTensor( [(image.size(-1), image.size(-2)) for image in MultiViewIs] ) # shape : W, H
      MultiViewThetas = torch.stack( MultiViewTheta )

    return affineFrames, forward_flow, backward_flow, heatmaps, mask, norm_trans_points, THETA, MultiViewTensor, MultiViewThetas, MultiViewShapes, MultiViewKRT, torch_is_3D, torch_index, torch_nopoints, torch_shape


  def __process_affine(self, frames, target, theta, nopoints, skipopt, aux_info=None):
    frames, target, theta = [frame.clone() for frame in frames], target.copy(), theta.clone()
    (C, H, W), (height, width) = frames[0].size(), self.shape
    if nopoints: # do not have label
      norm_trans_points = torch.zeros((3, self.NUM_PTS))
      heatmaps          = torch.zeros((self.NUM_PTS+1, height//self.downsample, width//self.downsample))
      mask              = torch.ones((self.NUM_PTS+1, 1, 1), dtype=torch.uint8)
    else:
      norm_trans_points = apply_affine2point(target.get_points(), theta, (H,W))
      norm_trans_points = apply_boundary(norm_trans_points)
      real_trans_points = norm_trans_points.clone()
      real_trans_points[:2, :] = denormalize_points(self.shape, real_trans_points[:2,:])
      heatmaps, mask = generate_label_map(real_trans_points.numpy(), height//self.downsample, width//self.downsample, self.sigma, self.downsample, nopoints, self.heatmap_type) # H*W*C
      heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1))).type(torch.FloatTensor)
      mask     = torch.from_numpy(mask.transpose((2, 0, 1))).type(torch.ByteTensor)

    affineFrames = [affine2image(frame, theta, self.shape) for frame in frames]

    if not skipopt:
      Gframes = [self.tensor2img(frame) for frame in affineFrames]
      forward_flow, backward_flow = [], []
      for idx in range( len(Gframes) ):
        if idx > 0:
          forward_flow.append( self.optflow.calc(Gframes[idx-1], Gframes[idx], None) )
        if idx+1 < len(Gframes):
          #backward_flow.append( self.optflow.calc(Gframes[idx], Gframes[idx+1], None) ) ## HDXY
          backward_flow.append( self.optflow.calc(Gframes[idx+1], Gframes[idx], None) )
      forward_flow  = torch.stack( [torch.from_numpy(x) for x in forward_flow] )
      backward_flow = torch.stack( [torch.from_numpy(x) for x in backward_flow] )
    else:
      forward_flow, backward_flow = torch.zeros((len(affineFrames)-1, height, width, 2)), torch.zeros((len(affineFrames)-1, height, width, 2))
    # affineFrames  #frames x #channel x #height x #width
    # forward_flow  (#frames-1) x #height x #width x 2
    # backward_flow (#frames-1) x #height x #width x 2
    return torch.stack(affineFrames), forward_flow, backward_flow, heatmaps, mask, norm_trans_points, theta


class StmBatchSampler(object):

  def __init__(self, dataset, ibatch, vbatch, mbatch):
    '''
    Args:
    - dataset: an instance of the VideoDatasetV2 class
    - ibatch: the batch size of images for one iteration
    - vbatch: the batch size of videos for one iteration
    - mbatch: the batch size of multiview for one iteration
    '''
    super(StmBatchSampler, self).__init__()
    self.length = len(dataset)
    self.IMG_indexes = []
    self.VID_indexes = []
    self.MID_indexes = []
    for i in tqdm(range(len(dataset))):
      if dataset.labels[i].is_none() == False and dataset.check_is_image( i ):
        self.IMG_indexes.append( i )
      if dataset.check_is_image( i ) == False:
        self.VID_indexes.append( i )
      if dataset.multiviews[i] is not None:
        self.MID_indexes.append( i )
    self.IMG_batch = ibatch
    self.VID_batch = vbatch
    self.MID_batch = mbatch
    #assert self.IMG_batch > 0 and self.MID_batch > 0, 'image batch size must be > 0, {:} and {:}'.format(self.IMG_batch, self.MID_batch)
    assert self.MID_batch >= 0, 'image batch size must be > 0, {:} and {:}'.format(self.IMG_batch, self.MID_batch)
    assert len(self.IMG_indexes) >= self.IMG_batch, '{:} vs {:}'.format(len(self.IMG_indexes), self.IMG_batch)
    assert len(self.VID_indexes) >= self.VID_batch and len(self.MID_indexes) >= self.MID_batch, '{:} vs {:} and {:} vs {:}'.format(len(self.VID_indexes), self.VID_batch, len(self.MID_indexes), self.MID_batch)
    if self.IMG_batch == 0: self.iters = len(self.VID_indexes) // self.VID_batch + 1
    else                  : self.iters = len(self.IMG_indexes) // self.IMG_batch + 1
    print ('In STM-Batch-Sampler, sample {:} annotated images, {:} video frames, {:} multiview-images from {:} data.'.format(len(self.IMG_indexes), len(self.VID_indexes), len(self.MID_indexes), len(dataset)))


  def __iter__(self):
    # yield a batch of indexes

    for index in range(self.iters):
      if self.IMG_batch == 0: images = []
      else                  : images = random.sample(self.IMG_indexes, self.IMG_batch)
      if self.VID_batch == 0: videos = []
      else                  : videos = random.sample(self.VID_indexes, self.VID_batch)
      mviews = random.sample(self.MID_indexes, self.MID_batch)
      batch = torch.LongTensor(images + videos + mviews)
      yield batch

  def __len__(self):
    # returns the number of iterations (episodes) per epoch
    return self.iters
