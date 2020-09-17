# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from os import path as osp
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
import torch
import torch.utils.data as data



def check_is_image(frames):
  assert len(frames) > 0, 'this is an empty frame list'
  is_image = True
  for frame in frames:
    if frame != frames[0]:
      is_image = False
  return is_image



class VideoDatasetV2(data.Dataset):

  def __init__(self, transform, sigma, downsample, heatmap_type, \
                      shape, use_gray, mean_file, data_indicator, config, tensor2img):

    self.transform    = transform
    self.sigma        = sigma
    self.downsample   = downsample
    self.heatmap_type = heatmap_type
    self.dataset_name = data_indicator
    self.shape        = shape # [H,W]
    self.use_gray     = use_gray
    self.video_config = copy( config )
    self.video_L      = config.video_L
    self.video_R      = config.video_R
    self.opt_backend  = config.optflow_backend
    self.optflow      = get_optflow_retval( config.optflow_backend )
    self.tensor2img   = tensor2img
    assert self.video_L >= 0 and self.video_R >=0, 'invalid video L and video R : {:} and {:}'.format(self.video_L, self.video_R)
    assert transform is not None, 'transform : {:}'.format(transform)
    if mean_file is None:
      self.mean_data  = None
      warnings.warn('VideolDatasetV2 initialized with mean_data = None')
    else:
      assert osp.isfile(mean_file), '{:} is not a file.'.format(mean_file)
      self.mean_data    = torch.load(mean_file)
    self.reset()
    print ('The video dataset initialization done : {:}'.format(self))


  def __repr__(self):
    return ('{name}(point-num={NUM_PTS}, shape={shape}, length={length}, sigma={sigma}, heatmap_type={heatmap_type}, range=L.{video_L}~R.{video_R}, backend={opt_backend}, dataset={dataset_name})'.format(name=self.__class__.__name__, **self.__dict__))


  def reset(self, num_pts=-1, boxid='default', only_pts=False):
    self.NUM_PTS = num_pts
    if only_pts: return
    self.length = 0
    self.datas = []
    self.labels = []
    self.NormDistances = []
    self.BOXID = boxid
    if self.mean_data is None:
      self.mean_face = None
    else:
      self.mean_face = torch.Tensor(self.mean_data[boxid].copy().T)
      assert (self.mean_face >= -1).all() and (self.mean_face <= 1).all(), 'mean-{:}-face : {:}'.format(boxid, self.mean_face)
    self.cache_file2index_DXY = {}
    #assert self.dataset_name is not None, 'The dataset name is None'


  def __len__(self):
    assert len(self.datas) == self.length, 'The length is not correct : {}'.format(self.length)
    return self.length


  def append(self, frames, label, distance):
    for frame in frames: assert osp.isfile(frame), 'can not find the frame path : {:}'.format(frame)
    self.datas.append( frames )           ;  self.labels.append( label )
    self.NormDistances.append( distance )
    self.length = self.length + 1
    self.cache_file2index_DXY[ frames[self.video_L] ] = len(self.datas) - 1


  def load_list(self, file_lists, num_pts, boxindicator, normalizeL, reset):
    if reset: self.reset(num_pts, boxindicator)
    else    : assert self.NUM_PTS == num_pts and self.BOXID == boxindicator, 'The number of point is inconsistance : {:} vs {:}'.format(self.NUM_PTS, num_pts)
    if isinstance(file_lists, str): file_lists = [file_lists]
    samples = []
    for idx, file_path in enumerate(file_lists):
      #print (':::: load list {:}/{:} : {:}'.format(idx, len(file_lists), file_path))
      xdata = torch.load(file_path)
      if isinstance(xdata, list)  : data = xdata          # image or video dataset list
      elif isinstance(xdata, dict): data = xdata['datas'] # multi-view dataset list
      else: raise ValueError('Invalid Type Error : {:}'.format( type(xdata) ))
      samples = samples + data
      print (':::: load list {:}/{:} : {:70s} || with {:} samples'.format(idx, len(file_lists), file_path, len(data)))
    # samples is a list, where each element is the annotation. Each annotation is a dict, contains 'points' (3,num_pts), and various box
    print ('Starting load {:} samples for VideoDataset-V2'.format(len(samples)))
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

    #for index, annotation in tqdm( enumerate(samples) ):
    for index in tqdm( range(len(samples)) ):
      annotation  = samples[index]
      image_path  = annotation['current_frame']
      points, box = annotation['points'], annotation['box-{:}'.format(boxindicator)]
      label = PointMeta2V(self.NUM_PTS, points, box, image_path, self.dataset_name)
      if normalizeL is None: normDistance = None
      else                 : normDistance = annotation['normalizeL-{:}'.format(normalizeL)]
      if annotation['previous_frame'] is None and annotation['next_frame'] is None and annotation['points'] is None:
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
      self.append(frames, label, normDistance)

    assert len(self.datas) == self.length, 'The length and the data is not right {:} vs {:}'.format(self.length, len(self.datas))
    assert len(self.labels) == self.length, 'The length and the labels is not right {:} vs {:}'.format(self.length, len(self.labels))
    assert len(self.NormDistances) == self.length, 'The length and the NormDistances is not right {:} vs {:}'.format(self.length, len(self.NormDistance))
    print ('Load data done for VideoDatasetV2, which has {:} images.'.format(self.length))


  def check_is_image(self, index):
    if index < 0: index = self.length + index
    assert index >= 0 and index < self.length, 'Invalid index : {:}'.format(index)
    return check_is_image( self.datas[index] )


  def __getitem__(self, index):
    assert index >= 0 and index < self.length, 'Invalid index : {:}'.format(index)
    frame_paths = self.datas[index]
    frames = [pil_loader(f_path, self.use_gray) for f_path in frame_paths]
    target = self.labels[index].copy()

    torch_is_image = torch.ByteTensor( [check_is_image(frame_paths)] )
    affineFrames, forward_flow, backward_flow, heatmaps, mask, norm_trans_points, THETA, transpose_theta, torch_index, torch_nopoints, torch_shape = self._process_(frames, target, index, check_is_image(frame_paths))
    return affineFrames, forward_flow, backward_flow, heatmaps, mask, norm_trans_points, THETA, transpose_theta, torch_index, torch_nopoints, torch_shape, torch_is_image

  def find_index(self, xpath):
    assert xpath in self.cache_file2index_DXY, 'Can not find this path : {:}'.format(xpath)
    index = self.cache_file2index_DXY[ xpath ]
    points = self.labels[ index ].get_points()
    return points

  def _process_(self, pil_frames, target, index, skip_opt):

    # transform the image and points
    frames, target, theta = self.transform(pil_frames, target)
    (C, H, W), (height, width) = frames[0].size(), self.shape

    # obtain the visiable indicator vector
    if target.is_none(): nopoints = True
    else               : nopoints = False

    if isinstance(theta, list) or isinstance(theta, tuple):
      affineFrames, forward_flow, backward_flow, heatmaps, mask, norm_trans_points, THETA, transpose_theta = [], [], [], [], [], []
      for _theta in theta:
        _affineFrames, _forward_flow, _backward_flow, _heatmaps, _mask, _norm_trans_points, _theta, _transpose_theta \
          = self.__process_affine(frames, target, _theta, nopoints, skip_opt)
        affineFrames.append(_affineFrames)
        forward_flow.append(_forward_flow)
        backward_flow.append(_backward_flow)
        heatmaps.append(_heatmaps)
        mask.append(_mask)
        norm_trans_points.append(_norm_trans_points)
        THETA.append(_theta)
        transpose_theta.append(_transpose_theta)
      affineFrames, forward_flow, backward_flow, heatmaps, mask, norm_trans_points, THETA, transpose_theta = \
          torch.stack(affineFrames), torch.stack(forward_flow), torch.stack(backward_flow), torch.stack(heatmaps), torch.stack(mask), torch.stack(norm_trans_points), torch.stack(THETA), torch.stack(transpose_theta)
    else:
      affineFrames, forward_flow, backward_flow, heatmaps, mask, norm_trans_points, THETA, transpose_theta = self.__process_affine(frames, target, theta, nopoints, skip_opt)

    torch_index = torch.IntTensor([index])
    torch_nopoints = torch.ByteTensor( [ nopoints ] )
    torch_shape = torch.IntTensor([H,W])

    return affineFrames, forward_flow, backward_flow, heatmaps, mask, norm_trans_points, THETA, transpose_theta, torch_index, torch_nopoints, torch_shape


  def __process_affine(self, frames, target, theta, nopoints, skip_opt, aux_info=None):
    frames, target, theta = [frame.clone() for frame in frames], target.copy(), theta.clone()
    (C, H, W), (height, width) = frames[0].size(), self.shape
    if nopoints: # do not have label
      norm_trans_points = torch.zeros((3, self.NUM_PTS))
      heatmaps          = torch.zeros((self.NUM_PTS+1, height//self.downsample, width//self.downsample))
      mask              = torch.ones((self.NUM_PTS+1, 1, 1), dtype=torch.uint8)
      transpose_theta   = identity2affine(False)
    else:
      norm_trans_points = apply_affine2point(target.get_points(), theta, (H,W))
      norm_trans_points = apply_boundary(norm_trans_points)
      real_trans_points = norm_trans_points.clone()
      real_trans_points[:2, :] = denormalize_points(self.shape, real_trans_points[:2,:])
      heatmaps, mask = generate_label_map(real_trans_points.numpy(), height//self.downsample, width//self.downsample, self.sigma, self.downsample, nopoints, self.heatmap_type) # H*W*C
      heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1))).type(torch.FloatTensor)
      mask     = torch.from_numpy(mask.transpose((2, 0, 1))).type(torch.ByteTensor)
      if torch.sum(norm_trans_points[2,:] == 1) < 3 or self.mean_face is None:
        warnings.warn('In GeneralDatasetV2 after transformation, no visiable point, using identity instead. Aux: {:}'.format(aux_info))
        transpose_theta = identity2affine(False)
      else:
        transpose_theta = solve2theta(norm_trans_points, self.mean_face.clone())

    affineFrames = [affine2image(frame, theta, self.shape) for frame in frames]

    if not skip_opt:
      Gframes = [self.tensor2img(frame) for frame in affineFrames]
      forward_flow, backward_flow = [], []
      for idx in range( len(Gframes) ):
        if idx > 0:
          forward_flow.append( self.optflow.calc(Gframes[idx-1], Gframes[idx], None) )
        if idx+1 < len(Gframes):
          #backward_flow.append( self.optflow.calc(Gframes[idx], Gframes[idx+1], None) )
          backward_flow.append( self.optflow.calc(Gframes[idx+1], Gframes[idx], None) )
      forward_flow  = torch.stack( [torch.from_numpy(x) for x in forward_flow] )
      backward_flow = torch.stack( [torch.from_numpy(x) for x in backward_flow] )
    else:
      forward_flow, backward_flow = torch.zeros((len(affineFrames)-1, height, width, 2)), torch.zeros((len(affineFrames)-1, height, width, 2))
    # affineFrames  #frames x #channel x #height x #width
    # forward_flow  (#frames-1) x #height x #width x 2
    # backward_flow (#frames-1) x #height x #width x 2
    return torch.stack(affineFrames), forward_flow, backward_flow, heatmaps, mask, norm_trans_points, theta, transpose_theta



class SbrBatchSampler(object):

  def __init__(self, dataset, ibatch, vbatch, sbr_sampler_use_vid):
    '''
    Args:
    - dataset: an instance of the VideoDatasetV2 class
    - ibatch: the batch size of images for one iteration
    - vbatch: the batch size of videos for one iteration
    '''
    super(SbrBatchSampler, self).__init__()
    self.length = len(dataset)
    self.IMG_indexes = []
    self.VID_indexes = []
    for i in range(len(dataset)):
      if dataset.labels[i].is_none() == False and (sbr_sampler_use_vid or dataset.check_is_image( i )):
        self.IMG_indexes.append( i )
      if dataset.check_is_image( i ) == False:
        self.VID_indexes.append( i )
    self.IMG_batch = ibatch
    self.VID_batch = vbatch
    if self.IMG_batch == 0: self.iters = len(self.VID_indexes) // self.VID_batch + 1
    else                  : self.iters = len(self.IMG_indexes) // self.IMG_batch + 1
    #assert self.IMG_batch > 0, 'image batch size must be greater than 0'
    assert len(self.IMG_indexes) >= self.IMG_batch, '{:} vs {:}'.format(len(self.IMG_indexes), self.IMG_batch)
    assert len(self.VID_indexes) >= self.VID_batch, '{:} vs {:}'.format(len(self.VID_indexes), self.VID_batch)
    print ('In SbrBatchSampler, sample {:} images and {:} videos from {:} datas'.format(len(self.IMG_indexes), len(self.VID_indexes), len(dataset)))


  def __iter__(self):
    # yield a batch of indexes

    for index in range(self.iters):
      if self.IMG_batch == 0: images = []
      else                  : images = random.sample(self.IMG_indexes, self.IMG_batch)
      if self.VID_batch == 0: videos = []
      else                  : videos = random.sample(self.VID_indexes, self.VID_batch)
      batchlist = images + videos
      assert len(batchlist) > 0, 'invalid batchlist : {:}'.format(batchlist)
      batch = torch.LongTensor(batchlist)
      yield batch

  def __len__(self):
    # returns the number of iterations (episodes) per epoch
    return self.iters
