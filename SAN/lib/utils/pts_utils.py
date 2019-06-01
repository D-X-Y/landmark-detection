##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
from scipy.ndimage.interpolation import zoom
import numbers, math
import numpy as np

def pts2bbox(pts):
  '''
  convert a set of 2d points to a bounding box
  parameter:
    pts : 2 x N numpy array, N should >= 2
    or  : 3 x N numpy array, N should >= 2, the last row means visiable or not
  return:
    bbox: 1 x 4 numpy array, TLBR format
  '''

  assert isinstance(pts, np.ndarray) and (pts.shape[0] == 2 or pts.shape[0] == 3), 'the input points should have shape 2 x num_pts'
  if pts.shape[0] == 3:
    pts = pts[0:1, pts[2,:] == 1]
  assert pts.shape[1] >= 2 and len(pts.shape) == 2, 'number of points should be larger or equal than 2, and the dimension of points tensor can only be 2'
  bbox = np.zeros((1, 4), dtype='float32')
  bbox[0, 0] = np.min(pts[0, :])          # x coordinate of left top point
  bbox[0, 1] = np.min(pts[1, :])          # y coordinate of left top point
  bbox[0, 2] = np.max(pts[0, :])          # x coordinate of bottom right point
  bbox[0, 3] = np.max(pts[1, :])          # y coordinate of bottom right point
  return bbox

def find_peaks_v1(heatmap):
  assert isinstance(heatmap, np.ndarray) and len(heatmap.shape) == 2
  #heatmap = heatmap * ( heatmap > threshold )
  index = np.unravel_index(heatmap.argmax(), heatmap.shape)
  return index, heatmap[index]

def find_peaks_v2(heatmap, threshold=0.000001, eps=np.finfo(float).eps):
  assert isinstance(heatmap, np.ndarray) and len(heatmap.shape) == 2  
  #heatmap = heatmap * ( heatmap > threshold )    
  if heatmap.max() < eps:
    return find_peaks_v1(heatmap)

  heatmap[ heatmap <= threshold ] = eps
  w, h = heatmap.shape
  x = np.sum(np.sum(heatmap, axis=1) * np.arange(0, w)) / heatmap.sum()    
  y = np.sum(np.sum(heatmap, axis=0) * np.arange(0, h)) / heatmap.sum()    
  x2, y2 = min(w-1, int(np.ceil(x))), min(h-1, int(np.ceil(y)))  
  x1, y1 = max(0, x2-1), max(0, y2-1)   
  ## Bilinear interpolation   
  if x1 == x2: 
    R1, R2 = heatmap[x1, y1], heatmap[x1, y2]
  else:   
    R1 = (x2-x)/(x2-x1)*heatmap[x1, y1] + (x-x1)/(x2-x1)*heatmap[x2, y1]   
    R2 = (x2-x)/(x2-x1)*heatmap[x1, y2] + (x-x1)/(x2-x1)*heatmap[x2, y2]   
  #print ('{}, {}'.format(heatmap[x1, y1], heatmap[x2, y2]))
  #print ('x1 : {}, y2 : {},  R1 : {}, R2 : {}'.format(x1, y1, R1, R2))
  if y1 == y2: 
    score = R1 
  else:   
    score = (y2-y)/(y2-y1)*R1 + (y-y1)/(y2-y1)*R2 
  return (x,y), score 

def find_batch_peaks(heatmap, radius, downsample):
  assert isinstance(heatmap, np.ndarray) and len(heatmap.shape) == 4, 'heatmap shape : {}'.format(heatmap.shape)
  batch, num_pts, h, w = heatmap.shape
  pts_locations = np.zeros( (batch, num_pts, 3), dtype='float32')
  # batch x [x, y, score]

  for bth in range(batch):
    for pts_index in range(num_pts):
      location, score = find_peaks_v1(heatmap[bth,pts_index,:,:])
      sh, sw = location[0] - radius,     location[1] - radius
      eh, ew = location[0] + radius + 1, location[1] + radius + 1
      sw, sh = max(0, sw), max(0, sh)
      ew, eh = min(w, ew), min(h, eh)
      #temp = zoom(heatmap[bth, pts_index, sh:eh, sw:ew], downsample, order=3)
      #loc, score = find_peaks_v2(temp)
      loc, score = find_peaks_v2(heatmap[bth, pts_index, sh:eh, sw:ew])
      pts_locations[bth, pts_index, 2] = score
      pts_locations[bth, pts_index, 1] = sh * downsample + loc[0] * downsample + downsample / 2.0 - 0.5
      pts_locations[bth, pts_index, 0] = sw * downsample + loc[1] * downsample + downsample / 2.0 - 0.5
  return pts_locations

def find_all_peaks(heatmap, radius, downsample, threshold, image_resize):
  assert isinstance(heatmap, np.ndarray), 'heatmap type : {}'.format(heatmap.shape)
  assert len(heatmap.shape) == 3, 'heatmap shape : {}'.format(heatmap.shape)
  assert threshold is None or isinstance(threshold, numbers.Number), 'threshold must greater than 0'
  assert image_resize is None or (isinstance(image_resize, numbers.Number) and image_resize > 0)
  w, h, num_pts = heatmap.shape
  pts_locations = np.zeros( (3, num_pts), dtype='float32')

  for pts_index in range(num_pts):
    location, score = find_peaks_v1(heatmap[:,:,pts_index])
    sw, sh = int(location[0] - radius),     int(location[1] - radius)
    ew, eh = int(location[0] + radius + 1), int(location[1] + radius + 1)
    sw, sh = max(0, sw), max(0, sh)
    ew, eh = min(w, ew), min(h, eh)
    #temp = zoom(heatmap[sw:ew, sh:eh, pts_index], downsample, order=3)
    #loc, score = find_peaks(temp)
    loc, score = find_peaks_v2(heatmap[sw:ew, sh:eh, pts_index])
    #if pts_index == 0:
    #  print ('location : {} , radius : {}'.format(location, radius))
    #  print ('({}, {}) - ({}, {})'.format(sw,ew,sh,eh))
    #  print ('loc : {}'.format(loc))
    if threshold is not None and (score < threshold):
      pts_locations[2, pts_index] = False
    else:
      pts_locations[2, pts_index] = score
      pts_locations[0, pts_index] = sh * downsample + loc[1] * downsample + downsample / 2.0 - 0.5
      pts_locations[1, pts_index] = sw * downsample + loc[0] * downsample + downsample / 2.0 - 0.5

      if image_resize is not None:
        pts_locations[0, pts_index] = pts_locations[0, pts_index] / image_resize
        pts_locations[1, pts_index] = pts_locations[1, pts_index] / image_resize

  return pts_locations

## pts = 3 * N numpy array; points location is based on the image with size (height*downsample, width*downsample)
## 

def generate_label_map_laplacian(pts, height, width, sigma, downsample, visiable=None):
  if isinstance(pts, numbers.Number):
    # this image does not provide the annotation, pts is a int number representing the number of points
    return np.zeros((height,width,pts+1), dtype='float32'), np.ones((1,1,1+pts), dtype='float32')

  assert isinstance(pts, np.ndarray) and len(pts.shape) == 2 and pts.shape[0] == 3, 'The shape of points : {}'.format(pts.shape)
  if isinstance(sigma, numbers.Number):
    sigma = np.zeros((pts.shape[1])) + sigma
  assert isinstance(sigma, np.ndarray) and len(sigma.shape) == 1 and sigma.shape[0] == pts.shape[1], 'The shape of sigma : {}'.format(sigma.shape)

  offset = downsample / 2.0 - 0.5
  num_points, threshold = pts.shape[1], 0.01

  if visiable is None: visiable = pts[2, :].astype('bool')
  assert visiable.shape[0] == num_points

  transformed_label = np.fromfunction( lambda y, x, pid : ((offset + x*downsample - pts[0,pid])**2 \
                                                        + (offset + y*downsample - pts[1,pid])**2) \
                                                          / -2.0 / sigma[pid] / sigma[pid],
                                                          (height, width, num_points), dtype=int)

  mask_heatmap      = np.ones((1, 1, num_points+1), dtype='float32')
  mask_heatmap[0, 0, :num_points] = visiable
  mask_heatmap[0, 0, num_points]  = 1
  
  transformed_label = (1+transformed_label) * np.exp(transformed_label)
  transformed_label[ transformed_label < threshold ] = 0
  transformed_label[ transformed_label >         1 ] = 1

  background_label  = 1 - np.amax(transformed_label, axis=2)
  background_label[ background_label < 0 ] = 0
  heatmap           = np.concatenate((transformed_label, np.expand_dims(background_label, axis=2)), axis=2).astype('float32')
  
  return heatmap*mask_heatmap, mask_heatmap

def generate_label_map_gaussian(pts, height, width, sigma, downsample, visiable=None):
  if isinstance(pts, numbers.Number):
    # this image does not provide the annotation, pts is a int number representing the number of points
    return np.zeros((height,width,pts+1), dtype='float32'), np.ones((1,1,1+pts), dtype='float32')

  assert isinstance(pts, np.ndarray) and len(pts.shape) == 2 and pts.shape[0] == 3, 'The shape of points : {}'.format(pts.shape)
  if isinstance(sigma, numbers.Number):
    sigma = np.zeros((pts.shape[1])) + sigma
  assert isinstance(sigma, np.ndarray) and len(sigma.shape) == 1 and sigma.shape[0] == pts.shape[1], 'The shape of sigma : {}'.format(sigma.shape)

  offset = downsample / 2.0 - 0.5
  num_points, threshold = pts.shape[1], 0.01

  if visiable is None: visiable = pts[2, :].astype('bool')
  assert visiable.shape[0] == num_points

  transformed_label = np.fromfunction( lambda y, x, pid : ((offset + x*downsample - pts[0,pid])**2 \
                                                        + (offset + y*downsample - pts[1,pid])**2) \
                                                          / -2.0 / sigma[pid] / sigma[pid],
                                                          (height, width, num_points), dtype=int)

  mask_heatmap      = np.ones((1, 1, num_points+1), dtype='float32')
  mask_heatmap[0, 0, :num_points] = visiable
  mask_heatmap[0, 0, num_points]  = 1
  
  transformed_label = np.exp(transformed_label)
  transformed_label[ transformed_label < threshold ] = 0
  transformed_label[ transformed_label >         1 ] = 1

  background_label  = 1 - np.amax(transformed_label, axis=2)
  background_label[ background_label < 0 ] = 0
  heatmap           = np.concatenate((transformed_label, np.expand_dims(background_label, axis=2)), axis=2).astype('float32')
  
  return heatmap*mask_heatmap, mask_heatmap
