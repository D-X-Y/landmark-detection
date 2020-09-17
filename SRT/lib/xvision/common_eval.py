# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, time, warnings, numpy as np
from sklearn.metrics import auc
from log_utils import print_log


def get_mean_point(points):
  assert points.ndim == 2 and (points.shape[0] == 2 or points.shape[0] == 3), '{:}'.format(points.shape)
  if points.shape[1] == 3:
    oks = points[2, :] == 1
    assert oks.sum() >= 1, 'points : {:}'.format(points)
    points = points[:2, oks]
  return points.mean(1)


def compute_PCKh(PCK_head, log):
  PCK_head = [np.array(x) for x in PCK_head]
  distances = np.concatenate((PCK_head[0], PCK_head[5]))
  print_log('PCKh - ankle    : {:6.3f} %'.format( (distances<0.5).mean() * 100 ), log)
  distances = np.concatenate((PCK_head[1], PCK_head[4]))
  print_log('PCKh - knee     : {:6.3f} %'.format( (distances<0.5).mean() * 100 ), log)
  distances = np.concatenate((PCK_head[2], PCK_head[3]))
  print_log('PCKh - hip      : {:6.3f} %'.format( (distances<0.5).mean() * 100 ), log)
  distances = np.concatenate((PCK_head[6], PCK_head[11]))
  print_log('PCKh - wrist    : {:6.3f} %'.format( (distances<0.5).mean() * 100 ), log)
  distances = np.concatenate((PCK_head[7], PCK_head[10]))
  print_log('PCKh - elbow    : {:6.3f} %'.format( (distances<0.5).mean() * 100 ), log)
  distances = np.concatenate((PCK_head[8], PCK_head[9]))
  print_log('PCKh - shoulder : {:6.3f} %'.format( (distances<0.5).mean() * 100 ), log)
  distances = np.concatenate((PCK_head[12], PCK_head[13]))
  print_log('PCKh - head     : {:6.3f} %'.format( (distances<0.5).mean() * 100 ), log)
  distances = np.concatenate(PCK_head)
  print_log('PCKh : {:6.4f} %'.format( (distances<0.5).mean() * 100 ), log)


def evaluate_normalized_mean_error(predictions, groundtruth, extra_normalizers, indicator, log):
  ## compute total average normlized mean error
  assert len(predictions) == len(groundtruth), 'The lengths of predictions and ground-truth are not consistent : {} vs {}'.format( len(predictions), len(groundtruth) )
  assert len(predictions) > 0, 'The length of predictions must be greater than 0 vs {}'.format( len(predictions) )
  assert extra_normalizers is None or len(extra_normalizers) == len(predictions), 'The length of extra_normalizers is not right {} vs {}'.format( len(extra_normalizers), len(predictions) )
  if indicator is None: warnings.warn('In evaluate NME, indicator is None, do not set normalizer distance.')

  num_images = len(predictions)
  for i in range(num_images):
    c, g = predictions[i], groundtruth[i]
    assert isinstance(c, np.ndarray) and isinstance(g, np.ndarray), 'The type of predictions is not right : [{:}] :: {} vs {} '.format(i, type(c), type(g))

  num_points = predictions[0].shape[1]
  error_per_image = np.zeros((num_images,1))
  if num_points == 16: PCK_head = [ [] for i in range(16) ]
  else               : PCK_head = None

  for i in range(num_images):
    detected_points = predictions[i]
    ground_truth_points = groundtruth[i]
    if indicator is None:
      interocular_distance = 1
    elif indicator.startswith('HandsyROT'):
      interocular_distance = 1
    elif indicator.startswith('face68_pupil'):
      # compute the inter-pupil distance
      Lpupil = get_mean_point(ground_truth_points[:, [36,37,38,39,40,41]])
      Rpupil = get_mean_point(ground_truth_points[:, [42,43,44,45,46,47]])
      # inter pupil distance
      interocular_distance = np.linalg.norm(Lpupil - Rpupil)
    elif indicator.startswith('Mugsy-18'):
      interocular_distance = np.linalg.norm(ground_truth_points[:2, 4] - ground_truth_points[:2, 8])
      assert num_points == 18 and bool(ground_truth_points[2,4]) and bool(ground_truth_points[2,8]), 'pts : {:}'.format(num_points)
    elif indicator.startswith('Sync-19'): # inter-ocular
      interocular_distance = np.linalg.norm(ground_truth_points[:2, 5] - ground_truth_points[:2, 8]) 
      assert num_points == 19 and bool(ground_truth_points[2,5]) and bool(ground_truth_points[2,8]), 'pts : {:}'.format(num_points)
    elif indicator.startswith('face68'):
      interocular_distance = np.linalg.norm(ground_truth_points[:2, 36] - ground_truth_points[:2, 45])
      assert num_points == 68 and bool(ground_truth_points[2,36]) and bool(ground_truth_points[2,45]), 'pts : {:}'.format(num_points)
    elif indicator.startswith('face49') or indicator.startswith('face51'):
      interocular_distance = np.linalg.norm(ground_truth_points[:2, 19] - ground_truth_points[:2, 28])
      assert (num_points == 51 or num_points == 49) and bool(ground_truth_points[2,19]) and bool(ground_truth_points[2,28])
    elif indicator.startswith('face98'):
      interocular_distance = np.linalg.norm(ground_truth_points[:2, 60] - ground_truth_points[:2, 72])
      assert num_points == 98 and bool(ground_truth_points[2,60]) and bool(ground_truth_points[2,72])
    elif indicator.startswith('face19'):
      assert num_points == 19 and extra_normalizers is not None and extra_normalizers[i] is not None, 'PTS={:}, normalizer={:}'.format(num_points, extra_normalizers[i])
      interocular_distance = extra_normalizers[i]
    elif indicator.startswith('pose16'): # MPII
      assert num_points == 16 and extra_normalizers is not None and extra_normalizers[i] is not None, 'PTS={:}, normalizer={:}'.format(num_points, extra_normalizers[i])
      interocular_distance = extra_normalizers[i]
      headsize = extra_normalizers[i]
    else:
      raise Exception('----> Unknown indicator : {:}'.format(indicator))
    dis_sum, pts_sum = 0, 0
    for j in range(num_points):
      if bool(ground_truth_points[2, j]):
        distance = np.linalg.norm(detected_points[:2, j] - ground_truth_points[:2, j])
        dis_sum, pts_sum = dis_sum + distance, pts_sum + 1
        if PCK_head is not None: # calculate PCKh
          PCK_head[j].append( distance / headsize )
    error_per_image[i] = dis_sum / (pts_sum*interocular_distance)

  normalise_mean_error = error_per_image.mean()
  # calculate the auc for 0.07
  max_threshold = 0.07
  threshold = np.linspace(0, max_threshold, num=2000)
  accuracys = np.zeros(threshold.shape)
  for i in range(threshold.size):
    accuracys[i] = np.sum(error_per_image < threshold[i]) * 1.0 / error_per_image.size
  area_under_curve07 = auc(threshold, accuracys) / max_threshold
  # calculate the auc for 0.08
  max_threshold = 0.08
  threshold = np.linspace(0, max_threshold, num=2000)
  accuracys = np.zeros(threshold.shape)
  for i in range(threshold.size):
    accuracys[i] = np.sum(error_per_image < threshold[i]) * 1.0 / error_per_image.size
  area_under_curve08 = auc(threshold, accuracys) / max_threshold
  # calculate the auc for 0.10
  max_threshold = 0.10
  threshold = np.linspace(0, max_threshold, num=2000)
  accuracys = np.zeros(threshold.shape)
  for i in range(threshold.size):
    accuracys[i] = np.sum(error_per_image < threshold[i]) * 1.0 / error_per_image.size
  area_under_curve10 = auc(threshold, accuracys) / max_threshold
  
  accuracy_under_007 = np.sum(error_per_image<0.07) * 100. / error_per_image.size
  accuracy_under_008 = np.sum(error_per_image<0.08) * 100. / error_per_image.size
  accuracy_under_010 = np.sum(error_per_image<0.10) * 100. / error_per_image.size

  print_log('Compute NME and AUC for {:} images with {:} points :: [(nms): mean={:.3f}, std={:.3f}], auc@0.07={:.3f}, auc@0.08={:.3f}, auc@0.10={:.3f}, acc@0.07={:.3f}, acc@0.08={:.3f}, acc@0.10={:.3f}'.format(num_images, num_points, normalise_mean_error*100, error_per_image.std()*100, \
    area_under_curve07*100, area_under_curve08*100, area_under_curve10*100, accuracy_under_007, accuracy_under_008, accuracy_under_010), log)

  if PCK_head is not None: compute_PCKh( PCK_head, log )

  for_pck_curve = []
  for x in range(0, 3501, 1):
    error_bar = x * 0.0001
    accuracy = np.sum(error_per_image < error_bar) * 1.0 / error_per_image.size
    for_pck_curve.append((error_bar, accuracy))
  
  return normalise_mean_error, accuracy_under_008, for_pck_curve, error_per_image
