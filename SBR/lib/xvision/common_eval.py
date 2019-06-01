# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import pdb, os, time
from log_utils import print_log
from datasets.dataset_utils import convert68to49, convert68to51
from sklearn.metrics import auc

def evaluate_normalized_mean_error(predictions, groundtruth, log, extra_faces):
  ## compute total average normlized mean error
  assert len(predictions) == len(groundtruth), 'The lengths of predictions and ground-truth are not consistent : {} vs {}'.format( len(predictions), len(groundtruth) )
  assert len(predictions) > 0, 'The length of predictions must be greater than 0 vs {}'.format( len(predictions) )
  if extra_faces is not None: assert len(extra_faces) == len(predictions), 'The length of extra_faces is not right {} vs {}'.format( len(extra_faces), len(predictions) )
  num_images = len(predictions)
  for i in range(num_images):
    c, g = predictions[i], groundtruth[i]
    assert isinstance(c, np.ndarray) and isinstance(g, np.ndarray), 'The type of predictions is not right : [{:}] :: {} vs {} '.format(i, type(c), type(g))

  num_points = predictions[0].shape[1]
  error_per_image = np.zeros((num_images,1))
  for i in range(num_images):
    detected_points = predictions[i]
    ground_truth_points = groundtruth[i]
    if num_points == 68:
      interocular_distance = np.linalg.norm(ground_truth_points[:2, 36] - ground_truth_points[:2, 45])
      assert bool(ground_truth_points[2,36]) and bool(ground_truth_points[2,45])
    elif num_points == 51 or num_points == 49:
      interocular_distance = np.linalg.norm(ground_truth_points[:2, 19] - ground_truth_points[:2, 28])
      assert bool(ground_truth_points[2,19]) and bool(ground_truth_points[2,28])
    elif num_points == 19:
      assert extra_faces is not None and extra_faces[i] is not None
      interocular_distance = extra_faces[i]
    else:
      raise Exception('----> Unknown number of points : {}'.format(num_points))
    dis_sum, pts_sum = 0, 0
    for j in range(num_points):
      if bool(ground_truth_points[2, j]):
        dis_sum = dis_sum + np.linalg.norm(detected_points[:2, j] - ground_truth_points[:2, j])
        pts_sum = pts_sum + 1
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
  
  accuracy_under_007 = np.sum(error_per_image<0.07) * 100. / error_per_image.size
  accuracy_under_008 = np.sum(error_per_image<0.08) * 100. / error_per_image.size

  print_log('Compute NME and AUC for {:} images with {:} points :: [(NME): mean={:.3f}, std={:.3f}], auc@0.07={:.3f}, auc@0.08-{:.3f}, acc@0.07={:.3f}, acc@0.08={:.3f}'.format(num_images, num_points, normalise_mean_error*100, error_per_image.std()*100, area_under_curve07*100, area_under_curve08*100, accuracy_under_007, accuracy_under_008), log)

  for_pck_curve = []
  for x in range(0, 3501, 1):
    error_bar = x * 0.0001
    accuracy = np.sum(error_per_image < error_bar) * 1.0 / error_per_image.size
    for_pck_curve.append((error_bar, accuracy))
  
  return normalise_mean_error, accuracy_under_008, for_pck_curve
