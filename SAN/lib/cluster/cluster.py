##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import numpy as np
from sklearn.preprocessing import normalize
import pdb

def cos_dis(x, y):
  x = normalize(x[:,np.newaxis], axis=0).ravel()
  y = normalize(y[:,np.newaxis], axis=0).ravel()
  return np.linalg.norm(x-y)

def filter_cluster(indexes, cluster_features, ratio):
  num_feature = cluster_features.shape[0]
  mean_feature = np.mean(cluster_features, axis=0)

  all_L1, all_L2, all_LC = [], [], []
  for i in range(num_feature):
    x = cluster_features[i]
    L1 = np.sum(np.abs((x-mean_feature)))
    L2 = np.linalg.norm(x-mean_feature)
    LC = cos_dis(x, mean_feature)
    all_L1.append( L1 )
    all_L2.append( L2 )
    all_LC.append( LC )
  all_L1 = np.array(all_L1)
  all_L2 = np.array(all_L2)
  all_LC = np.array(all_LC)
  threshold = (all_L2.max()-all_L2.min())*ratio+all_L2.min()
  selected = indexes[ all_L2 < threshold ]
  return selected.copy()
