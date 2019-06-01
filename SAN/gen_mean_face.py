import os, random, time
import pdb
import torch
from PIL import Image
import init_path
import numpy as np
from os import path as osp
import datasets
from san_vision import transforms
import torch.nn.functional as F

def normalize(L, x):
  return -1. + 2. * x / (L-1)

def np2variable(x, requires_grad=False, dtype=torch.FloatTensor):
  v = torch.autograd.Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
  return v

def face_align(face, point, target):
  spatial_size = np.array(face.size)
  point, target = point.copy(), target.copy()
  point[:,0] = normalize(spatial_size[0], point[:,0])
  point[:,1] = normalize(spatial_size[1], point[:,1])
  target[:,0] = normalize(spatial_size[0], target[:,0])
  target[:,1] = normalize(spatial_size[1], target[:,1])
  x, residual, rank, s = np.linalg.lstsq(target, point)
  theta = x.T[:2,:]
  theta = np2variable(theta).unsqueeze(0)
  image = np.array(face.copy()).transpose(2, 0, 1)
  image_var = np2variable(image, False).unsqueeze(0)
  grid_size = torch.Size([1, 3, int(spatial_size[1]), int(spatial_size[0])])
  grid = F.affine_grid(theta, grid_size)
  aligned_image = F.grid_sample(image_var, grid)
  aligned_image = aligned_image.data.numpy().squeeze()
  aligned_image = aligned_image.transpose(1, 2, 0)
  aligned_image = Image.fromarray(np.uint8(aligned_image))
  return aligned_image

def calculate_mean(list_file, num_pts, save_path):
  #style = 'Original'
  #save_dir = 'cache/{}'.format(style)
  save_dir = osp.dirname(save_path)
  print ('crop face images into {} <-> {}'.format(save_dir, save_path))
  if not osp.isdir(save_dir): os.makedirs(save_dir)
  transform  = transforms.Compose([transforms.PreCrop(0.2), transforms.TrainScale2WH((256, 256))])
  data = datasets.GeneralDataset(transform, 1, 8, 'gaussian', 'test')
  data.load_list(list_file, num_pts, True)
  ok_faces, ok_basenames, ok_points = [], [], []
  for i, tempx in enumerate(data):
    image, mask, points = tempx[0], tempx[2], tempx[3]
    #points = points[[0, 8, 16, 36, 39, 42, 45, 33, 48, 54, 27, 57],:]
    basename = osp.basename(data.datas[i])
    if torch.sum(mask) == num_pts + 1:
      ok_faces.append( image )
      ok_basenames.append( basename )
      ok_points.append( points.numpy() )
  print ('extract done {:} -> {:}'.format(len(data), len(ok_faces)))
  mean_landmark = np.array(ok_points).mean(axis=0)
  all_faces = []
  save_dir = save_dir + '-all'
  if not osp.isdir(save_dir): os.makedirs(save_dir)

  for face, point, basename in zip(ok_faces, ok_points, ok_basenames):
    aligned_face = face_align(face, point, mean_landmark)
    aligned_face.save(osp.join(save_dir, basename))
    all_faces.append( np.array(aligned_face) )
  all_faces = np.array(all_faces).mean(axis=0)
  mean_face = Image.fromarray(np.uint8(all_faces))
  mean_face.save(save_path)

def generate_300W(cluster_num):
  for i in range(cluster_num):
    save_path = osp.join('cache_data', 'cache', 'clusters', '300W-{:}'.format(cluster_num), '300W-{:}-{:}.png'.format(i, cluster_num))
    calculate_mean(['./snapshots/CLUSTER-300W_GTB-{:d}/cluster-{:02d}-{:02d}.lst'.format(cluster_num, i, cluster_num)], 68, save_path)

    save_path = osp.join('cache_data', 'cache', 'clusters', '300W-BASE-{:}'.format(cluster_num), '300W-BASE-{:}-{:}.png'.format(i, cluster_num))
    calculate_mean(['./snapshots/BASE-CLUSTER-300W_GTB-{:d}/cluster-{:02d}-{:02d}.lst'.format(cluster_num, i, cluster_num)], 68, save_path)

def generate_AFLW(cluster_num):
  for i in range(cluster_num):
    save_path = osp.join('cache_data', 'cache', 'clusters', 'AFLW-{:}'.format(cluster_num), 'AFLW-{:}-{:}.png'.format(i, cluster_num))
    calculate_mean(['./snapshots/CLUSTER-AFLW_GTB-{:d}/cluster-{:02d}-{:02d}.lst'.format(cluster_num, i, cluster_num)], 19, save_path)
    save_path = osp.join('cache_data', 'cache', 'clusters', 'AFLW-BASE-{:}'.format(cluster_num), 'AFLW-BASE-{:}-{:}.png'.format(i, cluster_num))
    calculate_mean(['./snapshots/BASE-CLUSTER-AFLW_GTB-{:d}/cluster-{:02d}-{:02d}.lst'.format(cluster_num, i, cluster_num)], 19, save_path)

if __name__ == '__main__':
  generate_300W(3)
  """
  for cluster_num in [3,4,5,6]:
    generate_AFLW(cluster_num)
    generate_300W(cluster_num)
  """
