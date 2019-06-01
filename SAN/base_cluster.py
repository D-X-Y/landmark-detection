##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
from __future__ import division

import os, sys, time, random, argparse, PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # please use Pillow 4.0.0 or it may fail for some images
from os import path as osp
import numbers, numpy as np
import init_path
import torch
import datasets
from shutil import copyfile
from san_vision import transforms
from utils import AverageMeter, print_log
from utils import convert_size2str, convert_secs2time, time_string, time_for_file
from visualization import draw_image_by_points, save_error_image
import debug, models, options
from sklearn.cluster import KMeans
from cluster import filter_cluster

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

opt = options.Options(model_names)
args = opt.opt
# Prepare options
if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)
torch.backends.cudnn.enabled   = True
#torch.backends.cudnn.benchmark = True

def main():
  # Init logger
  if not os.path.isdir(args.save_path): os.makedirs(args.save_path)
  log = open(os.path.join(args.save_path, 'cluster_seed_{}_{}.txt'.format(args.manualSeed, time_for_file())), 'w')
  print_log('save path : {}'.format(args.save_path), log)
  print_log('------------ Options -------------', log)
  for k, v in sorted(vars(args).items()):
    print_log('Parameter : {:20} = {:}'.format(k, v), log)
  print_log('-------------- End ----------------', log)
  print_log("Random Seed: {}".format(args.manualSeed), log)
  print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
  print_log("Pillow version : {}".format(PIL.__version__), log)
  print_log("torch  version : {}".format(torch.__version__), log)
  print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

  # General Data Argumentation
  mean_fill   = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
  normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
  transform  = transforms.Compose([transforms.PreCrop(args.pre_crop_expand), transforms.TrainScale2WH((args.crop_width, args.crop_height)),  transforms.ToTensor(), normalize])

  args.downsample = 8 # By default
  args.sigma = args.sigma * args.scale_eval

  data = datasets.GeneralDataset(transform, args.sigma, args.downsample, args.heatmap_type, args.dataset_name)
  data.load_list(args.train_list, args.num_pts, True)
  loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

  # Load all lists
  all_lines = {}
  for file_path in args.train_list:
    listfile = open(file_path, 'r')
    listdata = listfile.read().splitlines()
    listfile.close()
    for line in listdata:
      temp = line.split(' ')
      assert len(temp) == 6  or len(temp) == 7, 'This line has the wrong format : {}'.format(line)
      image_path = temp[0]
      all_lines[ image_path ] = line

  assert args.n_clusters >= 2, 'The cluster number must be greater than 2'
  resnet = models.resnet152(True).cuda()
  all_features = []
  for i, (inputs, target, mask, points, image_index, label_sign, ori_size) in enumerate(loader):
    input_vars = torch.autograd.Variable(inputs.cuda(), volatile=True)
    features, classifications = resnet(input_vars)
    features = features.cpu().data.numpy()
    all_features.append( features )
    if i % args.print_freq == 0:
      print_log('{} {}/{} extract features'.format(time_string(), i, len(loader)), log)
  all_features = np.concatenate(all_features, axis=0)
  kmeans_result = KMeans(n_clusters=args.n_clusters, n_jobs=args.workers).fit( all_features )
  print_log('kmeans [{}] calculate done'.format(args.n_clusters), log)
  labels = kmeans_result.labels_.copy()

  cluster_idx = []
  for iL in range(args.n_clusters):
    indexes = np.where( labels == iL )[0]
    cluster_idx.append( len(indexes) )
  cluster_idx = np.argsort(cluster_idx)
    
  for iL in range(args.n_clusters):
    ilabel = cluster_idx[iL]
    indexes = np.where( labels == ilabel )
    if isinstance(indexes, tuple) or isinstance(indexes, list): indexes = indexes[0]
    cluster_features = all_features[indexes,:].copy()
    filtered_index = filter_cluster(indexes.copy(), cluster_features, 0.8)

    print_log('{:} [{:2d} / {:2d}] has {:4d} / {:4d} -> {:4d} = {:.2f} images '.format(time_string(), iL, args.n_clusters, indexes.size, len(data), len(filtered_index), indexes.size*1./len(data)), log)
    indexes = filtered_index.copy()
    save_dir = osp.join(args.save_path, 'cluster-{:02d}-{:02d}'.format(iL, args.n_clusters))
    save_path = save_dir + '.lst'
    #if not osp.isdir(save_path): os.makedirs( save_path )
    print_log('save into {}'.format(save_path), log)
    txtfile = open( save_path , 'w')
    for idx in indexes:
      image_path = data.datas[idx]
      assert image_path in all_lines, 'Not find {}'.format(image_path)
      txtfile.write('{}\n'.format(all_lines[image_path]))
      #basename = osp.basename( image_path )
      #os.system( 'cp {} {}'.format(image_path, save_dir) )
    txtfile.close()

if __name__ == '__main__':
  main()
