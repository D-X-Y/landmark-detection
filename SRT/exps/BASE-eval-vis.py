# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os, sys, time, torch, random, argparse, PIL
import os.path as osp
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import numbers, numpy as np
from torch import multiprocessing as mp
from torch.nn import functional as F
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)

from utils import VideoWriter
from xvision import normalize_points, denormalize_points
from xvision import draw_image_by_points

from procedure import get_path2image, save_checkpoint
from procedure import prepare_seed, prepare_logger
from datasets  import EvalDataset, WrapParallel, WrapParallelV2
from xvision   import transforms2v as transforms
from log_utils import AverageMeter, time_for_file, convert_secs2time, time_string
from config_utils import load_configure
from models import obtain_pro_model, remove_module_dict, count_parameters_in_MB, load_checkpoint



def fc_solve(Xgts, Xpredictions, is_cuda=True):
  if is_cuda:
    Xgts, Xpredictions = Xgts.cuda(), Xpredictions.cuda()
  x_dim, y_dim = Xgts.size(2), Xpredictions.size(2)
  #A, B = torch.rand(1, x_dim, 2, requires_grad=True, device=Xgts.device), torch.rand(1, 2, y_dim, requires_grad=True, device=Xgts.device)
  A, B = torch.rand(1, x_dim, 2, requires_grad=True, device=Xgts.device), torch.rand(1, 2, y_dim, requires_grad=True, device=Xgts.device)
  #A, B = torch.rand(1, x_dim, 2, requires_grad=True, device=Xgts.device), torch.rand(1, 3, y_dim, requires_grad=True, device=Xgts.device)
  optim = torch.optim.Adam([A,B], lr=0.1)
  def get_pred(x, a, b):
    temp1 = torch.matmul(x, a)
    temp2 = torch.matmul(temp1, b)
    return temp2
  T_weights = torch.tensor([1,1,10], device=Xgts.device).view(1,3,1)
  min_loss, final_predicts, max_iters, _iter = 100000, None, 60000, 0
  #for _iter in range(total):
  while min_loss > 2:
    preds = get_pred(Xgts, A, B)
    #losses= F.mse_loss(Xpredictions, preds, reduction='none') * T_weights
    losses= F.l1_loss(Xpredictions, preds, reduction='none') * T_weights
    # select_top 10%
    dis_ls= torch.norm(losses[:,:2], dim=1)
    topk, indices = torch.topk(dis_ls.view(-1), dis_ls.numel()//10)
    loss  = losses.mean() + topk.mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    if _iter == int(max_iters*0.6) or _iter == int(max_iters*0.8):
      optim.param_groups[0]['lr'] *= 0.2
    #print ('_iter={:04d} : loss={:}'.format(_iter, loss.item()))
    if min_loss > loss.item():
      min_loss, final_predicts = loss.item(), preds.detach()
    _iter += 1
    if _iter > max_iters: break
  print ('{:} final-loss={:.3f}, minal-loss={:.3f}, with _iter={:}'.format(time_string(), loss.item(), min_loss, _iter))
  return final_predicts.cpu()

def fc_solve_v2(Xgts, Xpredictions, is_cuda=True):
  total = Xgts.size(0)
  X, Y  = Xgts[:,:1,:].mean(dim=2, keepdim=True), Xgts[:,1:2,:].mean(dim=2, keepdim=True)
  offset = torch.cat((X,Y,torch.zeros(total,1,1, device=Xgts.device)), dim=1)
  Xgts, Xpredictions = Xgts-offset, Xpredictions-offset
  # 1-33  ~ 1-17 轮廓 1-17 ~ 1-9 / 18-33 ~ 10 - 17
  new_preds_0 = fc_solve(Xgts[:,:,list(range( 0, 9))], Xpredictions[:,:,list(range( 0,17))])
  new_preds_1 = fc_solve(Xgts[:,:,list(range( 9,17))], Xpredictions[:,:,list(range(17,33))])
  # 34-42 ~ 18-22 左眉毛
  new_preds_2 = fc_solve(Xgts[:,:,list(range(17,22))], Xpredictions[:,:,list(range(33,42))])
  # 43-51 ~ 23-27 右眉毛
  new_preds_3 = fc_solve(Xgts[:,:,list(range(22,27))], Xpredictions[:,:,list(range(42,51))])
  # 52-60 ~ 28~36 鼻梁
  new_preds_4 = fc_solve(Xgts[:,:,list(range(27,36))], Xpredictions[:,:,list(range(51,60))])
  # 61-68 ~ 37~42 左眼
  new_preds_5 = fc_solve(Xgts[:,:,list(range(36,42))], Xpredictions[:,:,list(range(60,68))])
  # 69-76 ~ 43~48 右眼
  new_preds_6 = fc_solve(Xgts[:,:,list(range(42,48))], Xpredictions[:,:,list(range(68,76))])
  # 77-96 ~ 49~68 嘴巴
  new_preds_7 = fc_solve(Xgts[:,:,list(range(48,68))], Xpredictions[:,:,list(range(76,96))])
  left_eye_c  = torch.stack((new_preds_5[:,0].mean(dim=1), new_preds_5[:,1].mean(dim=1), new_preds_5[:,2].mean(dim=1))).permute(1,0).view(total, 3, 1)
  righ_eye_c  = torch.stack((new_preds_6[:,0].mean(dim=1), new_preds_6[:,1].mean(dim=1), new_preds_6[:,2].mean(dim=1))).permute(1,0).view(total, 3, 1)
  new_preds   = torch.cat((new_preds_0, new_preds_1, new_preds_2, new_preds_3, new_preds_4, new_preds_5, new_preds_6, new_preds_7, left_eye_c, righ_eye_c), dim=2)
  return new_preds+offset


def main(args):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = True
  torch.set_num_threads( args.workers )
  print ('Training Base Detector : prepare_seed : {:}'.format(args.rand_seed))
  prepare_seed(args.rand_seed)

  logger = prepare_logger(args)

  checkpoint = load_checkpoint( args.init_model )
  xargs      = checkpoint['args']
  logger.log('Previous args : {:}'.format(xargs))

  # General Data Augmentation
  if xargs.use_gray == False:
    mean_fill = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])
  else:
    mean_fill = (0.5,)
    normalize = transforms.Normalize(mean=[mean_fill[0]], std=[0.5])
  eval_transform  = transforms.Compose2V([transforms.ToTensor(), normalize, \
                                              transforms.PreCrop(xargs.pre_crop_expand), \
                                              transforms.CenterCrop(xargs.crop_max)])

  # Model Configure Load
  model_config = load_configure(xargs.model_config, logger)
  shape = (xargs.height, xargs.width)
  logger.log('--> {:}\n--> Sigma : {:}, Shape : {:}'.format(model_config, xargs.sigma, shape))


  # Evaluation Dataloader
  eval_loaders = []
  if args.eval_ilists is not None:
    for eval_ilist in args.eval_ilists:
      eval_idata = EvalDataset(eval_transform, xargs.sigma, model_config.downsample, xargs.heatmap_type, shape, xargs.use_gray, xargs.data_indicator)
      eval_idata.load_list(eval_ilist, args.num_pts, xargs.boxindicator, xargs.normalizeL, True)
      eval_iloader = torch.utils.data.DataLoader(eval_idata, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
      eval_loaders.append((eval_iloader, False))
  if args.eval_vlists is not None:
    for eval_vlist in args.eval_vlists:
      eval_vdata = EvalDataset(eval_transform, xargs.sigma, model_config.downsample, xargs.heatmap_type, shape, xargs.use_gray, xargs.data_indicator)
      eval_vdata.load_list(eval_vlist, args.num_pts, xargs.boxindicator, xargs.normalizeL, True)
      eval_vloader = torch.utils.data.DataLoader(eval_vdata, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
      eval_loaders.append((eval_vloader, True))

  # define the detector
  detector = obtain_pro_model(model_config, xargs.num_pts, xargs.sigma, xargs.use_gray)
  assert model_config.downsample == detector.downsample, 'downsample is not correct : {:} vs {:}'.format(model_config.downsample, detector.downsample)
  logger.log("=> detector :\n {:}".format(detector))
  logger.log("=> Net-Parameters : {:} MB".format(count_parameters_in_MB(detector)))
  logger.log('=> Eval-Transform : {:}'.format(eval_transform))

  detector = detector.cuda()
  net = torch.nn.DataParallel(detector)
  net.eval()
  net.load_state_dict( checkpoint['detector'] )
  cpu = torch.device('cpu')

  assert len(args.use_stable) == 2

  for iLOADER, (loader, is_video) in enumerate(eval_loaders):
    logger.log('{:} The [{:2d}/{:2d}]-th test set [{:}] = {:} with {:} batches.'.format(time_string(), iLOADER, len(eval_loaders), 'video' if is_video else 'image', loader.dataset, len(loader)))
    with torch.no_grad():
      all_points, all_results, all_image_ps = [], [], []
      for i, (inputs, targets, masks, normpoints, transthetas, image_index, nopoints, shapes) in enumerate(loader):
        image_index = image_index.squeeze(1).tolist()
        (batch_size, C, H, W), num_pts = inputs.size(), xargs.num_pts
        # batch_heatmaps is a list for stage-predictions, each element should be [Batch, C, H, W]
        if xargs.procedure == 'heatmap':
          batch_features, batch_heatmaps, batch_locs, batch_scos = net(inputs)
          batch_locs = batch_locs[:, :-1, :]
        else:
          batch_locs = net(inputs)
        batch_locs = batch_locs.detach().to(cpu)
        # evaluate the training data
        for ibatch, (imgidx, nopoint) in enumerate(zip(image_index, nopoints)):
          if xargs.procedure == 'heatmap':
            norm_locs = normalize_points((H,W), batch_locs[ibatch].transpose(1,0))
            norm_locs = torch.cat((norm_locs, torch.ones(1, num_pts)), dim=0)
          else:
            norm_locs  = torch.cat((batch_locs[ibatch].permute(1,0), torch.ones(1, num_pts)), dim=0)
          transtheta = transthetas[ibatch][:2,:]
          norm_locs  = torch.mm(transtheta, norm_locs)
          real_locs  = denormalize_points(shapes[ibatch].tolist(), norm_locs)
          #real_locs  = torch.cat((real_locs, batch_scos[ibatch].permute(1,0)), dim=0)
          real_locs  = torch.cat((real_locs, torch.ones(1, num_pts)), dim=0)
          xpoints    = loader.dataset.labels[imgidx].get_points().numpy()
          image_path = loader.dataset.datas[imgidx]
          # put into the list
          all_points.append( torch.from_numpy(xpoints) )
          all_results.append( real_locs )
          all_image_ps.append( image_path )
      total = len(all_points)
      logger.log('{:} The [{:2d}/{:2d}]-th test set finishes evaluation : {:} frames/images'.format(time_string(), iLOADER, len(eval_loaders), total))
   
    """
    if args.use_stable[0] > 0:
      save_dir = Path( osp.join(args.save_path, '{:}-X-{:03d}'.format(args.model_name, iLOADER)) )
      save_dir.mkdir(parents=True, exist_ok=True)
      wrap_parallel = WrapParallel(save_dir, all_image_ps, all_results, all_points, 180, (255, 0, 0))
      wrap_loader   = torch.utils.data.DataLoader(wrap_parallel, batch_size=args.workers, shuffle=False, num_workers=args.workers, pin_memory=True)
      for iL, INDEXES in enumerate(wrap_loader): _ = INDEXES
      cmd = 'ffmpeg -y -i {:}/%06d.png -framerate 30 {:}.avi'.format(save_dir, save_dir)
      logger.log('{:} possible >>>>> : {:}'.format(time_string(), cmd))
      os.system( cmd )

    if args.use_stable[1] > 0:
      save_dir = Path( osp.join(args.save_path, '{:}-Y-{:03d}'.format(args.model_name, iLOADER)) )
      save_dir.mkdir(parents=True, exist_ok=True)
      Xpredictions, Xgts = torch.stack(all_results), torch.stack(all_points)
      new_preds = fc_solve(Xgts, Xpredictions, is_cuda=True)
      wrap_parallel = WrapParallel(save_dir, all_image_ps, new_preds, all_points, 180, (0, 0, 255))
      wrap_loader   = torch.utils.data.DataLoader(wrap_parallel, batch_size=args.workers, shuffle=False, num_workers=args.workers, pin_memory=True)
      for iL, INDEXES in enumerate(wrap_loader): _ = INDEXES
      cmd = 'ffmpeg -y -i {:}/%06d.png -framerate 30 {:}.avi'.format(save_dir, save_dir)
      logger.log('{:} possible >>>>> : {:}'.format(time_string(), cmd))
      os.system( cmd )
    """
    Xpredictions, Xgts = torch.stack(all_results), torch.stack(all_points)
    save_path = Path( osp.join(args.save_path, '{:}-result-{:03d}.pth'.format(args.model_name, iLOADER)) )
    torch.save({'paths': all_image_ps,
                'ground-truths': Xgts,
                'predictions'  : all_results}, save_path)
    logger.log('{:} save into {:}'.format(time_string(), save_path))
    if False:
      new_preds = fc_solve_v2(Xgts, Xpredictions, is_cuda=True)
      # create the dir
      save_dir = Path( osp.join(args.save_path, '{:}-T-{:03d}'.format(args.model_name, iLOADER)) )
      save_dir.mkdir(parents=True, exist_ok=True)
      wrap_parallel = WrapParallelV2(save_dir, all_image_ps, Xgts, all_results, new_preds, all_points, 180, [args.model_name, 'SRT'])
      wrap_parallel[0]
      wrap_loader   = torch.utils.data.DataLoader(wrap_parallel, batch_size=args.workers, shuffle=False, num_workers=args.workers, pin_memory=True)
      for iL, INDEXES in enumerate(wrap_loader): _ = INDEXES
      cmd = 'ffmpeg -y -i {:}/%06d.png -vb 5000k {:}.avi'.format(save_dir, save_dir)
      logger.log('{:} possible >>>>> : {:}'.format(time_string(), cmd))
      os.system( cmd )

  logger.close() ; return



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluating landmark detectors', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # path
  parser.add_argument('--init_model',       type=str,                   help='.')
  parser.add_argument('--save_path' ,       type=str,                   help='.')
  parser.add_argument('--model_name',       type=str,                   help='.')
  parser.add_argument('--eval_vlists',      type=str,   nargs='+',      help='The list file path to the video testing dataset.')
  parser.add_argument('--eval_ilists',      type=str,   nargs='+',      help='The list file path to the image testing dataset.')
  parser.add_argument('--batch_size',       type=int,   default=2,      help='Batch size for training.')
  parser.add_argument('--print_freq',       type=int,   default=2,      help='.')
  parser.add_argument('--num_pts',          type=int,   default=2,      help='.')
  parser.add_argument('--workers',          type=int,   default=2,      help='.')
  parser.add_argument('--use_stable',       type=int,   nargs='+',      help='.')
  parser.add_argument('--rand_seed',        type=int,   default=-1,     help='.')

  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
