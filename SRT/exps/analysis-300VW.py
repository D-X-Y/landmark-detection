# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Supervision-by-Registration

import sys, time, torch, random, argparse, PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
import numbers, numpy as np
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from config_utils import obtain_sbr_args_v2 as obtain_args

from procedure import prepare_seed, prepare_logger, save_checkpoint, prepare_data_augmentation
from procedure import sbr_main_heatmap, sbr_main_regression
from procedure import basic_eval_all_heatmap, basic_eval_all_regression
from datasets  import GeneralDatasetV2 as IDataset, convert68to49
from datasets  import VideoDatasetV2 as VDataset, SbrBatchSampler
from xvision   import transforms2v as transforms, draw_image_by_points_failure_case as draw_image_by_points
from xvision   import normalize_points, denormalize_points
from log_utils import AverageMeter, time_for_file, convert_secs2time, time_string
from config_utils import load_configure
from models    import obtain_pro_temporal, remove_module_dict, load_checkpoint
from optimizer import obtain_optimizer

procedures = {'default-train'   : sbr_main_heatmap,
              'default-test'    : basic_eval_all_heatmap,
              'heatmap-train'   : sbr_main_heatmap,
              'heatmap-test'    : basic_eval_all_heatmap,
              'regression-train': sbr_main_regression,
              'regression-test' : basic_eval_all_regression}

def get_in_map(locs):
  assert locs.dim() == 4, 'locs : {:}'.format(locs.shape)
  return torch.sum((locs > -1) + (locs < 1), dim=-1) == 4

def FB_communication(criterion, locs, past2now, future2now, FBcheck, mask, config):
  # return the calculate target from the first frame to the whole sequence.
  batch, frames, num_pts, _ = locs.size()
  assert batch == past2now.size(0) == future2now.size(0) == FBcheck.size(0), '{:} vs {:} vs {:} vs {:}'.format(locs.size(), past2now.size(), future2now.size(), FBcheck.size())
  assert num_pts == past2now.size(2) == future2now.size(2) == FBcheck.size(1), '{:} vs {:} vs {:} vs {:}'.format(locs.size(), past2now.size(), future2now.size(), FBcheck.size())
  assert frames-1 == past2now.size(1) == future2now.size(1), '{:} vs {:} vs {:} vs {:}'.format(locs.size(), past2now.size(), future2now.size(), FBcheck.size())
  assert mask.dim() == 4 and mask.size(0) == batch and mask.size(1) == num_pts, 'mask : {:}'.format(mask.size())


  locs, past2now, future2now = locs.contiguous(), past2now.contiguous(), future2now.contiguous()
  FBcheck, mask = FBcheck.contiguous(), mask.view(batch, num_pts).contiguous()
  with torch.no_grad():
    past2now_l1_dis = criterion.loss_l1_func(locs[:,1:], past2now, reduction='none')
    futu2now_l1_dis = criterion.loss_l1_func(locs[:,:-1], future2now, reduction='none')

    inmap_ok = get_in_map( locs ).sum(1) == frames
    check_ok = torch.sqrt(FBcheck[:,:,0]**2 + FBcheck[:,:,1]**2) < config.fb_thresh
    distc_ok = (past2now_l1_dis.sum(-1) + futu2now_l1_dis.sum(-1))/4 < config.dis_thresh
    distc_ok = distc_ok.sum(1) == frames-1
    data_ok  = (inmap_ok.view(batch, 1, num_pts, 1) + check_ok.view(batch, 1, num_pts, 1) + distc_ok.view(batch, 1, num_pts, 1) + mask.view(batch, 1, num_pts, 1)) == 4
  return data_ok


def main(args):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = True
  torch.set_num_threads( args.workers )
  print ('Training Base Detector : prepare_seed : {:}'.format(args.rand_seed))
  prepare_seed(args.rand_seed)
  temporal_main, eval_all = procedures['{:}-train'.format(args.procedure)], procedures['{:}-test'.format(args.procedure)]

  logger = prepare_logger(args)

  # General Data Argumentation
  normalize, train_transform, eval_transform, robust_transform = prepare_data_augmentation(transforms, args)
  recover = transforms.ToPILImage(normalize)
  args.tensor2imageF = recover
  assert (args.scale_min+args.scale_max) / 2 == 1, 'The scale is not ok : {:} ~ {:}'.format(args.scale_min, args.scale_max)
  
  # Model Configure Load
  model_config = load_configure(args.model_config, logger)
  sbr_config   = load_configure(args.sbr_config, logger)
  shape = (args.height, args.width)
  logger.log('--> {:}\n--> Sigma : {:}, Shape : {:}'.format(model_config, args.sigma, shape))
  logger.log('--> SBR Configuration : {:}\n'.format(sbr_config))

  # Training Dataset
  train_data   = VDataset(eval_transform, args.sigma, model_config.downsample, args.heatmap_type, shape, args.use_gray, args.mean_point, \
                            args.data_indicator, sbr_config, transforms.ToPILImage(normalize, 'cv2gray'))
  train_data.load_list(args.train_lists, args.num_pts, args.boxindicator, args.normalizeL, True)
  if args.x68to49:
    assert args.num_pts == 68, 'args.num_pts is not 68 vs. {:}'.format(args.num_pts)
    if train_data is not None: train_data = convert68to49( train_data )
    args.num_pts = 49


  # define the temporal model (accelerated SBR)
  net = obtain_pro_temporal(model_config, sbr_config, args.num_pts, args.sigma, args.use_gray)
  assert model_config.downsample == net.downsample, 'downsample is not correct : {:} vs {:}'.format(model_config.downsample, net.downsample)
  logger.log("=> network :\n {}".format(net))

  logger.log('Training-data : {:}'.format(train_data))

  logger.log('arguments : {:}'.format(args))
  opt_config = load_configure(args.opt_config, logger)

  
  optimizer, scheduler, criterion = obtain_optimizer(net.parameters(), opt_config, logger)
  logger.log('criterion : {:}'.format(criterion))
  net, criterion = net.cuda(), criterion.cuda()
  net = torch.nn.DataParallel(net)

  last_info = logger.last_info()
  try:
    last_checkpoint = load_checkpoint(args.init_model)
    checkpoint = remove_module_dict(last_checkpoint['state_dict'], False)
    net.module.detector.load_state_dict( checkpoint )
  except:
    last_checkpoint = load_checkpoint(args.init_model)
    net.load_state_dict(last_checkpoint['state_dict'])
  
  detector = torch.nn.DataParallel(net.module.detector)
  logger.log("=> initialize the detector : {:}".format(args.init_model))

  net.eval()
  detector.eval()

  logger.log('SBR Config : {:}'.format(sbr_config))
  save_xdir  = logger.path('meta')
  random.seed( 111 )
  index_list = list(range(len(train_data)))
  random.shuffle(index_list)
  #selected_list = index_list[: min(200, len(index_list))]
  #selected_list = [7260, 11506, 39952, 75196, 51614, 41061, 37747, 41355]
  #for iidx, i in enumerate(selected_list):
  index_list.remove(47875)
  selected_list = [47875] + index_list
  save_xdir  = logger.path('meta')

  type_error_1, type_error_2, type_error, misses = 0, 0, 0, 0
  type_error_pts, total_pts = 0, 0
  for iidx, i in enumerate(selected_list):
    frames, Fflows, Bflows, targets, masks, normpoints, transthetas, meanthetas, image_index, nopoints, shapes, is_images = train_data[i]
  
    frames, Fflows, Bflows, is_images = frames.unsqueeze(0), Fflows.unsqueeze(0), Bflows.unsqueeze(0), is_images.unsqueeze(0)
    # batch_heatmaps is a list for stage-predictions, each element should be [Batch, Sequence, PTS, H/Down, W/Down]
    with torch.no_grad():
      if args.procedure == 'heatmap':
        batch_heatmaps, batch_locs, batch_scos, batch_past2now, batch_future2now, batch_FBcheck = net(frames, Fflows, Bflows, is_images)
      else:
        batch_locs, batch_past2now, batch_future2now, batch_FBcheck = net(frames, Fflows, Bflows, is_images)

    (batch_size, frame_length, C, H, W), num_pts, annotate_index = frames.size(), args.num_pts, train_data.video_L
    batch_locs = batch_locs.cpu()[:, :, :num_pts]
    video_mask = masks.unsqueeze(0)[:, :num_pts]
    batch_past2now   = batch_past2now.cpu()[:, :, :num_pts]
    batch_future2now = batch_future2now.cpu()[:, :, :num_pts]
    batch_FBcheck    = batch_FBcheck[:, :num_pts].cpu()
    FB_check_oks  = FB_communication(criterion, batch_locs, batch_past2now, batch_future2now, batch_FBcheck, video_mask, sbr_config)
    
    # locations
    norm_past_det_locs  = torch.cat((batch_locs[0,annotate_index-1,:num_pts].permute(1,0), torch.ones(1, num_pts)), dim=0)
    norm_noww_det_locs  = torch.cat((batch_locs[0,annotate_index  ,:num_pts].permute(1,0), torch.ones(1, num_pts)), dim=0)
    norm_next_det_locs  = torch.cat((batch_locs[0,annotate_index+1,:num_pts].permute(1,0), torch.ones(1, num_pts)), dim=0)
    norm_next_locs      = torch.cat((batch_past2now[0,annotate_index,:num_pts].permute(1,0), torch.ones(1, num_pts)), dim=0)
    norm_past_locs      = torch.cat((batch_future2now[0,annotate_index-1,:num_pts].permute(1,0), torch.ones(1, num_pts)), dim=0)
    transtheta = transthetas[:2,:]
    norm_past_det_locs = torch.mm(transtheta, norm_past_det_locs)
    norm_noww_det_locs = torch.mm(transtheta, norm_noww_det_locs)
    norm_next_det_locs = torch.mm(transtheta, norm_next_det_locs)
    norm_next_locs     = torch.mm(transtheta, norm_next_locs)
    norm_past_locs     = torch.mm(transtheta, norm_past_locs)
    real_past_det_locs = denormalize_points(shapes.tolist(), norm_past_det_locs)
    real_noww_det_locs = denormalize_points(shapes.tolist(), norm_noww_det_locs)
    real_next_det_locs = denormalize_points(shapes.tolist(), norm_next_det_locs)
    real_next_locs     = denormalize_points(shapes.tolist(), norm_next_locs)
    real_past_locs     = denormalize_points(shapes.tolist(), norm_past_locs)
    gt_noww_points     = train_data.labels[image_index.item()].get_points()
    gt_past_points     = train_data.find_index( train_data.datas[image_index.item()][annotate_index-1] )
    gt_next_points     = train_data.find_index( train_data.datas[image_index.item()][annotate_index+1] )
    
    FB_check_oks = FB_check_oks[:num_pts].squeeze()
    #import pdb; pdb.set_trace()
    if FB_check_oks.sum().item() > 2:
      # type 1 error : detection at both (t) and (t-1) is wrong, while pass the check
      is_type_1, (T_wrong, T_total) = check_is_1st_error([real_past_det_locs,real_noww_det_locs,real_next_det_locs], 
                                     [gt_past_points,    gt_noww_points    ,gt_next_points], FB_check_oks, shapes)
      # type 2 error : detection at frame t is ok, while tracking are wrong and frame at (t-1) is wrong:
      spec_index, is_type_2 = check_is_2nd_error(real_noww_det_locs, gt_noww_points, [real_past_locs,real_next_locs], [gt_past_points,gt_next_points], FB_check_oks, shapes)
      type_error_1 += is_type_1
      type_error_2 += is_type_2
      type_error   += is_type_1 or is_type_2
      type_error_pts, total_pts = type_error_pts + T_wrong, total_pts + T_total
      if is_type_2:
        RED, GREEN, BLUE = (255, 0,   0), (0, 255,   0), (0,   0, 255)
        [image_past, image_noww, image_next] = train_data.datas[image_index.item()]
        crop_box = train_data.labels[image_index.item()].get_box().tolist()
        point_index = FB_check_oks.nonzero().squeeze().tolist()
        colors = [ GREEN if _i in point_index else RED for _i in range(num_pts)] + [BLUE for _i in range(num_pts)]
        
        I_past_det = draw_image_by_points(image_past, torch.cat((real_past_det_locs, gt_past_points[:2]), dim=1), 3, colors, crop_box, (400,500))
        I_noww_det = draw_image_by_points(image_noww, torch.cat((real_noww_det_locs, gt_noww_points[:2]), dim=1), 3, colors, crop_box, (400,500))
        I_next_det = draw_image_by_points(image_next, torch.cat((real_next_det_locs, gt_next_points[:2]), dim=1), 3, colors, crop_box, (400,500))
        I_past     = draw_image_by_points(image_past, torch.cat((real_past_locs    , gt_past_points[:2]), dim=1), 3, colors, crop_box, (400,500))
        I_next     = draw_image_by_points(image_next, torch.cat((real_next_locs    , gt_next_points[:2]), dim=1), 3, colors, crop_box, (400,500))
        ### 
        I_past.save    ( str(save_xdir / '{:05d}-v1-a-pastt.png'.format(i)) )
        I_noww_det.save( str(save_xdir / '{:05d}-v1-b-curre.png'.format(i)) )
        I_next.save    ( str(save_xdir / '{:05d}-v1-c-nextt.png'.format(i)) )

        I_past_det.save( str(save_xdir / '{:05d}-v1-det-a-past.png'.format(i)) )
        I_noww_det.save( str(save_xdir / '{:05d}-v1-det-b-curr.png'.format(i)) )
        I_next_det.save( str(save_xdir / '{:05d}-v1-det-c-next.png'.format(i)) )

        logger.log('TYPE-ERROR : {:}, landmark-index : {:}'.format(i, spec_index))
    else:
      misses += 1
    string = 'Handle {:05d}/{:05d} :: {:05d}'.format(iidx, len(selected_list), i)
    string+= ', error-1 : {:} ({:.2f}%), error-2 : {:} ({:.2f}%)'.format(type_error_1, type_error_1*100.0/(iidx+1), type_error_2, type_error_2*100.0/(iidx+1))
    string+= ', error : {:} ({:.2f}%), miss : {:}'.format(type_error, type_error*100.0/(iidx+1), misses)
    string+= ', final-error : {:05d} / {:05d} = {:.2f}%'.format(type_error_pts, total_pts, type_error_pts * 100.0 / total_pts)
    logger.log(string)


def check_is_1st_error(detections, GT_points, selects, shape):
  detections = [normalize_points(shape.tolist(), x[:2]) for x in detections]
  GT_points  = [normalize_points(shape.tolist(), x[:2]) for x in GT_points ]
  detections = [x[:, selects] for x in detections]
  GT_points  = [x[:, selects] for x in GT_points]

  all_errors = []
  for det, gt in zip(detections, GT_points):
    #error, xis = get_error(det, gt, None, 0.05)
    is_errors = get_error_v2(det, gt, None, 0.05)
    all_errors.append( is_errors )
  all_errors = np.array(all_errors)
  temp = all_errors.sum(0) > 0
  return all_errors.any(), (temp.sum(), temp.size)


def get_error_v2(pointsA, pointsB, indexes, threshold):
  if indexes is not None:
    pointsA, pointsB = pointsA[:2, FB_check_oks], pointsB[:2, FB_check_oks]
  else:
    pointsA, pointsB = pointsA[:2], pointsB[:2]
  pointsA, pointsB = pointsA.numpy(), pointsB.numpy()
  is_errors = []
  for j in range(pointsA.shape[1]):
    distance = np.linalg.norm(pointsA[:,j] - pointsB[:,j])
    if distance > threshold: is_errors.append( True )
    else                   : is_errors.append( False )
  return is_errors


def check_is_2nd_error(det, GT, tracks, GTracks, selects, shape):
  det = normalize_points(shape.tolist(), det[:2,selects])
  GT  = normalize_points(shape.tolist(), GT [:2,selects])
  erroe_det, xis = get_error(det, GT, None, 0.005)
  tracks  = [normalize_points(shape.tolist(), x[:2,selects]) for x in tracks ]
  GTracks = [normalize_points(shape.tolist(), x[:2,selects]) for x in GTracks]
  error_A, xis_A = get_error(tracks[0], GTracks[0], None, 0.05)
  error_B, xis_B = get_error(tracks[1], GTracks[1], None, 0.05)
  for i, (er_det, er_A, er_B) in enumerate(zip(erroe_det, error_A, error_B)):
    if (er_det < 0.005) and (er_A > 0.05 or er_B > 0.05):
      return selects[i].item(), True
  return -1, False
  #return xis and (xis_A or xis_B)
  

def get_error(pointsA, pointsB, indexes, threshold):
  if indexes is not None:
    pointsA, pointsB = pointsA[:2, FB_check_oks], pointsB[:2, FB_check_oks]
  else:
    pointsA, pointsB = pointsA[:2], pointsB[:2]
  pointsA, pointsB = pointsA.numpy(), pointsB.numpy()
  disss, is_error = [], False
  for j in range(pointsA.shape[1]):
    distance = np.linalg.norm(pointsA[:,j] - pointsB[:,j])
    if distance > threshold: is_error = True
    disss.append( float(distance) )
  return disss, is_error


if __name__ == '__main__':
  args = obtain_args()
  main(args)
