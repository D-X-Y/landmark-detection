##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import os, torch
import utils

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
  assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
  multiple = 1
  for (gamma, step) in zip(gammas, schedule):
    if (epoch == step):
      multiple = gamma
      break
  all_lrs = []
  for param_group in optimizer.param_groups:
    param_group['lr'] = multiple * param_group['lr']
    all_lrs.append( param_group['lr'] )
  return set(all_lrs)

def save_checkpoint(state, save_path, filename, log):
  filename = os.path.join(save_path, filename)
  torch.save(state, filename)
  utils.print_log ('save checkpoint into {}'.format(filename), log)
  return filename
