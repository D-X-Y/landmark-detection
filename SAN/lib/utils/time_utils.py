##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import time, sys
import numpy as np

def time_for_file():
  ISOTIMEFORMAT='%d-%h-at-%H-%M-%S'
  return '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def time_string_short():
  ISOTIMEFORMAT='%Y%m%d'
  string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def time_print(string, is_print=True):
  if (is_print):
    print('{} : {}'.format(time_string(), string))

class AverageMeter(object):     
  """Computes and stores the average and current value"""    
  def __init__(self):   
    self.reset()
  
  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0    
  
  def update(self, val, n=1): 
    self.val = val    
    self.sum += val * n     
    self.count += n
    self.avg = self.sum / self.count    

class LossRecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses + sys.float_info.max

  def update(self, train_loss, idx, val_loss=None):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses[idx, 0] = train_loss
    if val_loss is not None:
      self.epoch_losses[idx, 1] = val_loss
    self.current_epoch = idx + 1
  
  def min_loss(self, Train=True):
    if Train:
      idx = np.argmin(self.epoch_losses[:self.current_epoch, 0])
      return idx, self.epoch_losses[idx, 0]
    else:
      idx = np.argmin(self.epoch_losses[:self.current_epoch, 1])
      if self.epoch_losses[idx, 1] >= sys.float_info.max / 10:
        return idx, -1.
      else:
        return idx, self.epoch_losses[idx, 1]
    
def convert_size2str(torch_size):
  dims = len(torch_size)
  string = '['
  for idim in range(dims):
    string = string + ' {}'.format(torch_size[idim])
  return string + ']'
  
def convert_secs2time(epoch_time):    
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)  
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

def print_log(print_string, log):
  print("{}".format(print_string))
  if log is not None:
    log.write('{}\n'.format(print_string))
    log.flush()
