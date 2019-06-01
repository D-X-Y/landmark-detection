# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path
import importlib, warnings
import os, sys, time, numpy as np
import scipy.misc 
if sys.version_info.major == 2: # Python 2.x
  from StringIO import StringIO as BIO
else:                           # Python 3.x
  from io import BytesIO as BIO

class Logger(object):
  
  def __init__(self, log_dir, logstr):
    """Create a summary writer logging to log_dir."""
    self.log_dir = Path(log_dir)
    self.model_dir = Path(log_dir) / 'checkpoint'
    self.meta_dir = Path(log_dir) / 'metas'
    self.log_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
    self.model_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
    self.meta_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

    self.logger_path = self.log_dir / '{:}.log'.format(logstr)
    self.logger_file = open(self.logger_path, 'w')


  def __repr__(self):
    return ('{name}(dir={log_dir})'.format(name=self.__class__.__name__, **self.__dict__))

  def path(self, mode):
    if mode == 'meta'   : return self.meta_dir
    elif mode == 'model': return self.model_dir
    elif mode == 'log'  : return self.log_dir
    else: raise TypeError('Unknow mode = {:}'.format(mode))

  def last_info(self):
    return self.log_dir / 'last-info.pth'

  def extract_log(self):
    return self.logger_file

  def close(self):
    self.logger_file.close()

  def log(self, string, save=True):
    print (string)
    if save:
      self.logger_file.write('{:}\n'.format(string))
      self.logger_file.flush()
