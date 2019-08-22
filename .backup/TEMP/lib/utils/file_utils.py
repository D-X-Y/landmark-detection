# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, sys, glob, numbers
from os import path as osp

def mkdir_if_missing(path):
  if not osp.isdir(path):
    os.makedirs(path)

def is_path_exists(pathname):                                                                                                                                                                          
  try:
    return isinstance(pathname, str) and pathname and os.path.exists(pathname) 
  except OSError:
    return False

def fileparts(pathname):
  '''
  this function return a tuple, which contains (directory, filename, extension)
  if the file has multiple extension, only last one will be displayed
  '''
  pathname = osp.normpath(pathname)
  if len(pathname) == 0:
    return ('', '', '')
  if pathname[-1] == '/':
    if len(pathname) > 1:
      return (pathname[:-1], '', '')  # ignore the final '/'
    else:
      return (pathname, '', '') # ignore the final '/'
  directory = osp.dirname(osp.abspath(pathname))
  filename  = osp.splitext(osp.basename(pathname))[0]
  ext       = osp.splitext(pathname)[1]
  return (directory, filename, ext)

def load_txt_file(file_path):
  '''
  load data or string from text file.
  '''
  with open(file_path, 'r') as cfile:
    content = cfile.readlines()
  cfile.close()
  content = [x.strip() for x in content]
  num_lines = len(content)
  return content, num_lines

def load_list_from_folder(folder_path, ext_filter=None, depth=1):
  '''
  load a list of files or folders from a system path

  parameter:
    folder_path: root to search 
    ext_filter: a string to represent the extension of files interested
    depth: maximum depth of folder to search, when it's None, all levels of folders will be searched
  '''
  folder_path = osp.normpath(folder_path)
  assert isinstance(depth, int) , 'input depth is not correct {}'.format(depth)
  assert ext_filter is None or (isinstance(ext_filter, list) and all(isinstance(ext_tmp, str) for ext_tmp in ext_filter)) or isinstance(ext_filter, str), 'extension filter is not correct'
  if isinstance(ext_filter, str):    # convert to a list
    ext_filter = [ext_filter]

  fulllist = list()
  wildcard_prefix = '*'
  for index in range(depth):
    if ext_filter is not None:
      for ext_tmp in ext_filter:
        curpath = osp.join(folder_path, wildcard_prefix + '.' + ext_tmp)
        fulllist += glob.glob(curpath)
    else:
      curpath = osp.join(folder_path, wildcard_prefix)
      fulllist += glob.glob(curpath)
    wildcard_prefix = osp.join(wildcard_prefix, '*')

  fulllist = [osp.normpath(path_tmp) for path_tmp in fulllist]
  num_elem = len(fulllist)

  return fulllist, num_elem

def load_list_from_folders(folder_path_list, ext_filter=None, depth=1):
  '''
  load a list of files or folders from a list of system path
  '''
  assert isinstance(folder_path_list, list) or isinstance(folder_path_list, str), 'input path list is not correct'
  if isinstance(folder_path_list, str):
    folder_path_list = [folder_path_list]

  fulllist = list()
  num_elem = 0
  for folder_path_tmp in folder_path_list:
    fulllist_tmp, num_elem_tmp = load_list_from_folder(folder_path_tmp, ext_filter=ext_filter, depth=depth)
    fulllist += fulllist_tmp
    num_elem += num_elem_tmp

  return fulllist, num_elem
