from scipy.ndimage.interpolation import zoom
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, numbers, numpy as np

def np2variable(x, is_cuda=True, requires_grad=True, dtype=torch.FloatTensor):
  if isinstance(x, np.ndarray):
    v = torch.autograd.Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
  elif isinstance(x, torch.FloatTensor):
    v = torch.autograd.Variable(x.type(dtype), requires_grad=requires_grad)
  else:
    raise Exception('Do not know this type : {}'.format( type(x) ))

  if is_cuda: return v.cuda()
  else:       return v

def variable2np(x):
  if x.is_cuda:
    x = x.cpu()
  if isinstance(x, torch.autograd.Variable):
    return x.data.numpy()
  else:
    return x.numpy()

def get_parameters(model, bias):
  for m in model.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      if bias:
        yield m.bias
      else:
        yield m.weight
    elif isinstance(m, nn.BatchNorm2d):
      if bias:
        yield m.bias
      else:
        yield m.weight

def load_weight_from_dict(model, weight_state_dict, param_pair=None, remove_prefix=True):
  if remove_prefix: weight_state_dict = remove_module_dict(weight_state_dict)
  all_parameter = model.state_dict()
  all_weights   = []
  finetuned_layer, random_initial_layer = [], []
  for key, value in all_parameter.items():
    if param_pair is not None and key in param_pair:
      all_weights.append((key, weight_state_dict[ param_pair[key] ]))
    elif key in weight_state_dict:
      all_weights.append((key, weight_state_dict[key]))
      finetuned_layer.append(key)
    else:
      all_weights.append((key, value))
      random_initial_layer.append(key)
  print ('==>[load_model] finetuned layers : {}'.format(finetuned_layer))
  print ('==>[load_model] keeped layers : {}'.format(random_initial_layer))
  all_weights = OrderedDict(all_weights)
  model.load_state_dict(all_weights)

def remove_module_dict(state_dict):
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
  return new_state_dict

def count_network_param(net):
  num_params = 0
  for param in net.parameters():
    num_params += param.numel()
  return num_params
