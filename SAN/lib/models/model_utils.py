from scipy.ndimage.interpolation import zoom
from collections import OrderedDict
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, numbers, numpy as np

def np2variable(x, is_cuda=True, requires_grad=True, dtype=torch.FloatTensor):
  if isinstance(x, np.ndarray):
    v = torch.autograd.Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
  elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.Tensor):
    v = torch.autograd.Variable(x.type(dtype), requires_grad=requires_grad)
  else:
    raise Exception('Do not know this type : {:}'.format( type(x) ))

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

def roi_pooling(input, rois, size=(7,7)):
  assert rois.dim() == 2 and rois.size(1) == 5, 'rois shape is wrong : {}'.format(rois.size())
  output = []
  num_rois = rois.size(0)
  size = np.array(size)
  spatial_size = np.array([input.size(3), input.size(2)])
  for i in range(num_rois):
    roi = variable2np(rois[i])
    im_idx = int(roi[0])
    theta = utils.crop2affine(spatial_size, roi[1:])
    theta = np2variable(theta, input.is_cuda).unsqueeze(0)
    grid_size = torch.Size([1, 3, int(size[1]), int(size[0])])
    grid = F.affine_grid(theta, grid_size)
    roi_feature = F.grid_sample(input.narrow(0, im_idx, 1), grid)
    output.append( roi_feature )
  return torch.cat(output, 0)

class ModelConfig():
  def __init__(self, pts_num, num_stages, pretrained, softargmax_patch):
    assert isinstance(pts_num, int), 'The pts-num is not right : {}'.format(pts_num)
    assert isinstance(num_stages, int), 'The stage-num is not right : {}'.format(num_stages)
    assert isinstance(pretrained, bool), 'The format of pretrained is not right : {}'.format(pretrained)
    assert isinstance(softargmax_patch, numbers.Number), 'The format of softargmax_patch is not right : {}'.format(softargmax_patch)

    self.pts_num = pts_num
    self.num_stages = num_stages
    self.pretrained = pretrained
    self.argmax = softargmax_patch
    
  def __repr__(self):
    return ('{name}(points={pts_num}, stage={num_stages}, PreTrain={pretrained}, ArgMax={argmax})'.format(name=self.__class__.__name__, **self.__dict__))

  def copy(self):
    return copy.deepcopy(self)

def print_network(net, net_str, log):
  num_params = 0
  for param in net.parameters():
    num_params += param.numel()
  utils.print_log(net, log)
  utils.print_log('Total number of parameters for {} is {}'.format(net_str, num_params), log)

def count_network_param(net):
  num_params = 0
  for param in net.parameters():
    num_params += param.numel()
  return num_params
