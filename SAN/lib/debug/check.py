import torch
import numpy as np

def tocpudata(x):
  if x.is_cuda: return x.cpu().data
  else:         return x.data

def tonp(x):
  if x.is_cuda: return x.cpu().data.numpy()
  else:         return x.data.numpy()

def register_nan_checks(model):
  def check_grad(module, grad_input, grad_output):
    # print(module) you can add this to see that the hook is called
    if any(np.all(np.isnan(tonp(gi))) for gi in grad_input if gi is not None):
      print('NaN gradient in ' + type(module).__name__)
  model.apply(lambda module: module.register_backward_hook(check_grad))

def check_data(dataset):
  length = len(dataset)
  for i, data in enumerate(dataset):
    if i + 1 >= length:
      nxt = 'none'
    else:
      nxt = dataset.datas[i+1]
    print (' {:05d} / {:05d} : {:} -->> {:}'.format(i, length, dataset.datas[i], nxt))
