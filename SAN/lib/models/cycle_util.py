import os
import os.path as osp
import torch
from torch.optim import lr_scheduler
from san_vision import transforms

def get_scheduler(optimizer, opt):
  if opt.lr_policy == 'lambda':
    def lambda_rule(epoch):
      lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
  elif opt.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
  elif opt.lr_policy == 'plateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
  else:
    return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
  return scheduler

def save_network(save_dir, save_name, network, gpu_ids):
  if not osp.isdir(save_dir): os.makedirs(save_dir)
  save_path = osp.join(save_dir, '{}.pth'.format(save_name))
  torch.save(network.cpu().state_dict(), save_path)
  if len(gpu_ids) and torch.cuda.is_available():
    network.cuda(gpu_ids[0])

def load_network(save_dir, save_name, network):
  save_path = osp.join(save_dir, '{}.pth'.format(save_name))
  assert osp.isfile(save_path), '{} does not exist'.format(save_path)
  network.load_state_dict(torch.load(save_path))

def tensor2im(image_tensor):
  mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  # only extract the first image
  image_tensor = image_tensor[0]
  Xinput = []
  for t, m, s in zip(image_tensor, mean, std):
    t = torch.mul(t, s)
    t = torch.add(t, m)
    Xinput.append( t )
  xinput = torch.stack(Xinput)
  if xinput.is_cuda: xinput = xinput.cpu()
  image = transforms.ToPILImage()(xinput)
  return image
