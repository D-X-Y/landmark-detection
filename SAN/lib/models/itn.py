import os, itertools, torch, numpy as np
from collections import OrderedDict
from os import path as osp
import utils
from utils.image_pool import ImagePool
from .generator_model import ResnetGenerator
from .discriminator_model import NLayerDiscriminator
from .gan_loss import GANLoss
from .cycle_util import get_scheduler, save_network, load_network, tensor2im
from .model_utils import print_network
from .initialization import weights_init_xavier

def define_G(gpu_ids=[]):
  netG = ResnetGenerator(gpu_ids=gpu_ids)
  if len(gpu_ids) > 0:
    netG.cuda(gpu_ids[0])
  netG.apply(weights_init_xavier)
  return netG

def define_D(gpu_ids=[]):
  netD = NLayerDiscriminator(use_sigmoid=False, gpu_ids=gpu_ids)
  if len(gpu_ids) > 0:
    netD.cuda(gpu_ids[0])
  netD.apply(weights_init_xavier)
  return netD

class ITN():
  def __repr__(self):
    return ('{name})'.format(name=self.__class__.__name__, **self.__dict__))

  def initialize(self, opt, log):
    self.opt = opt
    self.gpu_ids = opt.gpu_ids
    self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

    nb = opt.cycle_batchSize
    crop_height, crop_width = opt.crop_height, opt.crop_width
    self.input_A = self.Tensor(nb, 3, crop_height, crop_width)
    self.input_B = self.Tensor(nb, 3, crop_height, crop_width)

    # load/define networks
    # The naming conversion is different from those used in the paper
    # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

    self.netG_A = define_G(gpu_ids=self.gpu_ids)
    self.netG_B = define_G(gpu_ids=self.gpu_ids)

    self.netD_A = define_D(gpu_ids=self.gpu_ids)
    self.netD_B = define_D(gpu_ids=self.gpu_ids)

    # for training 
    self.fake_A_pool = ImagePool(opt.pool_size)
    self.fake_B_pool = ImagePool(opt.pool_size)
    # define loss functions
    self.criterionGAN = GANLoss(use_lsgan=True, tensor=self.Tensor)
    self.criterionCycle = torch.nn.L1Loss()
    self.criterionIdt = torch.nn.L1Loss()
    # initialize optimizers
    self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                        lr=opt.cycle_lr, betas=(opt.cycle_beta1, 0.999))
    self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.cycle_lr, betas=(opt.cycle_beta1, 0.999))
    self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.cycle_lr, betas=(opt.cycle_beta1, 0.999))
    self.optimizers = []
    self.schedulers = []
    self.optimizers.append(self.optimizer_G)
    self.optimizers.append(self.optimizer_D_A)
    self.optimizers.append(self.optimizer_D_B)
    for optimizer in self.optimizers:
      self.schedulers.append(get_scheduler(optimizer, opt))

    utils.print_log('------------ Networks initialized -------------', log)
    print_network(self.netG_A, 'netG_A', log)
    print_network(self.netG_B, 'netG_B', log)
    print_network(self.netD_A, 'netD_A', log)
    print_network(self.netD_B, 'netD_B', log)
    utils.print_log('-----------------------------------------------', log)

  def set_mode(self, mode):
    if mode.lower() == 'train':
      self.netG_A.train()
      self.netG_B.train()
      self.netD_A.train()
      self.netD_B.train()
      self.criterionGAN.train()
      self.criterionCycle.train()
      self.criterionIdt.train()
    elif mode.lower() == 'eval':
      self.netG_A.eval()
      self.netG_B.eval()
      self.netD_A.eval()
      self.netD_B.eval()
    else:
      raise NameError('The wrong mode : {}'.format(mode))

  def set_input(self, input):
    input_A = input['A']
    input_B = input['B']
    self.input_A.resize_(input_A.size()).copy_(input_A)
    self.input_B.resize_(input_B.size()).copy_(input_B)

  def prepaer_input(self):
    self.real_A = torch.autograd.Variable(self.input_A)
    self.real_B = torch.autograd.Variable(self.input_B)

  def test(self):
    self.real_A = torch.autograd.Variable(self.input_A, volatile=True)
    self.fake_B = self.netG_A.forward(self.real_A)
    self.rec_A = self.netG_B.forward(self.fake_B)

    self.real_B = torch.autograd.Variable(self.input_B, volatile=True)
    self.fake_A = self.netG_B.forward(self.real_B)
    self.rec_B = self.netG_A.forward(self.fake_A)

  def backward_D_basic(self, netD, real, fake):
    # Real
    pred_real = netD.forward(real)
    loss_D_real = self.criterionGAN(pred_real, True)
    # Fake
    pred_fake = netD.forward(fake.detach())
    loss_D_fake = self.criterionGAN(pred_fake, False)
    # Combined loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    # backward
    loss_D.backward()
    return loss_D

  def backward_D_A(self):
    fake_B = self.fake_B_pool.query(self.fake_B)
    self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

  def backward_D_B(self):
    fake_A = self.fake_A_pool.query(self.fake_A)
    self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

  def backward_G(self):
    lambda_idt = self.opt.identity
    lambda_A = self.opt.lambda_A
    lambda_B = self.opt.lambda_B
    # Identity loss
    if lambda_idt > 0:
      # G_A should be identity if real_B is fed.
      self.idt_A = self.netG_A.forward(self.real_B)
      self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
      # G_B should be identity if real_A is fed.
      self.idt_B = self.netG_B.forward(self.real_A)
      self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
    else:
      self.loss_idt_A = 0
      self.loss_idt_B = 0

    # GAN loss
    # D_A(G_A(A))
    self.fake_B = self.netG_A.forward(self.real_A)
    pred_fake = self.netD_A.forward(self.fake_B)
    self.loss_G_A = self.criterionGAN(pred_fake, True)
    # D_B(G_B(B))
    self.fake_A = self.netG_B.forward(self.real_B)
    pred_fake = self.netD_B.forward(self.fake_A)
    self.loss_G_B = self.criterionGAN(pred_fake, True)
    # Forward cycle loss
    self.rec_A = self.netG_B.forward(self.fake_B)
    self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
    # Backward cycle loss
    self.rec_B = self.netG_A.forward(self.fake_A)
    self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
    # combined loss
    self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
    self.loss_G.backward()

  def optimize_parameters(self):
    # forward
    self.prepaer_input()
    # G_A and G_B
    self.optimizer_G.zero_grad()
    self.backward_G()
    self.optimizer_G.step()
    # D_A
    self.optimizer_D_A.zero_grad()
    self.backward_D_A()
    self.optimizer_D_A.step()
    # D_B
    self.optimizer_D_B.zero_grad()
    self.backward_D_B()
    self.optimizer_D_B.step()

  def get_current_errors(self):
    D_A = self.loss_D_A.item()
    G_A = self.loss_G_A.item()
    Cyc_A = self.loss_cycle_A.item()
    D_B = self.loss_D_B.item()
    G_B = self.loss_G_B.item()
    Cyc_B = self.loss_cycle_B.item()
    if self.opt.identity > 0.0:
      idt_A = self.loss_idt_A.item()
      idt_B = self.loss_idt_B.item()
      return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
    else:
      return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B)])

  def get_current_visuals(self, isTrain):
    real_A = tensor2im(self.real_A.data)
    rec_A = tensor2im(self.rec_A.data)
    fake_A = tensor2im(self.fake_A.data)

    real_B = tensor2im(self.real_B.data)
    rec_B = tensor2im(self.rec_B.data)
    fake_B = tensor2im(self.fake_B.data)

    if isTrain and self.opt.identity > 0.0:
      idt_A = tensor2im(self.idt_A.data)
      idt_B = tensor2im(self.idt_B.data)
      return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
    else:
      return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

  def save(self, save_dir, log):
    save_network(save_dir, 'G_A', self.netG_A, self.gpu_ids)
    save_network(save_dir, 'D_A', self.netD_A, self.gpu_ids)
    save_network(save_dir, 'G_B', self.netG_B, self.gpu_ids)
    save_network(save_dir, 'D_B', self.netD_B, self.gpu_ids)
    utils.print_log('save the model into {}'.format(save_dir), log)

  def load(self, save_dir, log):
    load_network(save_dir, 'G_A', self.netG_A)
    load_network(save_dir, 'D_A', self.netD_A)
    load_network(save_dir, 'G_B', self.netG_B)
    load_network(save_dir, 'D_B', self.netD_B)
    utils.print_log('load the model from {}'.format(save_dir), log)

  # update learning rate (called once every epoch)
  def update_learning_rate(self, log):
    for scheduler in self.schedulers:
      scheduler.step()
    lr = self.optimizers[0].param_groups[0]['lr']
    utils.print_log('learning rate = {:.7f}'.format(lr), log)

def itn_model(model_config, opt, log):
  itnetwork = ITN()
  itnetwork.initialize(opt, log)
  return itnetwork
