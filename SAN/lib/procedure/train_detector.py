##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import os, copy
import os.path as osp
import torch
import time, numpy as np
from utils import AverageMeter, print_log, convert_size2str, convert_secs2time, time_string, time_for_file
from .san_util import adjust_learning_rate, save_checkpoint
from san_vision import Eval_Meta, show_stage_loss, compute_stage_loss
from debug import main_debug_save
from models import variable2np
from .evaluate_detector import evaluation

def train_san_epoch(opt, net, train_loader, eval_loaders, log):

  print_log('The ITN-CPM Network is : {}'.format(net), log)
  print_log('Train ITN-CPM using LR={:.6f}, Decay={:.6f}'.format(opt.learning_rate, opt.decay), log)
  # obtain the parameters
  if hasattr(net, 'specify_parameter'):
    net_param_dict = net.specify_parameter(opt.learning_rate, opt.decay)
  else:
    net_param_dict = net.parameters()
  # define loss function
  criterion = torch.nn.MSELoss(False)
  if opt.use_cuda:
    net = torch.nn.DataParallel(net, device_ids=opt.gpu_ids)
    net.cuda()
    criterion.cuda()
  # define optimizer
  optimizer = torch.optim.SGD(net_param_dict, lr=opt.learning_rate, momentum=opt.momentum,
                              weight_decay=opt.decay, nesterov=True)
  if opt.resume:
    assert osp.isfile(opt.resume), 'The resume file is not here : {}'.format(opt.resume)
    print_log("=> loading checkpoint '{}' start".format(opt.resume), log)
    checkpoint       = torch.load(opt.resume)
    net.load_state_dict(checkpoint['state_dict'])
    if opt.pure_resume == False:
      opt.start_epoch = checkpoint['epoch']
      optimizer.load_state_dict(checkpoint['optimizer'])
    else:
      print_log('pure resume, only load the parameters, skip optimizer and start-epoch', log)
    if opt.eval_once:
      evaluation(eval_loaders, net, log, osp.join(opt.save_path, 'epoch-once'), opt)
      return
      


  # Main Training and Evaluation Loop
  start_time, epoch_time = time.time(), AverageMeter()
  
  for epoch in range(opt.start_epoch, opt.epochs):
    all_lrs = adjust_learning_rate(optimizer, epoch, opt.gammas, opt.schedule)

    need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (opt.epochs-epoch))

    print_log('\n==>>{:s} Epoch[{:03d}/{:03d}], [Time Left: {:02d}:{:02d}:{:02d}], learning_rate : [{:.8f}, {:.8f}], Identity-Weight={:}'.format(time_string(), epoch, opt.epochs, need_hour, need_mins, need_secs, min(all_lrs), max(all_lrs), opt.weight_of_idt), log)

    # train for one epoch
    train_loss = train(train_loader, net, criterion, optimizer, epoch, opt, log)
    print_log('==>>{:s} Average Loss for Epoch[{:03d}] total=[{:.5f}] '.format(time_string(), epoch, train_loss), log)

    # remember best prec@1 and save checkpoint
    save_name = save_checkpoint({
          'epoch': epoch,
          'args' : copy.deepcopy(opt),
          'state_dict': net.state_dict(),
          'optimizer' : optimizer.state_dict(),
          }, opt.save_path, 'checkpoint_{}.pth.tar'.format(epoch), log)

    # evaluate
    if opt.evaluation:
      print_log('===>>{:s} Evaluate epoch [{:03d}] '.format(time_string(), epoch), log)
      evaluation(eval_loaders, net, log, osp.join(opt.save_path, 'epoch-{:03d}'.format(epoch)), opt)

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

# train function (forward, backward, update)
def train(train_loader, net, criterion, optimizer, epoch, opt, log):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  forward_time = AverageMeter()
  visible_points = AverageMeter()
  losses = AverageMeter()
  # switch to train mode
  net.module.set_mode('train')
  criterion.train()

  if opt.debug_save:
    debug_save_dir = os.path.join(opt.save_path, 'debug-train-{:02d}'.format(epoch))
    if not os.path.isdir(debug_save_dir): os.makedirs(debug_save_dir)

  end = time.time()
  for i, (inputs, target, mask, points, image_index, label_sign, _) in enumerate(train_loader):
    # inputs : Batch, Squence, Channel, Height, Width
    # data prepare
    target = target.cuda(async=True)
    # get the real mask
    mask.masked_scatter_((1-label_sign).unsqueeze(-1).unsqueeze(-1), torch.ByteTensor(mask.size()).zero_())
    mask_var   = mask.cuda(async=True)

    batch_size, num_pts = inputs.size(0), mask.size(1)-1
    image_index = variable2np(image_index).squeeze(1).tolist()
    # check the label indicator, whether is has annotation or not
    sign_list = variable2np(label_sign).astype('bool').squeeze(1).tolist()

    # measure data loading time
    data_time.update(time.time() - end)
    cvisible_points = torch.sum(mask[:,:-1,:,:]) * 1. / batch_size
    visible_points.update(cvisible_points, batch_size)

    # batch_cpms is a list for CPM stage-predictions, each element should be [Batch, Squence, C, H, W]
    # batch_locs and batch_scos are two sequence-list of point-list, each element is [Batch, 2] / [Batch, 1]
    # batch_next and batch_back are two sequence-list of point-list, each element is [Batch, 2] / [Batch, 2]
    batch_cpms, batch_locs, batch_scos, generated = net(inputs)
    
    forward_time.update(time.time() - end)

    total_labeled_cpm = int(np.sum(sign_list))

    # collect all cpm stages for the middle frame
    cpm_loss, each_stage_loss_values = compute_stage_loss(criterion, target, batch_cpms, mask_var, total_labeled_cpm, opt.weight_of_idt)
  
    # measure accuracy and record loss
    losses.update(cpm_loss.item(), batch_size)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    cpm_loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    need_hour, need_mins, need_secs = convert_secs2time(batch_time.avg * (len(train_loader)-i-1))
    last_time = '[{:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
    end = time.time()

    if i % opt.print_freq == 0 or i+1 == len(train_loader):
      print_log('  Epoch: [{:03d}/{:03d}][{:03d}/{:03d}]  '
                'Time {batch_time.val:5.2f} ({batch_time.avg:5.2f}) '
                'Data {data_time.val:5.2f} ({data_time.avg:5.2f}) '
                'Forward {forward_time.val:5.2f} ({forward_time.avg:5.2f}) '
                'Loss {loss.val:6.3f} ({loss.avg:6.3f})  '.format(
                    epoch, opt.epochs, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, forward_time=forward_time, loss=losses)
                  + last_time + show_stage_loss(each_stage_loss_values) \
                  + ' In={} Tar={} Mask={}'.format(convert_size2str(inputs.size()), convert_size2str(target.size()), convert_size2str(mask_var.size())) \
                  + ' Vis-PTS : {:2d} ({:.1f})'.format(int(visible_points.val), visible_points.avg), log)

    # Just for debug and show the intermediate results.
    if opt.debug_save:
      print_log('DEBUG --- > [{:03d}/{:03d}] '.format(i, len(train_loader)), log)
      main_debug_save(debug_save_dir, train_loader, image_index, inputs, batch_locs, target, points, sign_list, batch_cpms, generated, log)
  
  return losses.avg
