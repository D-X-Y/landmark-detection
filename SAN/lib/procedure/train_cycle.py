##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import os, sys, time
import os.path as osp
import torch
import torch.nn as nn
from utils import AverageMeter, print_log, convert_size2str, convert_secs2time, time_string, time_for_file

def convert2string(errors):
  string = 'D_A : {:.3f} G_A : {:.3f}'.format(errors['D_A'], errors['G_A'])
  string = string + ' D_B : {:.3f} G_B : {:.3f}'.format(errors['D_B'], errors['G_B'])
  if 'idt_A' in errors:
    string = string + ' idt_A : {:.3f}'.format(errors['idt_A'])
  if 'idt_B' in errors:
    string = string + ' idt_B : {:.3f}'.format(errors['idt_B'])
  return string

def save_visual(save_dir, visuals):
  if not osp.isdir(save_dir): os.makedirs(save_dir)
  visuals['real_A'].save( osp.join(save_dir, 'real_A.png') )
  visuals['real_B'].save( osp.join(save_dir, 'real_B.png') )
  visuals['fake_A'].save( osp.join(save_dir, 'fake_A.png') )
  visuals['fake_B'].save( osp.join(save_dir, 'fake_B.png') )
  visuals['rec_A'].save( osp.join(save_dir, 'rec_A.png') )
  visuals['rec_B'].save( osp.join(save_dir, 'rec_B.png') )
  if 'idt_A' in visuals: visuals['idt_A'].save( osp.join(save_dir, 'idt_A.png') )
  if 'idt_B' in visuals: visuals['idt_B'].save( osp.join(save_dir, 'idt_B.png') )

def train_cycle_gan(dataset, model, opt, log):

  save_dir = osp.join(opt.save_path, 'cycle-gan')
  print_log('save dir into {}'.format(save_dir), log)
  model.set_mode('train')
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.cycle_batchSize, shuffle=True, num_workers=opt.workers)
  epoch_start_time = time.time()
  epoch_time, total_steps = AverageMeter(), 0

  final_epoch = opt.niter + opt.niter_decay
  return_dir = osp.join(save_dir, 'itn-epoch-{}-{}'.format(final_epoch, final_epoch+1))
  if osp.isdir(return_dir):
    print_log('Exist cycle-gan model-save dir : {}, therefore skip train cycle-gan'.format(return_dir), log)
    return return_dir
  else:
    print_log('Does not find cycle-gan model-save dir, start training', log)

  for epoch in range(opt.epoch_count, final_epoch + 1):
    need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (final_epoch+1-epoch))
    print_log('\n==>>{:s} Epoch[{:03d}/{:03d}], [Time Left: {:02d}:{:02d}:{:02d}]'.format(time_string(), epoch, opt.niter+opt.niter_decay+1, need_hour, need_mins, need_secs), log)

    batch_time, data_time, epoch_iter = AverageMeter(), AverageMeter(), 0
    iter_time = time.time()
    for i, data in enumerate(dataloader):
      # prepare input
      total_steps += opt.cycle_batchSize
      epoch_iter += opt.cycle_batchSize
      model.set_input(data)
      # count time
      data_time.update(time.time() - iter_time)

      # forward and backward time
      model.optimize_parameters()
      batch_time.update(time.time() - iter_time)

      if i % opt.print_freq == 0:
        errors = model.get_current_errors()
        need_hour, need_mins, need_secs = convert_secs2time(batch_time.avg * (len(dataloader)-i))
        print_log(' Epoch: [{:3d}/{:03d}][{:4d}/{:4d}]  '.format(epoch, opt.niter+opt.niter_decay+1, i, len(dataloader)) \
                + ' Time {batch_time.val:5.1f} ({batch_time.avg:5.1f}) | {data_time.val:5.1f} ({data_time.avg:5.1f}) '.format(batch_time=batch_time, data_time=data_time) \
                + ' iter : {:5d}, {:5d}. '.format(epoch_iter, total_steps) \
                + '[{:02d}:{:02d}:{:02d}]  '.format(need_hour, need_mins, need_secs) \
                + convert2string(errors), log)

      if (opt.visual_freq > 0) and (i % opt.visual_freq == 0 or i + 1 == len(dataloader)):
        visuals = model.get_current_visuals(True)
        vis_save_dir = osp.join(save_dir, 'visual', '{:03d}-{:04d}'.format(epoch, i))
        save_visual(vis_save_dir, visuals)
      iter_time = time.time()

    # save model
    cur_save_dir = osp.join(save_dir, 'itn-epoch-{}-{}'.format(epoch, opt.niter+opt.niter_decay+1))
    model.save(cur_save_dir, log)
    epoch_time.update(time.time() - epoch_start_time)
    epoch_start_time = time.time()
    model.update_learning_rate(log)

  return cur_save_dir
