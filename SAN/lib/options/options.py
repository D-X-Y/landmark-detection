##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import os, sys, time, pdb, random, argparse
import torch
import init_path

class Options():
  def __init__(self, model_names):
    parser = argparse.ArgumentParser(description='Train Style Aggregated Network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)   
    parser.add_argument('--train_list',       type=str,   nargs='+',      help='The list file path to the video training dataset.')
    parser.add_argument('--eval_lists',       type=str,   nargs='+',      help='The list file path to the video testing dataset.')
    parser.add_argument('--num_pts',          type=int,                   help='Number of point.')
    parser.add_argument('--arch',             metavar='ARCH', default='itn_cpm', help='model architectures')
    parser.add_argument('--cpm_stage',        type=int,                   help='Number of stages in CPM model.')
    # Data Argumentation
    parser.add_argument('--sigma',            type=float,                 help='sigma distance for CPM.')
    parser.add_argument('--scale_prob',       type=float, default=1.1,    help='argument scale probability.')
    parser.add_argument('--scale_min',        type=float,                 help='argument scale : minimum scale factor.')
    parser.add_argument('--scale_max',        type=float,                 help='argument scale : maximum scale factor.')
    parser.add_argument('--scale_eval',       type=float,                 help='argument scale : maximum scale factor.')
    parser.add_argument('--rotate_max',       type=int,                   help='argument rotate : maximum rotate degree.')
    parser.add_argument('--pre_crop_expand',  type=float,                 help='parameters for pre-crop expand ratio')
    parser.add_argument('--crop_height',      type=int,                   help='argument crop : crop height.')
    parser.add_argument('--crop_width',       type=int,                   help='argument crop : crop width.')
    parser.add_argument('--crop_perturb_max', type=int,                   help='argument crop : center of maximum perturb distance.')
    parser.add_argument('--arg_flip',         dest='arg_flip',            action='store_true', help='Using flip data argumentation or not ')
    parser.add_argument('--dataset_name',     type=str,                   metavar='N', help='The specific name of the datasets.')
    parser.add_argument('--heatmap_type',     type=str,   choices=['gaussian','laplacian'], metavar='N', help='The method for generating the heatmap.')
    parser.add_argument('--argmax_size',      type=int,   default=8,      metavar='N', help='The patch size for the soft-argmax function')
    parser.add_argument('--weight_of_idt',    type=float,                 metavar='N', help='The weight of identity loss in CPM')
    # Cycle-GAN
    parser.add_argument('--niter',            type=int, default=100,      help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay',      type=int, default=100,      help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--epoch_count',      type=int, default=1,        help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--cycle_batchSize',  type=int, default=1,        help='input batch size')
    parser.add_argument('--pool_size',        type=int, default=50,       help='the size of image buffer that stores previously generated images')
    parser.add_argument('--visual_freq',      type=int, default=-1,       help='frequence of visualization')
    parser.add_argument('--cycle_beta1',      type=float, default=0.5,    help='momentum term of adam')
    parser.add_argument('--cycle_lr',         type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_policy',        type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
    parser.add_argument('--identity',         type=float, default=0.0,    help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
    parser.add_argument('--lambda_A',         type=float, default=10.0,   help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B',         type=float, default=10.0,   help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--cycle_model_path', type=str,                   help='The dir to save the cycle model.')
    parser.add_argument('--cycle_a_lists',    type=str,   nargs='+',      help='The list file path to cycle-gan a dataset.')
    parser.add_argument('--cycle_b_lists',    type=str,   nargs='+',      help='The list file path to cycle-gan b dataset.')
    # Optimization options
    parser.add_argument('--evaluation',       dest='evaluation',          action='store_true', help='evaluation or not')
    parser.add_argument('--pure_resume',      dest='pure_resume',         action='store_true', help='Only load the model not resume.')
    parser.add_argument('--eval_once',        dest='eval_once',           action='store_true', help='evaluation only once for evaluation ')
    parser.add_argument('--eval_init',        dest='eval_init',           action='store_true', help='evaluation only once for evaluation then start normal training')
    parser.add_argument('--debug_save',       dest='debug_save',          action='store_true', help='debug to save the data')
    parser.add_argument('--pretrain',         dest='pretrain',            action='store_true', help='pre-train model or not')
    parser.add_argument('--convert68to49',    dest='convert68to49',       action='store_true', help='convert 68 to 49.')
    parser.add_argument('--convert68to51',    dest='convert68to51',       action='store_true', help='convert 68 to 51.')
    parser.add_argument('--error_bar',        type=float,                 help='For drawing the image with large distance error.')
    parser.add_argument('--epochs',           type=int,   default=300,    help='Number of epochs to train.')
    parser.add_argument('--batch_size',       type=int,   default=2,      help='Batch size for training.')
    parser.add_argument('--eval_batch',       type=int,   default=4,      help='Batch size for testing.')
    parser.add_argument('--learning_rate',    type=float, default=0.1,    help='The Learning Rate.')
    parser.add_argument('--momentum',         type=float, default=0.9,    help='Momentum.')
    parser.add_argument('--decay',            type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule',         type=int,   nargs='+',      help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas',           type=float, nargs='+',      help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    # Checkpoints
    parser.add_argument('--print_freq',       type=int,   default=200,    metavar='N', help='print frequency (default: 200)')
    parser.add_argument('--print_freq_eval',  type=int,   default=200,    metavar='N', help='print frequency for evaluation (default: 200)')
    parser.add_argument('--save_path',        type=str,   default='./',                help='Folder to save checkpoints and log.')
    parser.add_argument('--resume',           type=str,   default='',     metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--start_epoch',      type=int,   default=0,      metavar='N', help='manual epoch number (useful on restarts)')
    # cluster kmeans
    parser.add_argument('--style_train_root',       type=str,                   help='To train style-discriminative feature.')
    parser.add_argument('--style_eval_root',       type=str,                   help='To train style-discriminative feature.')
    parser.add_argument('--n_clusters',       type=int,                   help='number of n_clusters')
    # Acceleration
    parser.add_argument('--gpu_ids',          type=str,                   help='empty for CPU, other for GPU-IDs')
    parser.add_argument('--workers',          type=int,   default=2,      help='number of data loading workers (default: 2)')
    # Random seed
    parser.add_argument('--manualSeed',       type=int,                   help='manual seed')
    self.opt = parser.parse_args()
    if self.opt.gpu_ids is None:
      str_ids = []
    else:
      str_ids = self.opt.gpu_ids.split(',')
    self.opt.gpu_ids = []
    for str_id in str_ids:
      if int(str_id) >= 0: self.opt.gpu_ids.append( int(str_id) )

    if len(self.opt.gpu_ids) > 0:
      self.opt.use_cuda = True
      torch.cuda.set_device(self.opt.gpu_ids[0])
      assert torch.cuda.is_available(), 'Use gpu [{}] but torch.cuda is not available'.format(self.opt.gpu_ids)
    else:
      self.opt.use_cuda = False

  def show(self, log):
    print_log('------------ Options ------------', log)
    for k, v in sorted(args.items()):
      print_log('{}: {}'.format(str(k), str(v)), log)
    print_log('-------------- End --------------', log)
