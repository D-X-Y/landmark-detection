#!/usr/bin/env sh
ROOT=`pwd`
BATCH_WGAN=snapshots/300W-STYLE-WGAN-batch-DET/tensorboard-20-Jul
BATCH_GP=snapshots/300W-STYLE-WGANGP-batch-DET/tensorboard-20-Jul
INSTANCE_WGAN=snapshots/300W-STYLE-WGAN-instance-DET/tensorboard-20-Jul
INSTANCE_GP=snapshots/300W-STYLE-WGANGP-instance-DET/tensorboard-20-Jul

#tensorboard --logdir BATCH-WGAN:${ROOT}/${BATCH_WGAN},BATCH-GP:${ROOT}/${BATCH_GP},INSTANCE-WGAN:${ROOT}/${INSTANCE_WGAN},INSTANCE-GP:${ROOT}/${INSTANCE_GP}
tensorboard --logdir BATCH-GP:${ROOT}/${BATCH_GP},INSTANCE-GP:${ROOT}/${INSTANCE_GP}
