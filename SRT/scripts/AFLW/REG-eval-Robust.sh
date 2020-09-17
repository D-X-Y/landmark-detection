# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash ./scripts/AFLW/REG-eval-Robust.sh 0 snapshots/WELL-AFLW-default-REG-112x112-G1/last-info.pth
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for gpu-devices, and the-snapshot-path"
  exit 1
fi
gpus=$1
snapshot=$2
if [ ! -f ${snapshot} ]; then
  echo "snapshot: ${snapshot} is not a file"
  exit 1
fi

CUDA_VISIBLE_DEVICES=${gpus} python ./exps/ROBUST-eval.py \
    --eval_lists  ./cache_data/lists/AFLW/test.pth \
                  ./cache_data/lists/AFLW/test.front.pth \
                  ./cache_data/lists/AFLW/test-less-30-yaw.pth \
                  ./cache_data/lists/AFLW/test-more-60-yaw.pth \
    --mean_point  ./cache_data/lists/AFLW/train-mean.pth \
    --save_path   ./snapshots/test-robustness/AFLW \
    --init_model  ${snapshot} \
    --robust_scale 0.2 --robust_offset 0.1 --robust_rotate 30 \
    --rand_seed 1111 \
    --print_freq 200 --workers 10
