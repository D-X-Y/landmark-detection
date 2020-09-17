# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
# bash ./scripts/WFLW/HEAT-EVAL-VIS.sh 0 snapshots/WFLW/HRNet-DET-WFLW-default-256x256-S3/last-info.pth HRNet '0 1'
set -e
echo script name: $0
echo $# arguments
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for gpu devices and init-model"
  exit 1
fi
gpus=$1
checkpoint=$2
net_name=$3
use_stable=$4
det=default
use_gray=1
batch_size=16
save_dir=./snapshots/WFLW/EVAL-${det}-256x256


CUDA_VISIBLE_DEVICES=${gpus} python ./exps/BASE-eval-vis.py \
    --eval_vlists ./cache_data/lists/300VW/A/114.pth \
                  ./cache_data/lists/300VW/A/124.pth \
                  ./cache_data/lists/300VW/A/125.pth \
                  ./cache_data/lists/300VW/A/126.pth \
                  ./cache_data/lists/300VW/A/150.pth \
                  ./cache_data/lists/300VW/A/158.pth \
                  ./cache_data/lists/300VW/A/401.pth \
                  ./cache_data/lists/300VW/A/402.pth \
                  ./cache_data/lists/300VW/A/505.pth \
                  ./cache_data/lists/300VW/B/203.pth \
                  ./cache_data/lists/300VW/B/208.pth \
                  ./cache_data/lists/300VW/B/211.pth \
                  ./cache_data/lists/300VW/B/212.pth \
                  ./cache_data/lists/300VW/B/213.pth \
                  ./cache_data/lists/300VW/B/214.pth \
                  ./cache_data/lists/300VW/C/410.pth \
                  ./cache_data/lists/300VW/C/411.pth \
                  ./cache_data/lists/300VW/C/516.pth \
                  ./cache_data/lists/300VW/C/517.pth \
                  ./cache_data/lists/300VW/C/526.pth \
                  ./cache_data/lists/300VW/C/528.pth \
                  ./cache_data/lists/300VW/C/529.pth \
                  ./cache_data/lists/300VW/C/530.pth \
                  ./cache_data/lists/300VW/C/531.pth \
                  ./cache_data/lists/300VW/C/533.pth \
                  ./cache_data/lists/300VW/C/557.pth \
                  ./cache_data/lists/300VW/C/558.pth \
    --num_pts     68  --use_stable ${use_stable} \
    --model_name   ${net_name}  \
    --init_model   ${checkpoint} \
    --save_path    ${save_dir} \
    --batch_size ${batch_size} \
    --print_freq 600 --workers 8
