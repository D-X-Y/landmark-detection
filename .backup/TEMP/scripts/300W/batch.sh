#!/usr/bin/env sh
echo $# arguments
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameter for GPU-ID"
  exit 1
fi
GPU=$1

sh scripts/300W/300W_CPM_SGD.sh ${GPU} V1 V1 DET
sh scripts/300W/300W_HG_RMSP.sh ${GPU} V1 V1 DET
sh scripts/300W/300W_HG_RMSP.sh ${GPU} V1 V2 DET
sh scripts/300W/300W_HG_ADAM.sh ${GPU} V1 V1 DET
sh scripts/300W/300W_HG_ADAM.sh ${GPU} V1 V2 DET
