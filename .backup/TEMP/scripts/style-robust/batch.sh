#!/usr/bin/env sh
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameter for GPU-ID and GAN Type"
  exit 1
fi
GPU=$1
GAN=$2

sh scripts/style-robust/300W-Style-${GAN}.sh ${GPU} instance 0 DET
sh scripts/style-robust/300W-Style-${GAN}.sh ${GPU} instance 1 DET
sh scripts/style-robust/300W-Style-${GAN}.sh ${GPU} batch 0 DET
sh scripts/style-robust/300W-Style-${GAN}.sh ${GPU} batch 1 DET
