#!/usr/bin/env sh
set -e
ROOT='~/Desktop'
GANS='LSGAN WGAN WGANGP'
NORMS='batch instance'

for GAN in ${GANS};
do
for NORM in ${NORMS};
do
  scp -r Oculus:${ROOT}/landmark-detection/snapshots/300W-STYLE-${GAN}-${NORM}-AMS0-DET/metas/epoch-499-500-eval ${GAN}-${NORM}-AMS0
  scp -r Oculus:${ROOT}/landmark-detection/snapshots/300W-STYLE-${GAN}-${NORM}-AMS1-DET/metas/epoch-499-500-eval ${GAN}-${NORM}-AMS1
done
done
