# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
import os, sys, time

def get_list():
  range_of_motion = ['171204_pose1', '171204_pose2', '171204_pose3', '171204_pose4', '171204_pose5', '171204_pose6', '171026_pose1', '171026_pose2', '171026_pose3']
  haggling = ['161202_haggling1', '160422_haggling1', '160226_haggling1', '160224_haggling1']
  ultimatum = ['160422_ultimatum1', '160226_ultimatum1', '160224_ultimatum1', '160224_ultimatum2', '151125_ultimatum1']
  mafia = ['160422_mafia2', '160422_mafia1', '160226_mafia1', '160226_mafia2', '160224_mafia1', '160224_mafia2', '151125_mafia']
  dance = ['150821_dance1', '150821_dance2', '150821_dance3', '150821_dance4', '150821_dance5', '160317_moonbaby1', '160317_moonbaby2', '160317_moonbaby3']
  musical_instruments = ['160906_band1', '160906_band2', '160906_band3', '160906_band4', '150406_drum3', '150406_drum4', '150406_drum5', '150406_drum6', '150406_drum7', '150303_celloScene1', '150303_celloScene2', '150303_celloScene3', '150303_celloScene4', '150209_celloCapture1', '150209_celloCapture2']
  toddler = ['160906_ian5', '160906_ian3', '160906_ian2', '160906_ian1', '160401_ian3', '160401_ian2', '160401_ian1', '131015_extra', '131015_extra2', '131015_baseball', '131015_baby1', '131015_baby2']
  alllist = range_of_motion + haggling + ultimatum + mafia + dance + musical_instruments + toddler
  return alllist

if __name__ == '__main__':
  lists = get_list()
  os.system('git clone https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox.git')
  prefix = 'bash ../panoptic-toolbox/scripts/getData.sh'
  for name in lists:
    xstr = prefix + ' {:}'.format(name)
    print (xstr)
