# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh
set -e
# http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html
USER=${user}
PASS=${password}

splits="a b c d e f g h i"
for NAME in ${splits}; do
 wget --user ${USER} --password ${PASS} "http://zeus.robots.ox.ac.uk/voxceleb2/mp4/vox2_dev_mp4a"${NAME}
done

wget --user ${USER} --password ${PASS} http://zeus.robots.ox.ac.uk/voxceleb2/mp4/vox2_test_mp4.zip

#cat vox2_dev* > vox2_mp4.zip
