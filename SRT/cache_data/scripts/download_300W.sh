# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#!/usr/bin/env sh

PWD=`pwd`
save_dir=${HOME}/datasets/landmark-datasets
echo 'PWD : ' ${PWD}
echo 'DIR : ' ${save_dir}

W300DIR=${save_dir}/300W
echo '300W : ' ${W300DIR}

mkdir -p ${W300DIR}
cd ${W300DIR}
wget https://ibug.doc.ic.ac.uk/download/annotations/ibug.zip
wget https://ibug.doc.ic.ac.uk/download/annotations/afw.zip
wget https://ibug.doc.ic.ac.uk/download/annotations/helen.zip
wget https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip

unzip ${W300DIR}/ibug.zip  -d ${W300DIR}/ibug
unzip ${W300DIR}/afw.zip   -d ${W300DIR}/afw
unzip ${W300DIR}/helen.zip -d ${W300DIR}/helen
unzip ${W300DIR}/lfpw.zip  -d ${W300DIR}/lfpw

mv ${W300DIR}/ibug/image_092\ _01.jpg ${W300DIR}/ibug/image_092_01.jpg
mv ${W300DIR}/ibug/image_092\ _01.pts ${W300DIR}/ibug/image_092_01.pts
