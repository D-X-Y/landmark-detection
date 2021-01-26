# [Supervision by Registration and Triangulation for Landmark Detection](https://xuanyidong.com/resources/papers/TPAMI-20-SRT.pdf)
By Xuanyi Dong, Yi Yang, Shih-En Wei, Xinshuo Weng, Yaser Sheikh, Shoou-I Yu

University of Technology Sydney, Facebook Reality Labs

## Introduction
We present Supervision by Registration and Triangulation (SRT), an unsupervised approach that utilizes unlabeled multi-view video to improve the accuracy and precision of landmark detectors. Being able to utilize unlabeled data enables our detectors to learn from massive amounts of unlabeled data freely available and not be limited by the quality and quantity of manual human annotations. To utilize unlabeled data, there are two key observations: (1) the detections of the same landmark in adjacent frames should be coherent with registration, i.e., optical flow. (2) the detections of the same landmark in multiple synchronized and calibrated views should correspond to a single 3D point, i.e., multi-view consistency. Registration and multi-view consistency are sources of supervision that do not require manual labeling, thus it can be leveraged to augment existing training data during detector training. End-to-end training is made possible by differentiable registration and 3D triangulation modules. Experiments with 11 datasets and a newly proposed metric to measure precision demonstrate accuracy and precision improvements in landmark detection on both images and video.

**Some pre-trained models and training logs can be found at [Google Drive](https://drive.google.com/drive/folders/1ABt0XUK3Imqnvahqzyw3lkj06Ri5PSvQ?usp=sharing).**
The paper can be found at [arXiv](https://arxiv.org/abs/2101.09866), [IEEE TPAMI](https://ieeexplore.ieee.org/document/9050873), or [our website](https://xuanyidong.com/resources/papers/TPAMI-20-SRT.pdf).


## Requirements

- Python 3.7
- PyTorch >= 1.0 
- torchvision >= 0.2.1
- tqdm
- OpenCV 3.4
- accimage or PIL
We strongly recommend using the anaconda Python distribution.



## Data Preparation

In the `cache_data` directory, see README to generate the dataset file for each dataset.


## Training and Evaluation


### The Regression-based Detector

Training and evaluating the simple regression model on 300-W (49 key-points or 68 key-points).
`0` indicates the GPU id. `RD01` indicates using 100-epochs for training.
```
bash scripts/X-300W/REG-300W-P49.sh 0 RD01
bash scripts/X-300W/REG-300W-P68.sh 0 RD01
```

Augmenting the 300-W trained simple regression model with registration (SBR) on 300-VW.
`W01` indicates the weight of the SBR loss as `0.1`.
```
bash scripts/X-300W/REG-300W-SBR-300VW-P49.sh 0 W01
bash scripts/X-300W/REG-300W-SBR-300VW-P68.sh 0 W01
```

Augmenting the 300-W trained simple regression model with SBR on the VoxCeleb2 dataset.
`W01` indicates the weight of the SBR loss as `0.1`.
```
bash scripts/X-300W/REG-300W-SBR-VOX-P49.sh   0 W01
bash scripts/X-300W/REG-300W-SBR-VOX-P68.sh   0 W01
```

Augmenting the 300-W trained simple regression model with SBR on the Panoptic-Face dataset.
`W01` indicates the weight of the SBR loss as `0.1`.
```
bash scripts/X-300W/REG-300W-SBR-PF-P49.sh    0 W01
bash scripts/X-300W/REG-300W-SBR-PF-P68.sh    0 W01
```

Augmenting the 300-W trained simple regression model with triangulation (SBT) on the Panoptic-Face dataset.
`W01` indicates the weight of the `SBT` loss as `0.1`. `16` indicates the batch size for multi-view data.
```
bash scripts/X-300W/REG-300W-SBT-PF-P49.sh    0 W01 16
```

Augmenting the 300-W trained simple regression model with both registration (SBR) and triangulation (SBT) on the Panoptic-Face dataset.
`W01` indicates the weight of the `SBR` loss as `0.1`.
`W05` indicates the weight of the `SBT` loss as `0.5`.
```
bash scripts/X-300W/REG-300W-SRT-PF-P49.sh    0 W01 W05
```

### The Heatmap-based Detector

Training and evaluating the hourglass model on 300W, where `ADAM` and `RMSP` indicate different kinds of optimizers.
```
bash ./scripts/300W/HEAT/DET-HG-68.sh 0 ADAM
bash ./scripts/300W/HEAT/DET-HG-68.sh 0 RMSP
bash ./scripts/300W/HEAT/DET-HG-49.sh 0 ADAM
bash ./scripts/300W/HEAT/DET-HG-49-V2.sh 0 ADAM
```

Augmenting the 300-W trained hourglass model with SBR on 300VW.
```
bash ./scripts/300W/HEAT/SBR-HG-300VW-68.sh 0 8
bash ./scripts/300W/HEAT/SBR-HG-300VW-49.sh 0 8
bash ./scripts/300W/HEAT/OK-SBR-HG-300VW-68.sh 0,1 L1 W01 16 8
bash ./scripts/300W/HEAT/OK-SBR-HG-300VW-49.sh 0,1 L1 W01 16 8
```

Augmenting the 300-W trained hourglass model with SBT on the Panoptic-Face dataset.
```
bash scripts/300W/HEAT/SBT-HG-PF-49.sh 0 L1   W01 16 8
```

More scripts can be found at `scripts`.


## Citation

If this project helps your research, please cite the following papers:
```
@inproceedings{dong2020srt,
  title     = {Supervision by Registration and Triangulation for Landmark Detection},
  author    = {Dong, Xuanyi and Yang, Yi and Wei, Shih-En and Weng, Xinshuo and Sheikh, Yaser and Yu, Shoou-I},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  volume    = {},
  number    = {},
  keywords  = {Landmark Detection;Optical Flow;Triangulation;Deep Learning},
  doi       = {10.1109/TPAMI.2020.2983935},
  ISSN      = {1939-3539},
  year      = {2020},
  month     = {},
  note      = {\mbox{doi}:\url{10.1109/TPAMI.2020.2983935}}
}
```

## Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/D-X-Y/landmark-detection/issues).

