# Landmark Detection

This project contains three landmark detection algorithms, implemented in [PyTorch](pytorch.org).

- Style Aggregated Network for Facial Landmark Detection, CVPR 2018
- Supervision-by-Registration: An Unsupervised Approach to Improve the Precision of Facial Landmark Detectors, CVPR 2018
- Teacher Supervises Students How to Learn from Partially Labeled Images for Facial Landmark Detection, ICCV 2019
- Supervision by Registration and Triangulation for Landmark Detection, TPAMI 2020


## Style Aggregated Network for Facial Landmark Detection

The training and testing codes for [SAN (CVPR 2018)](https://xuanyidong.com/publication/cvpr-2018-san/) are located in the [SAN directory](https://github.com/D-X-Y/landmark-detection/tree/master/SAN).

## Supervision-by-Registration: An Unsupervised Approach to Improve the Precision of Facial Landmark Detectors

The training and testing codes for [Supervision-by-Registration (CVPR 2018)](https://xuanyidong.com/publication/cvpr-2018-sbr/) are located in the [SBR directory](https://github.com/D-X-Y/landmark-detection/tree/master/SBR).

## Teacher Supervises Students How to Learn from Partially Labeled Images for Facial Landmark Detection

The model codes for [Teacher Supervises Students (TS3) (ICCV 2019)](https://arxiv.org/abs/1908.02116) are located in the [TS3 directory](https://github.com/D-X-Y/landmark-detection/tree/master/TS3).

## Supervision by Registration and Triangulation for Landmark Detection

The training and testing codes for [SRT (TPAMI) 2020](https://ieeexplore.ieee.org/document/9050873) are located in the [SRT directory](https://github.com/D-X-Y/landmark-detection/tree/master/SRT).

## Citation
If this project helps your research, please cite the following papers:
```
@inproceedings{dong2018san,
   title={Style Aggregated Network for Facial Landmark Detection},
   author={Dong, Xuanyi and Yan, Yan and Ouyang, Wanli and Yang, Yi},
   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
   pages={379--388},
   doi={10.1109/CVPR.2018.00047},
   year={2018}
}
@inproceedings{dong2018sbr,
  title={{Supervision-by-Registration}: An Unsupervised Approach to Improve the Precision of Facial Landmark Detectors},
  author={Dong, Xuanyi and Yu, Shoou-I and Weng, Xinshuo and Wei, Shih-En and Yang, Yi and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={360--368},
  doi={10.1109/CVPR.2018.00045},
  year={2018}
}
@inproceedings{dong2019teacher,
  title={Teacher Supervises Students How to Learn from Partially Labeled Images for Facial Landmark Detection},
  author={Dong, Xuanyi and Yang, Yi},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages={783--792},
  doi={10.1109/ICCV.2019.00087},
  year={2019}
}
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
