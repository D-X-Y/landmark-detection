# Supervision-by-Registration: An Unsupervised Approach to Improve the Precision of Facial Landmark Detectors
By Xuanyi Dong, Shoou-I Yu, Xinshuo Weng, Shih-En Wei, Yi Yang, Yaser Sheikh

University of Technology Sydney, Facebook Reality Labs

## Introduction
We propose a method to find facial landmarks (e.g. corner of eyes, corner of mouth, tip of nose, etc) more precisely.
Our method utilizes the fact that objects move smoothly in a video sequence (i.e. optical flow registration) to improve an existing facial landmark detector.
The key novelty is that no additional human annotations are necessary to improve the detector, hence it is an “unsupervised approach”.

![demo](https://github.com/facebookresearch/supervision-by-registration/blob/master/cache_data/cache/demo.gif)

## Citation
If you find that Supervision-by-Registration helps your research, please cite the paper:
```
@inproceedings{dong2018sbr,
  title={{Supervision-by-Registration}: An Unsupervised Approach to Improve the Precision of Facial Landmark Detectors},
  author={Dong, Xuanyi and Yu, Shoou-I and Weng, Xinshuo and Wei, Shih-En and Yang, Yi and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={360--368},
  year={2018}
}
```

## Requirements
- PyTorch >= 0.4.0
- Python3.6

## Data Preparation

See the README in `cache_data`.

### Dataset Format
Each dataset is saved as one file, in which each row indicates one specific face in one image or one video frame.
The format of one line : 
```
image_path annotation_path x1 y1 x2 y2 (face_size)
```
- *image_path*: the image (video frame) file path of that face.
- *annotation_path*: the annotation file path of that face (annotation is the coordinates of all landmarks)
- *x1, y1, x2, y2*: the coordinates of left-upper and right-lower points of the face bounding box.
- *face_size*: an optional item. If set this value, we use the `face_size` to compute the NME; otherwise, we use the distance between two pre-defined points to compute the NME.

## Training

See the `configs` directory for some example configurations.
### Basic Training
```
python ./exps/basic_main.py [<required arguments>]
```
The argument list is loaded by `./lib/config_utils/basic_args.py`.
An examples script can is `./scripts/300W-DET.sh`, and you can simple run to train the base detector on the `300-W` dataset.
```
bash scripts/300W-DET.sh
```

### Improving the Detector by SBR
```
python ./exps/lk_main.py [<required arguments>]
```
The argument list is loaded by `./lib/config_utils/lk_args.py`.


#### An example to train SBR on the unlabeled sequences
The `init_model` parameter is the path to the detector trained in the `Basic Training` section.
```
bash scripts/demo_sbr.sh
```
To see visualization results use the commands in `Visualization`.

#### An example to train SBR on your own data
See the script `./scripts/sbr_example.sh`, and some parameters should be replaced by your own data.


## Evaluation

When using the `basic_main.py` or `lk_main.py`, we evaluate the testing datasets automatically.

To evaluate a single image, you can use the following script to compute the coordinates of 68 facial landmarks of the target image:
```
python ./exps/eval.py --image ./cache_data/cache/self.jpeg --model ./snapshots/300W-CPM-DET/checkpoint/cpm_vgg16-epoch-049-050.pth --face 250 150 900 1100 --save ./cache_data/cache/test.jpeg
```
- image : the input image path
- model : the snapshot path
- face  : the face bounding box
- save  : save the visualized results


## Visualization

After training the SBR on the demo video or models on other datasets, you can use the `./exps/vis.py` code to generate the visualization results.
```
python ./exps/vis.py --meta snapshots/CPM-SBR/metas/eval-start-eval-00-01.pth --save cache_data/cache/demo-detsbr-vis
ffmpeg -start_number 3 -i cache_data/cache/demo-detsbr-vis/image%04d.png -b:v 30000k -vf "fps=30" -pix_fmt yuv420p cache_data/cache/demo-detsbr-vis.mp4

python ./exps/vis.py --meta snapshots/CPM-SBR/metas/eval-epoch-049-050-00-01.pth --save cache_data/cache/demo-sbr-vis
ffmpeg -start_number 3 -i cache_data/cache/demo-sbr-vis/image%04d.png -b:v 30000k -vf "fps=30" -pix_fmt yuv420p cache_data/cache/demo-sbr-vis.mp4
```
- meta : the saved prediction files
- save : the directory path to save the visualization results


## License
supervision-by-registration is released under the [CC-BY-NC license](https://github.com/facebookresearch/supervision-by-registration/blob/master/LICENSE).


## Useful Information

### 1. train on your own video data
You should look at the `./lib/datasets/VideoDataset.py` and `./lib/datasets/parse_utils.py`, and add how to find the neighbour frames when giving one image path.
For more details, see the `parse_basic` function in `lib/datasets/parse_utils.py`.

### 2. warnings when training the AFLW datase
It is ok to show the following warnings. Since some images in the AFLW dataset are in the wrong format, PIL will raise some warnings when loading these images. These warnings do not affect the training performance.
```
TiffImagePlugin.py:756: UserWarning: Corrupt EXIF data.  Expecting to read 12 bytes but only got 6.
```

### Contact
To ask questions or report issues, please open an issue on [the issues tracker](https://github.com/facebookresearch/supervision-by-registration/issues).
