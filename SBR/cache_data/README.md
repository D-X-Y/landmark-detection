# Dataset Preparation
The raw dataset should be put into the `$HOME/datasets/landmark-datasets`. The layout should be organized as the following screen shot.

![layout](https://github.com/facebookresearch/supervision-by-registration/blob/master/cache_data/cache/dir-layout.png)

## [300-W](https://ibug.doc.ic.ac.uk/resources/300-W/)

### Download
- 300-W consits of several different datasets
- Create directory to save images and annotations: mkdir ~/datasets/landmark-datasets/300W
- To download i-bug: https://ibug.doc.ic.ac.uk/download/annotations/ibug.zip
- To download afw: https://ibug.doc.ic.ac.uk/download/annotations/afw.zip
- To download helen: https://ibug.doc.ic.ac.uk/download/annotations/helen.zip
- To download lfpw: https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip
- To download the bounding box annotations: https://ibug.doc.ic.ac.uk/media/uploads/competitions/bounding_boxes.zip
- In the folder of `~/datasets/landmark-datasets/300W`, there are four zip files ibug.zip, afw.zip, helen.zip, and lfpw.zip
```
unzip ibug.zip -d ibug
mv ibug/image_092\ _01.jpg ibug/image_092_01.jpg
mv ibug/image_092\ _01.pts ibug/image_092_01.pts

unzip afw.zip -d afw
unzip helen.zip -d helen
unzip lfpw.zip -d lfpw
unzip bounding_boxes.zip ; mv Bounding\ Boxes Bounding_Boxes
```
The 300W directory is in `$HOME/datasets/landmark-datasets/300W` and the sturecture is:
```
-- afw
-- Bounding_boxes
-- helen
-- ibug
-- lfpw
```

Then you use the script to generate the 300-W list files.
```
python generate_300W.py
```
All list files will be saved into `./lists/300W/`. The files `*.DET` use the face detecter results for face bounding box. `*.GTB` use the ground-truth results for face bounding box.

#### can not find the `*.mat` files for 300-W.
The download link is in the official [300-W website](https://ibug.doc.ic.ac.uk/resources/300-W).
```
https://ibug.doc.ic.ac.uk/media/uploads/competitions/bounding_boxes.zip
```
The zip file should be unzipped, and all extracted mat files should be put into `$HOME/datasets/landmark-datasets/300W/Bounding_Boxes`.

## [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)

Download the aflw.tar.gz file in `$HOME/datasets/landmark-datasets` and extract it by `tar xzvf aflw.tar.gz`.
```
mkdir $HOME/datasets/landmark-datasets/AFLW
cp -r aflw/data/flickr $HOME/datasets/landmark-datasets/AFLW/images
```

The structure of AFLW is:
```
--images
  --0
  --2
  --3
```

Download the [AFLWinfo_release.mat](http://mmlab.ie.cuhk.edu.hk/projects/compositional/AFLWinfo_release.mat) from [this website](http://mmlab.ie.cuhk.edu.hk/projects/compositional.html) into `./cache_data`. This is the revised annotation of the full AFLW dataset.

Generate the AFLW dataset list file into `./lists/AFLW`.
```
python aflw_from_mat.py
```

## [300VW](https://ibug.doc.ic.ac.uk/resources/300-VW/)
Download `300VW_Dataset_2015_12_14.zip` into `$HOME/datasets/landmark-datasets` and unzip it into `$HOME/datasets/landmark-datasets/300VW_Dataset_2015_12_14`.

Use the following command to extract the raw video into the image format.
```
python extrct_300VW.py
sh ./cache/Extract300VW.sh
```

Generate the 300-VW dataset list file.
```
python generate_300VW.py
```

## a short demo video sequence

The raw video is `./cache_data/cache/demo-sbr.mp4`.
- use `ffmpeg -i ./cache/demo-sbr.mp4 ./cache/demo-sbrs/image%04d.png` to extract the frames into `/cache/demo-sbrs/`
Then use `python demo_list.py` to generate the list file for the demo video.

# Citation
If you use the 300-W dataset, please cite the following papers.
```
@article{sagonas2016300,
  title={300 faces in-the-wild challenge: Database and results},
  author={Sagonas, Christos and Antonakos, Epameinondas and Tzimiropoulos, Georgios and Zafeiriou, Stefanos and Pantic, Maja},
  journal={Image and Vision Computing},
  volume={47},
  pages={3--18},
  year={2016},
  publisher={Elsevier}
}
@inproceedings{sagonas2013300,
  title={300 faces in-the-wild challenge: The first facial landmark localization challenge},
  author={Sagonas, Christos and Tzimiropoulos, Georgios and Zafeiriou, Stefanos and Pantic, Maja},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={397--403},
  year={2013},
  organization={IEEE}
}
```
If you use the 300-VW dataset, please cite the following papers.
```
@inproceedings{chrysos2015offline,
  title={Offline deformable face tracking in arbitrary videos},
  author={Chrysos, Grigoris G and Antonakos, Epameinondas and Zafeiriou, Stefanos and Snape, Patrick},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={1--9},
  year={2015}
}
@inproceedings{shen2015first,
  title={The first facial landmark tracking in-the-wild challenge: Benchmark and results},
  author={Shen, Jie and Zafeiriou, Stefanos and Chrysos, Grigoris G and Kossaifi, Jean and Tzimiropoulos, Georgios and Pantic, Maja},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={50--58},
  year={2015}
}
@inproceedings{tzimiropoulos2015project,
  title={Project-out cascaded regression with an application to face alignment},
  author={Tzimiropoulos, Georgios},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3659--3667},
  year={2015}
}
```
If you use the AFLW dataset, please cite the following papers.
```
@inproceedings{koestinger2011annotated,
  title={Annotated facial landmarks in the wild: A large-scale, real-world database for facial landmark localization},
  author={Koestinger, Martin and Wohlhart, Paul and Roth, Peter M and Bischof, Horst},
  booktitle={IEEE International Conference on Computer Vision Workshops},
  pages={2144--2151},
  year={2011},
  organization={IEEE}
}
```
