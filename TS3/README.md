# Teacher Supervises Students How to Learn from Partially Labeled Images for Facial Landmark Detection
By Xuanyi Dong, Yi Yang

This paper presents a semi-supervised facial landmark detection algorithm.
In short, we propose to learn a teacher network that takes a pseudo labeled face as input and outputs the quality of its pseudo label. As a result, pseudo labeled samples which are qualified by the teacher will be used to train the student detector. We achieve higher accuracy than typical semi-supervised facial landmark methods.

## Model Configs

We released the model definition files in the `models` folder, including two student detectors and one teacher network.

## Full Project Codes?

This project is done at about June 2018.
The original codes use an old PyTorch version (0.3.1) and are a little bit messy. It would take me much time to re-organize codes and move it to PyTorch 1.0.1.
Thanks for your patience for waiting.

## Citation
If this project helps your research, please cite the following papers:
```
@inproceedings{dong2019teacher,
  title={Teacher Supervises Students How to Learn from Partially Labeled Images for Facial Landmark Detection},
  author={Dong, Xuanyi and Yang, Yi},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

## Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/D-X-Y/landmark-detection/issues).
