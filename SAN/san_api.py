import sys
from os import path as osp
from pathlib import Path

import numpy as np
import torch
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


# this aims to avoid `sys.path` changing outside this module
this_dir = osp.dirname(osp.abspath(__file__))
lib_path = osp.join(this_dir, 'lib')
sys.path.insert(0, lib_path)

import models
from datasets.dataset_utils import pil_loader
from san_vision import transforms
from datasets.point_meta import Point_Meta

sys.path.pop(0)


class SanLandmarkDetector(object):
    '''Wraps SAN Face Landmark Detector module to simple API for end user

    Example:
        ```python
        image_path = './cache_data/cache/test_1.jpg'
        model_path = './snapshots/checkpoint_49.pth.tar'
        face = (819.27, 432.15, 971.70, 575.87)

        from san_api import SanLandmarkDetector
        det = SanLandmarkDetector(model_path, device)
        locs, scores = det.detect(image_path, face)
        ```
    '''
    def __init__(self, model_path, device=None, benchmark: bool=True):
        '''
        Args:
            module_path: path to pre-trained model (available to download, see README)
            device: CUDA device to use. str or torch.device instance
                warning: this is restricted to 'cpu' or 'cuda' only
                    ('cuda:1' won't work due to main package arcitecture)
                default is choose 'cuda' if available
            benchmark: to enable cudnn benchmark mode or not
        '''
        self.model_path = model_path
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if benchmark:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        snapshot = torch.load(self.model_path, map_location=self.device)
        self.param = snapshot['args']

        self.transform  = transforms.Compose([
            transforms.PreCrop(self.param.pre_crop_expand),
            transforms.TrainScale2WH((self.param.crop_width, self.param.crop_height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.net = models.__dict__[self.param.arch](self.param.modelconfig, None)
        self.net.train(False).to(self.device)

        weights = models.remove_module_dict(snapshot['state_dict'])
        self.net.load_state_dict(weights)

    def detect(self, image, face):
        '''
        Args:
            image: either path to image or actual image: PIL, numpy of Tensor (HxWxC dims)
            face: 
        Returns:
            (
                locations: 68x2 array of points detected,
                scores: 68 confidence levels for each point,
            )
        '''
        if isinstance(image, str) or isinstance(image, Path):
            image = pil_loader(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = Image.fromarray(image.numpy())
        else:
            raise ValueError(f'Unsupported input image type {type(image)}')

        meta = Point_Meta(self.param.num_pts, None, np.array(face), '', 'custom')
        image, meta = self.transform(image, meta)
        temp_save_wh = meta.temp_save_wh
        cropped_size = torch.IntTensor( [temp_save_wh[1], temp_save_wh[0], temp_save_wh[2], temp_save_wh[3]] )

        # network forward
        with torch.no_grad():
            inputs = image.unsqueeze(0).to(self.device)
            _, batch_locs, batch_scos, _ = self.net(inputs)

        # obtain the locations on the image in the orignial size
        np_batch_locs, np_batch_scos, cropped_size = batch_locs.cpu().numpy(), batch_scos.cpu().numpy(), cropped_size.numpy()
        locations, scores = np_batch_locs[0,:-1,:], np.expand_dims(np_batch_scos[0,:-1], -1)

        scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2) , cropped_size[1] * 1. / inputs.size(-1)

        locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + cropped_size[3]
        return locations.round().astype(np.int), scores
