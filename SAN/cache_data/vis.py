import PIL, numpy as np
import sys, pdb, os, torch
from os import path as osp
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
print ('LIB : {:}'.format(lib_dir))
import models, datasets
from san_vision import transforms
from visualization import merge_images, generate_color_from_heatmaps, overlap_two_pil_image
PRINT_GAP = 500

def visual(clist, cdir, num_pts):
  if not cdir.exists(): os.makedirs(str(cdir))
  shape = 256
  transform  = transforms.Compose([transforms.PreCrop(0.2), transforms.TrainScale2WH((shape, shape))])
  data = datasets.GeneralDataset(transform, 2, 1, 'gaussian', 'test')
  data.load_list(clist, num_pts, True)
  for i, tempx in enumerate(data):
    image = tempx[0]
    heats = models.variable2np(tempx[1]).transpose(1,2,0)
    xheat = generate_color_from_heatmaps(heats, index=-1)
    xheat = PIL.Image.fromarray(np.uint8(xheat*255))

    cimage = overlap_two_pil_image(image, xheat)

    basename = osp.basename(data.datas[i]).split('.')[0]
    basename = str(cdir) + '/' + basename + '-{:}.jpg'

    image.save(basename.format('ori'))
    xheat.save(basename.format('heat'))
    cimage.save(basename.format('over'))

    if i % PRINT_GAP == 0:
      print ('--->>> process the {:4d}/{:4d}-th image'.format(i, len(data)))

if __name__ == '__main__':
  clist = './lists/300W/Original/300w.train.GTB'
  cdir = Path('./cache/temp-visualize')
  visual(clist, cdir, 68)
