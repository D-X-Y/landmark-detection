# Usage:
#   python resize_mv.py create small > MV-SMALL.sh
#   python resize_mv.py create big   > MV-BIG.sh
# tar xzvf 330012.tgz ; tar xzvf 330014.tgz ; tar xzvf 330028.tgz ; tar xzvf 330030.annotations.tgz ; tar xzvf 330030.tgz ; tar xzvf 330031.tgz ; tar xzvf 330034.tgz ; tar xzvf KRTs.tgz
#
import os, sys, cv2, torch
from os import path as osp
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
#print ('lib-dir : {:}'.format(lib_dir))


def get_views():
  xlist = ['330012', '330028', '330030', '330034', '330014', '330031']
  return xlist


def resize_image(mugsy_dir, targt_dir, name):
  views = get_views()
  for view in views:
    source = mugsy_dir / view / name
    target = targt_dir / view / name
    assert source.exists(), '{:} does not exist'.format(source)
    image = cv2.imread(str( source ))
    assert image.shape == (1280, 960, 3), '{:} has the shape of {:}'.format(source, image.shape)
    resized_image = cv2.resize(image, (240, 320))
    cv2.imwrite(str(target), resized_image)
  

def recove_image(sourceD, targetD, name):
  views = get_views()
  for view in views:
    source = sourceD / view / name
    target = targetD / view / name
    assert source.exists(), '{:} does not exist'.format(source)
    image = cv2.imread(str( source ))
    assert image.shape == (320 , 240, 3), '{:} has the shape of {:}'.format(source, image.shape)
    resized_image = cv2.resize(image, (960, 1280))
    cv2.imwrite(str(target), resized_image)


if __name__ == '__main__':
  HOME_STR = 'DOME_HOME'
  if HOME_STR not in os.environ: HOME_STR = 'HOME'
  assert HOME_STR in os.environ, 'Doest not find the HOME dir : {}'.format(HOME_STR)
  mugsy_dir = osp.join(os.environ[HOME_STR], 'datasets', 'landmark-datasets', 'multiview')
  targt_dir = osp.join(os.environ[HOME_STR], 'datasets', 'landmark-datasets', 'small-mv')
 
  assert len(sys.argv) == 3, 'There must be one arg vs {:}'.format(sys.argv)
  
  if sys.argv[1] == 'create':
    allviews = get_views()
    if sys.argv[2] == 'small':
      images = list( (Path(mugsy_dir) / allviews[0]).glob('*.png') )
      assert len(images) == 63265, '{:} vs 63265'.format( len(images) )
      if not osp.isdir( targt_dir ): os.makedirs( targt_dir )
      for view in allviews:
        if not osp.isdir( osp.join(targt_dir, view) ):
          os.makedirs( osp.join(targt_dir, view) )
      images = [x.name for x in images]
      for image in images:
        print ('python {:} {:} {:}'.format(os.path.abspath(__file__), 'small', image))
    elif sys.argv[2] == 'big':
      images = list( (Path(targt_dir) / allviews[0]).glob('*.png') )
      assert len(images) == 63265, '{:} vs 63265'.format( len(images) )
      if not osp.isdir( mugsy_dir ): os.makedirs( mugsy_dir )
      for view in allviews:
        if not osp.isdir( osp.join(mugsy_dir, view) ):
          os.makedirs( osp.join(mugsy_dir, view) )
      images = [x.name for x in images]
      for image in images:
        print ('python {:} {:} {:}'.format(os.path.abspath(__file__), 'big'  , image))
    else: raise ValueError('Invalid argv : {:}'.format( sys.argv ))
  elif sys.argv[1] == 'small':
    assert len(sys.argv) == 3, 'invalid commands : {:}'.format(sys.argv)
    name = sys.argv[2]
    resize_image(Path(mugsy_dir), Path(targt_dir), name)
  elif sys.argv[1] == 'big':
    assert len(sys.argv) == 3, 'invalid commands : {:}'.format(sys.argv)
    name = sys.argv[2]
    recove_image(Path(targt_dir), Path(mugsy_dir), name)
  else:
    raise ValueError('Invalid argv : {:}'.format( sys.argv ))
