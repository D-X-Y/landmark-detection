# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 1. extract raw data
#    tar xvf 171026_pose1.tar ; tar xvf 171026_pose2.tar ; tar xvf 171026_pose3.tar ; tar xvf 171204_pose1.tar ; tar xvf 171204_pose2.tar
#    tar xvf 171204_pose3.tar ; tar xvf 171204_pose4.tar ; tar xvf 171204_pose5.tar ; tar xvf 171204_pose6.tar
# 2. extract video frames
#    python GEN_Panoptic_XX.py create ./cache/extract_panoptic.sh
#    ./parallel ./cache/extract_panoptic.sh 100
# 3. undistort frames
#    python GEN_Panoptic_XX.py undistortion > ./cache/undistortImage.sh
# 4. generate the list
#    python GEN_Panoptic_XX.py genlist face
#    python GEN_Panoptic_XX.py genlist pose
# [option] : delete the generated Panoptic files
#    python GEN_Panoptic_XX.py clean > ./cache/delet_Panoptic.sh
#
import os
from os import path as osp
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import sys, math, torch, copy, json, subprocess
from collections import OrderedDict, defaultdict
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
print ('lib-dir : {:}'.format(lib_dir))
import datasets
from utils.file_utils import load_list_from_folders, load_txt_file
from xvision import draw_points, normalize_L
from xvision import draw_image_by_points
#torch.set_num_threads( 1 )


def isRotationMatrix(R) :
  Rt = np.transpose(R)
  shouldBeIdentity = np.dot(Rt, R)
  I = np.identity(3, dtype = R.dtype)
  n = np.linalg.norm(I - shouldBeIdentity)
  return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
  assert(isRotationMatrix(R))
  sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
  singular = sy < 1e-6
 
  if not singular :
    x = math.atan2(R[2,1] , R[2,2])
    y = math.atan2(-R[2,0], sy)
    z = math.atan2(R[1,0], R[0,0])
  else :
    x = math.atan2(-R[1,2], R[1,1])
    y = math.atan2(-R[2,0], sy)
    z = 0
  return np.array([x, y, z])

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
  R_x = np.array([[1,         0,                  0                   ],
                  [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                  [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                  ])
  R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                  [0,                     1,      0                   ],
                  [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                  ])
  R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                  [math.sin(theta[2]),    math.cos(theta[2]),     0],
                  [0,                     0,                      1]
                  ])
  R = np.dot(R_z, np.dot( R_y, R_x ))
  return R


def filter_views(video, cameras):
  if video == '171026_pose3':
    ok_views = [(0,0), (0,21), (0,24), (0,25), (0,29)]
  elif video == '171026_pose2':
    ok_views = [(0,0), (0,21), (0,24), (0,25), (0,29)]
  elif video == '171026_pose1':
    ok_views = [(0,0), (0,21), (0,24), (0,25), (0,29)]
  elif video == '171204_pose1':
    ok_views = [(0,0), (0,21), (0,24), (0,25), (0,29)]
  elif video == '171204_pose2':
    ok_views = [(0,0), (0,21), (0,24), (0,25), (0,29)]
  elif video == '171204_pose3':
    ok_views = [(0,0), (0,21), (0,24), (0,25), (0,29)]
  elif video == '171204_pose4':
    ok_views = [(0,0), (0,21), (0,24), (0,25), (0,29)]
  elif video == '171204_pose5':
    ok_views = [(0,0), (0,21), (0,24), (0,25), (0,29)]
  elif video == '171204_pose6':
    ok_views = [(0,0), (0,21), (0,24), (0,25), (0,29)]
  else:
    raise ValueError('Invalid video : {:}'.format( video ))
  Xcameras = {}
  for key, camera in cameras.items():
    if key in ok_views:
      Xcameras[ key ] = camera
  return Xcameras


def load_json_file(path):
  # Load camera calibration file
  assert path.exists(), 'Path : {:} does not exist.'.format( path )
  with open('{:}'.format(path)) as cfile:
    calib = json.load(cfile)
  # Cameras are identified by a tuple of (panel#,node#)
  cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}
  for k,cam in cameras.items():
    cam['K'] = np.matrix(cam['K'])
    cam['distCoef'] = np.array(cam['distCoef'])
    cam['R'] = np.matrix(cam['R'])
    cam['t'] = np.array(cam['t']).reshape((3,1))
  return cameras


def get_frame_infos(video):
  cmd_n = 'ffprobe -select_streams v -show_streams {:} 2>/dev/null | grep nb_frames | sed -e \'s/nb_frames=//\''.format( video )
  p = subprocess.Popen(cmd_n, shell=True, stdout=subprocess.PIPE)
  (output, err) = p.communicate()
  try:
    nums = int(output.strip())
    fail = False
  except:
    nums, fail = -1, True
  #cmd_fps = 'ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate {:}'.format( video )
  #p = subprocess.Popen(cmd_n, shell=True, stdout=subprocess.PIPE)
  #(output, err) = p.communicate()
  #fps = int(output.strip())
  return nums, fail


def get_frame_dims(video):
  cmd_n = 'ffprobe -v error -show_entries stream=width,height -of csv=p=0:s=x {:}'.format( video )
  p = subprocess.Popen(cmd_n, shell=True, stdout=subprocess.PIPE)
  (output, err) = p.communicate()
  HW = output.decode().strip().split('x')
  assert len(HW) == 2, 'Invalid HW : {:}'.format( output )
  # width, height
  return int(HW[0]), int(HW[1])


def get_face(json_path):
  with open(json_path) as dfile:
    fframe = json.load(dfile)
  faces, Rfaces = fframe['people'], []
  for element in faces:
    if element['id'] < 0: continue
    face = np.array( element['face70']['landmarks'] ).reshape((-1,3))
    Rfaces.append( face )
  return Rfaces


def get_pose(json_path):
  with open(json_path) as dfile:
    fframe = json.load(dfile)
  poses, Rposes = fframe['bodies'], []
  for element in poses:
    pose = np.array( element['joints19'] ).reshape((-1,4))
    Rposes.append( pose )
  return Rposes


def create_frame_script(root, script):
  root = Path(root)
  assert root.exists(), 'root : {:} not exist'.format(root)
  subdirs = [x for x in root.glob("*") if x.is_dir()]
  print ('There are {:} dirs in {:}'.format( len(subdirs), root ))

  if script is None: save_file = None
  else             : save_file = open(script, 'w')

  # check json file
  multiview_videos = []
  for video in subdirs:
    json_path = video / 'calibration_{:}.json'.format(video.name)
    if json_path.exists():
      multiview_videos.append( (video, load_json_file(json_path)) )
    else:
      print('{:} does not exist'.format( json_path ))
  print ('After filtering non-calibration videos, there left {:} videos'.format( len(multiview_videos) ))

  for video, cameras in multiview_videos:
    Cameras = {key:cam for key, cam in cameras.items() if cam['type'] == 'hd'}
    hd_video_dir = video / 'hdVideos'
    hd_frame_dir = video / 'hdFrames'
    if not hd_frame_dir.exists(): hd_frame_dir.mkdir(exist_ok=True)
    face_tar, face_dest = video / 'hdFace3d.tar', video / 'annots'
    assert face_tar.exists(), '{:} does not exists'.format(face_tar)
    if not face_dest.exists(): face_dest.mkdir(exist_ok=True)
    if save_file: save_file.write('tar xvf {:} -C {:}\n'.format(face_tar, face_dest))

    hand_tar, hand_dest = video / 'hdHand3d.tar', video / 'annots'
    assert hand_tar.exists(), '{:} does not exists'.format(hand_tar)
    if not hand_dest.exists(): hand_dest.mkdir(exist_ok=True)
    if save_file: save_file.write('tar xvf {:} -C {:}\n'.format(hand_tar, hand_dest))

    pose_tar, pose_dest = video / 'hdPose3d_stage1_coco19.tar', video / 'annots'
    assert pose_tar.exists(), '{:} does not exists'.format(pose_tar)
    if not pose_dest.exists(): pose_dest.mkdir(exist_ok=True)
    if save_file: save_file.write('tar xvf {:} -C {:}\n'.format(pose_tar, pose_dest))
  
    video_frames = -1
    OKcameras = {}
    for key, camera in Cameras.items():
      video_path = hd_video_dir / 'hd_{:}.mp4'.format(camera['name'])
      frame_path = hd_frame_dir / 'hd_{:}'.format(camera['name'])
      if video_path.exists():
        _frames, fail = get_frame_infos( video_path )
        if fail: print('{:} is broken'.format(video_path))
        else   :
          if video_frames == -1: video_frames = _frames
          else                 : assert video_frames == _frames, 'video : {:} with {:} , frames = {:} vs {:}'.format(video, video_path.name, _frames, video_frames)
          OKcameras[key] = copy.deepcopy( camera )
          OKcameras[key]['frame_num'] = video_frames
          OKcameras[key]['frame_path'] = str( frame_path )
          OKcameras[key]['video_path'] = str( video_path )
      else:
        print ('{:} does not exist.'.format(video_path))
      #if frame_path.exists(): frame_path.rmdir()
    print ('[{:}] has {:} avaliable cameras with {:} frames'.format(video, len(OKcameras), video_frames))
    torch.save(OKcameras, str(video/'calibration_{:}.pth'.format(video.name)))
    for key, camera in OKcameras.items():
      video_path = hd_video_dir / 'hd_{:}.mp4'.format(camera['name'])
      frame_path = hd_frame_dir / 'hd_{:}'.format(camera['name'])
      if not frame_path.exists(): frame_path.mkdir(exist_ok=True)
      command = 'ffmpeg -i {:} -q:v 1 -f image2 -start_number 0 {:}/{:}_%08d.png'.format(video_path, frame_path, camera['name'])
      #command = 'ffmpeg -i {:} -q:v 1 -f image2 -start_number 0 -t 00:00:35 {:}/{:}_%08d.png'.format(video_path, frame_path, camera['name'])
      #print('@ {:}'.format(command))
      if save_file: save_file.write('{:}\n'.format(command))
  
  if save_file is not None:
    print ('save into {:}'.format(script))
    save_file.close()


def GetAnnoDir(UDir, indicator, base):
  if indicator == 'face':
    annoDir = UDir / 'annots' / 'hdFace3d'
  elif indicator == 'pose':
    annoDir = UDir / 'annots' / 'hdPose3d_stage1_coco19'
  else: raise ValueError('Invalid Dir : {:} = {:}'.format(UDir, indicator))

  assert annoDir.exists(), '{:} does not exist'.format( annoDir )

  if base is None:
    annoPath = None
  elif indicator == 'face':
    annoPath = annoDir / 'faceRecon3D_hd{:}.json'.format(base)
  elif indicator == 'pose':
    annoPath = annoDir / 'body3DScene_{:}.json'.format(base)
  else: raise ValueError('Invalid Dir : {:}'.format( annoPath ))
  if annoPath: assert annoPath.exists(), '{:} does not exist'.format(annoPath)
  #annoPath = UndistoredDir.parent.parent / 'annots' / 'hdFace3d' / 'faceRecon3D_hd{:}.json'.format(name.split('.')[0])
  return annoDir, annoPath


def panoptic_generate(root, SAVE_DIR, indicator, max_frames=2000):
  if not SAVE_DIR.exists(): SAVE_DIR.mkdir(exist_ok=True)
  root = Path(root)
  assert root.exists(), 'root : {:} not exist'.format(root)
  subdirs = [x for x in root.glob("*") if x.is_dir()]
  print ('There are {:} dirs in {:}'.format( len(subdirs), root ))

  def check_cameras(tdata):
    meta_names = None
    for key, camera in tdata.items():
      frame_dir = Path( camera['frame_path'].replace('hdFrames', 'UndistoredHDFrames') )
      #frame_dir = Path( camera['frame_path'] )
      video_pth = camera['video_path']
      frames = list(frame_dir.glob('*.png'))
      assert camera['type'] == 'hd', 'camera : {:}'.format( camera )
      frames = sorted(frames)
      if meta_names is None:
        meta_names = [x.name[6:] for x in frames]
      else:
        for x, y in zip(meta_names, frames):
          assert x == y.name[6:], '{:} vs {:}'.format(x, y)
    return meta_names

  cache_file = SAVE_DIR / 'cache.pth'
  print ('cache file : {:}'.format( cache_file ))
  if not cache_file.exists():
    multiview_videos = []
    for index, video in enumerate(subdirs):
      torch_path = video / 'calibration_{:}.pth'.format(video.name)
      if torch_path.exists():
        tdata = torch.load( torch_path )
        names = check_cameras(tdata)
        print ('[{:02d}/{:02d}]-[{:}] has {:} cameras with {:} frames for each'.format(index, len(subdirs), video, len(tdata), len(names)))
        multiview_videos.append( (video, tdata, names) )
      else:
        print ('[{:02d}/{:02d}]-[{:}] does not exist'.format(index, len(subdirs), video))
    torch.save(multiview_videos, cache_file)
  else:
    print ('Find the cache, load pre-processed cache : {:}'.format(cache_file))
    multiview_videos = torch.load(cache_file)
  print ('There are {:} avaliable multi-view videos, limit the maximum number of frames by {:}'.format( len(multiview_videos), max_frames ))

  def get_frame_path(cdir, prefix, name, xlist):
    (base, ext), returnx = name.split('.'), []
    for offset in xlist:
      xname = '{:}_{:08d}.{:}'.format( prefix, int(base)+offset, ext )
      cpath = cdir / xname
      if cpath.exists(): returnx.append( str(cpath) )
      else             : returnx.append( None )
    return returnx

  def projectPointsSimple(X, K, R, t):
    X, K, R, t = torch.from_numpy(X), torch.from_numpy(K), torch.from_numpy(R), torch.from_numpy(t)
    x = torch.mm(R,X) + t
    x = x / x[2:3,:]
    return torch.mm(K, x).numpy()

  def clear_points(xxdatas):
    new = []
    for x in xxdatas:
      x = copy.deepcopy(x)
      x['points'] = None
      new.append( x )
    return new
  def clear_video(xxdatas):
    new = []
    for x in xxdatas:
      x = copy.deepcopy(x)
      x['previous_frame'] = None
      x['next_frame'] = None
      new.append( x )
    return new
  # save_four_ways('Panoptic-DEMO', Datas, all_cameras, base_path)
  def save_four_ways(name, XDatas, ALL_cameras, base_path):
    torch.save({'name': name, 'datas': XDatas                           , 'all_cameras': ALL_cameras}, base_path + '.pth')
    torch.save({'name': name, 'datas': clear_points(XDatas)             , 'all_cameras': ALL_cameras}, base_path + '-nopts.pth')
    torch.save({'name': name, 'datas': clear_video(XDatas)              , 'all_cameras': ALL_cameras}, base_path + '-novid.pth')
    torch.save({'name': name, 'datas': clear_video(clear_points(XDatas)), 'all_cameras': ALL_cameras}, base_path + '-nopts-novid.pth')
    print ('--------- {:} datas, {:} multiview cameras'.format(len(XDatas), len(ALL_cameras)))

  save_path, simple_path = SAVE_DIR / 'all-{:}-{:}'.format(indicator, max_frames), SAVE_DIR / 'simple-{:}-{:}'.format(indicator, max_frames)
  save_path, simple_path = str(save_path), str(simple_path)

  Datas, all_cameras = [], {}
  for index, (video, cameras, names) in enumerate(multiview_videos):
    cameras_index = len(all_cameras)
    # Load 3D points
    name2points3D = {}
    for name in names:
      _, annoPath = GetAnnoDir(video, indicator, name.split('.')[0])
      if indicator == 'face':
        pointsList = get_face(annoPath)
      elif indicator == 'pose':
        pointsList = get_pose(annoPath)
      else: raise ValueError('Invalid indicator : {:}'.format(indicator))
      if len(pointsList) > 0:
        name2points3D[name] = pointsList
      else:
        print('WARNING : {:03d}/{:03d} :: {:} has zero face/pose.'.format(index, len(multiview_videos), name))

    if indicator == 'face':
      Xcameras = filter_views(video.name, cameras)
    else:
      Xcameras = cameras
    for key, camera in Xcameras.items():
      camera['frame_path'] = camera['frame_path'].replace('hdFrames', 'UndistoredHDFrames')
    all_cameras[cameras_index] = copy.deepcopy( Xcameras )
    print ('After filtering, there are {:}/{:} cameras'.format(len(Xcameras), len(cameras)))
    
    # save into datas
    names = sorted(list(name2points3D.keys()))
    for iname, name in enumerate(names):
      pointsList = name2points3D[ name ]
      for key, camera in Xcameras.items():
        UndistoredDir = Path(camera['frame_path'].replace('hdFrames', 'UndistoredHDFrames'))
        Past, Now, Next = get_frame_path(UndistoredDir, camera['name'], name, [-1,0,1])
        assert Now is not None, '{:}\n{:}\n'.format(UndistoredDir, camera['name'], name)
        assert Path(Now).exists(), '{:} does not exists'.format( Now )
        for points in pointsList:
          try:
            points2D = projectPointsSimple(points[:,:3].T, camera['K'], camera['R'], camera['t'])
          except:
            import pdb; pdb.set_trace()
            print('Check invalid data')
          if points.shape[1] == 4: points2D[2,:] = (points[:,3] >= 0.1)
          box = (float(points2D[0].min()), float(points2D[1].min()), float(points2D[0].max()), float(points2D[1].max()))

          data = {'cameras_index' : cameras_index,
                  'camera_key'    : key,
                  'camera_name'   : camera['name'],
                  'previous_frame': Past,
                  'current_frame' : Now,
                  'next_frame'    : Next,
                  'points'        : points2D,
                  'points-X{:}'.format(indicator) : points2D,
                  'box-default'   : box,
                  'normalizeL-default': None}
          Datas.append( data )
          #import pdb; pdb.set_trace()
          #if iname == 0:
          #  pil_image = draw_image_by_points(Now, points2D, 2, (255,0,0), None, None)
          #  pil_image.save( str( SAVE_DIR / 'TEMPS' / (video.name + '-' + Path(Now).name) ) )
    print('Process {:03d}/{:03d} videos : {:}, {:} in total.'.format(index, len(multiview_videos), video.name, len(Datas)))
    # save a small dataset
    if index == 1:
      save_four_ways('Panoptic-DEMO', Datas, all_cameras, simple_path)
      print ('Save a small demo Panoptic into {:}'.format(simple_path))

  save_four_ways('Panoptic-{:}'.format(indicator), Datas, all_cameras, save_path)
  print('save Panoptic into {:}, with {:} frames and {:} cameras'.format(save_path, len(Datas), sum(len(xs) for _, xs in all_cameras.items())))

  for index, (video, cameras, names) in enumerate(multiview_videos):
    basex = str( SAVE_DIR / 'x-{:}'.format(video.name) )
    xlist = []
    for x in Datas:
      if video.name in x['current_frame']: xlist.append( x )
    xlist = copy.deepcopy( xlist )
    save_four_ways('Panoptic-SMALL', xlist, all_cameras, basex)
    print ('{:}/{:} : {:} save into {:}.'.format(index, len(multiview_videos), video.name, basex))


def UndistortImage(image, K, distortion):
  h, w, c = image.shape
  mapx, mapy = cv2.initUndistortRectifyMap(K, distortion, None, K, (w,h), cv2.CV_32FC1)
  undistortI = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
  return undistortI


def undistortion(root, seq, video):
  root = Path(root)
  assert root.exists(), 'root : {:} not exist'.format(root)
  if seq is None:
    subdirs = [x for x in root.glob("*") if x.is_dir()]
    print ('There are {:} dirs in {:}'.format( len(subdirs), root ))
    for subdir in subdirs:
      torch_path = subdir / 'calibration_{:}.pth'.format(subdir.name)
      cameras = torch.load( torch_path )
      for key, cam in cameras.items():
        assert cam['type'] == 'hd', 'invalid type : {:}'.format( cam['type'] )
        print('python GEN_Panoptic_XX.py undistortion {:} {:}'.format(subdir.name, cam['name']))
  else:
    video_path = root / seq
    assert video_path.exists(), '{:} does not exist'.format(video_path)
    cameras = torch.load( video_path / 'calibration_{:}.pth'.format(seq) )
    print ('There are {:} cameras in {:}, and pick {:}'.format( len(cameras), video_path, video ) )
    key = video.split('_')
    assert len(key) == 2, 'invalid key = {:}'.format( key )
    key = (int(key[0]), int(key[1]))
    camera = cameras[key]
    frame_dir = camera['frame_path']
    save_dir  = Path( frame_dir.replace('hdFrames', 'UndistoredHDFrames') )
    images    = list( Path(frame_dir).glob('*.png') )
    images    = sorted( images )
    start, nums = 300, min(3300, len(images)) # old version
    #start, nums = 300, min(1000, len(images))
    print ('KEY : {:} | {:} has {:}/{:} frames save into {:}'.format(key, frame_dir, nums, len(images), save_dir))
    if not save_dir.exists(): save_dir.mkdir(parents=True, exist_ok=True)
    K, distortion = camera['K'], camera['distCoef']
    for index in tqdm( range(start, nums) ):
      image = images[index]
      I = cv2.imread( str(image) )
      UndistortedI = UndistortImage(I, K, distortion)
      cv2.imwrite(str(save_dir/image.name), UndistortedI)


def delete_all(root):
  root = Path(root)
  assert root.exists(), 'root : {:} not exist'.format(root)
  subdirs = [x for x in root.glob("*") if x.is_dir()]
  for video in subdirs:
    PTH = video / 'calibration_{:}.pth'.format(video.name)
    hdFrames = video / 'hdFrames'
    UndistoredHDFrames = video / 'UndistoredHDFrames'
    #cmd = 'rm {:} ; rm -rf {:} ; rm -rf {:}'.format(PTH, hdFrames, UndistoredHDFrames)
    #print ('{:}'.format( cmd ))
    temps = [x for x in UndistoredHDFrames.glob("*") if x.is_dir()]
    for temp in temps: print('rm -rf {:}'.format(temp))
    temps = [x for x in hdFrames.glob("*") if x.is_dir()]
    for temp in temps: print('rm -rf {:}'.format(temp))
    AnnoP = video / 'annots'
    for x in AnnoP.glob("*"):
      if x.is_dir(): print('rm -rf {:}'.format(x))


if __name__ == '__main__':
  HOME_STR = 'DOME_HOME'
  if HOME_STR not in os.environ: HOME_STR = 'HOME'
  assert HOME_STR in os.environ, 'Doest not find the HOME dir : {}'.format(HOME_STR)
  this_dir = osp.dirname(os.path.abspath(__file__))
  SAVE_DIR = Path(this_dir) / 'lists' / 'Panoptic_Pose'
  print ('This dir : {:}, HOME : [{:}] : {:}'.format(this_dir, HOME_STR, os.environ[HOME_STR]))
  PanopticDir = osp.join( os.environ[HOME_STR], 'datasets', 'landmark-datasets', 'Panoptic_Pose')
  assert len(sys.argv) >= 2, 'There must have one arg vs {:}'.format( sys.argv )

  if sys.argv[1] == 'create':
    if len(sys.argv) == 3  : script_path = sys.argv[2]
    elif len(sys.argv) == 2: script_path = None
    else: raise ValueError('Invalid argv : {:}'.format(sys.argv))
    create_frame_script(PanopticDir, script_path)
  elif sys.argv[1] == 'genlist':
    assert len(sys.argv) == 3, 'invalid argv : {:}'.format( sys.argv )
    indicator = sys.argv[2].upper()
    assert indicator == 'FACE' or indicator == 'POSE', 'invalid indicator : {:}'.format( indicator )
    SAVE_DIR = Path(this_dir) / 'lists' / 'Panoptic-{:}'.format(indicator)
    panoptic_generate(PanopticDir, SAVE_DIR, indicator.lower())
  elif sys.argv[1] == 'undistortion':
    if len(sys.argv) == 4  : seq, video = sys.argv[2], sys.argv[3]
    elif len(sys.argv) == 2: seq, video = None, None
    else: raise ValueError('Invalid argv : {:}'.format(sys.argv))
    undistortion(PanopticDir, seq, video)
  elif sys.argv[1] == 'clean':
    delete_all(PanopticDir)
  else:
    raise ValueError('Invalid argv : {:}'.format( sys.argv ))
