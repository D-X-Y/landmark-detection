import numpy as np
import os

## bbox is a vector has two values
def padHeight(image, padValue, bbox):
  assert isinstance(image,np.ndarray), 'incorrect type : {}'.format(type(image))
  assert len(image.shape) == 3 and image.shape[2] == 3, 'incorrect shape : {}'.format(image.shape)
  height, width = image.shape[0], image.shape[1]
  bbox[0] = np.max ( [ bbox[0], height] )
  bbox[0] = np.ceil( bbox[0] / 8.0 ) * 8
  #print ('----------- h/w : {} , bbox : {}'.format(image.shape, bbox))
  bbox[1] = np.max ( [ bbox[1], width ] )
  #print ('----------- h/w : {} , bbox : {}'.format(image.shape, bbox))
  bbox[1] = np.ceil( bbox[1] / 8.0 ) * 8 
  #print ('----------- h/w : {} , bbox : {}'.format(image.shape, bbox))

  pad = [ np.floor((bbox[0]-height)/2), np.floor((bbox[1]-width)/2) ] # up, left
  pad.append( bbox[0] - height - pad[0] ) # down
  pad.append( bbox[1] - width  - pad[1] ) # right
  pad = np.array( pad, dtype='int')

  img_padded = image.copy()
  pad_up     = np.zeros( (pad[0], img_padded.shape[1], image.shape[2]), dtype='float32') + padValue
  img_padded = np.concatenate( (pad_up,   img_padded), axis=0).astype(image.dtype)
  #print ('pad_up    shape : {}, img_padded shape : {}'.format(pad_up.shape,   img_padded.shape))
  pad_left   = np.zeros( (img_padded.shape[0], pad[1], image.shape[2]), dtype='float32') + padValue
  img_padded = np.concatenate( (pad_left, img_padded), axis=1).astype(image.dtype)
  #print ('pad_left  shape : {}, img_padded shape : {}'.format(pad_left.shape, img_padded.shape))
  pad_down   = np.zeros( (pad[2], img_padded.shape[1], image.shape[2]), dtype='float32') + padValue
  img_padded = np.concatenate( (img_padded, pad_down), axis=0)
  #print ('pad_down  shape : {}, img_padded shape : {}'.format(pad_down.shape, img_padded.shape))
  pad_right  = np.zeros( (img_padded.shape[0], pad[1], image.shape[2]), dtype='float32') + padValue
  img_padded = np.concatenate( (img_padded,pad_right), axis=1).astype(image.dtype)
  #print ('pad_right shape : {}, img_padded shape : {}'.format(pad_right.shape, img_padded.shape))
  #cv2.imwrite('{}/test.jpg'.format(os.environ['HOME']), img_padded)
  return img_padded, pad


## re-scale the score heat-map (H*W*C) to the size of original image
def resize2scaled_img(heatmap, pad):
  assert isinstance(heatmap, np.ndarray), 'incorrect type : {}'.format(type(image))
  assert len(heatmap.shape) == 3 and heatmap.shape[2] >= 1, 'incorrect shape : {}'.format(heatmap.shape)
  assert len(pad) == 4, 'incorrect pad shape : {}'.format(pad)
  score = heatmap.copy()
  if pad[0] < 0:
    pad_up    = np.zeros( (-pad[0], score.shape[1], score.shape[2]), dtype=heatmap.dtype)
    score     = np.concatenate( (pad_up, score), axis=0).astype(heatmap.dtype)
  elif pad[0] > 0:
    score     = score[pad[0]:, :, :]

  if pad[1] < 0:
    padleft   = np.zeros( (score.shape[0], -pad[1], score.shape[2]), dtype=heatmap.dtype)
    score     = np.concatenate( (padleft, score), axis=1).astype(heatmap.dtype)
  elif pad[1] > 0:
    score     = score[:, :-pad[1], :]

  if pad[2] < 0:
    paddown   = np.zeros( (-pad[2], score.shape[1], score.shape[2]), dtype=heatmap.dtype)
    score     = np.concatenate( (score, paddown), axis=0).astype(heatmap.dtype)
  elif pad[2] > 0:
    score     = score[pad[2]:, :, :]

  if pad[3] < 0:
    padright  = np.zeros( (score.shape[0], -pad[3], score.shape[2]), dtype=heatmap.dtype)
  elif pad[3] > 0:
    score     = score[:, :-pad[3], :]

  return score

def im2float(_image):
  image = _image.copy()
  if (image.dtype == 'uint8'):
    image = image.astype(np.float32) / 255
  elif (image.dtype == 'uint16'):
    image = image.astype(np.float32) / 65535
  else:
    assert False, 'unsupport dtype : {}'.format(image.dtype)
  return image
