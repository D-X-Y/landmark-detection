##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import numpy as np

def bboxcheck_TLBR(bbox):
    '''
    check the input bounding box to be TLBR format

    parameter:
        bbox:   N x 4 numpy array, TLBR format
    
    return:
        True or False
    '''
    OK1 = isinstance(bbox, np.ndarray) and bbox.shape[1] == 4 and bbox.shape[0] > 0
    OK2 = (bbox[:, 3] >= bbox[:, 1]).all() and (bbox[:, 2] >= bbox[:, 0]).all()
    return OK1 and OK2

def bbox2center(bbox):
    '''
    convert a bounding box to a point, which is the center of this bounding box

    parameter:
        bbox:   N x 4 numpy array, TLBR format

    return:
        center: 2 x N numpy array, x and y correspond to first and second row respectively
    '''
    assert bboxcheck_TLBR(bbox), 'the input bounding box should be TLBR format'

    num_bbox = bbox.shape[0]        
    center = np.zeros((num_bbox, 2), dtype='float32')
    center[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.
    center[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.

    return np.transpose(center)

def bbox_TLBR2TLWH(bbox):
    '''
    transform the input bounding box with TLBR format to TLWH format

    parameter:
        bbox: N X 4 numpy array, TLBR format

    return 
        bbox: N X 4 numpy array, TLWH format
    '''
    assert bboxcheck_TLBR(bbox), 'the input bounding box should be TLBR format'

    bbox_TLWH = np.zeros_like(bbox)
    bbox_TLWH[:, 0] = bbox[:, 0]
    bbox_TLWH[:, 1] = bbox[:, 1]
    bbox_TLWH[:, 2] = bbox[:, 2] - bbox[:, 0]
    bbox_TLWH[:, 3] = bbox[:, 3] - bbox[:, 1]
    return bbox_TLWH
