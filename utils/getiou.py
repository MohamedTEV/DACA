import numpy as np
import cv2
import random
import torch
import torch.nn as nn


def box2d_iou(box1, box2, mode=0):
    # box2 = box2.cpu().detach().numpy().astype(int)

    ''' Compute 2D bounding box IoU.
    Input:
        box1: tuple of (xmin,ymin,xmax,ymax)
        box2: tuple of (xmin,ymin,xmax,ymax)
        mode : int
            0: w.r.t. to both
            1: w.r.t. to the first argument
            2: w.r.t. to the second argument
    Output:
        iou: 2D IoU scalar
    '''
    return get_iou({'x1':box1[0], 'y1':box1[1], 'x2':box1[2], 'y2':box1[3]}, \
        {'x1':box2[0], 'y1':box2[1], 'x2':box2[2], 'y2':box2[3]}, mode)


def get_iou(bb1, bb2, mode=0):
    """
    Calculate the Intersection over Union (IoU) of two 2D bounding boxes.
    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    mode : int
        0: w.r.t. to both
        1: w.r.t. to the first argument
        2: w.r.t. to the second argument
    Returns
    -------
    float
        in [0, 1]
    """
    # print("inside", bb2['x1'], bb2['x2'])
    
    # to fix misdetections where the bbox happens to be a line
    if bb1['x2'] <= bb1['x1']:
        bb1['x2'] += abs(bb1['x1'] - bb1['x2']) + 1
    if bb2['x2'] <= bb2['x1']:
        bb2['x2'] += abs(bb2['x1'] - bb2['x2']) + 1
    if bb1['y2'] <= bb1['y1']:
        bb1['y2'] += abs(bb1['y1'] - bb1['y2']) + 1
    if bb2['y2'] <= bb2['y1']:
        bb2['y2'] += abs(bb2['y1'] - bb2['y2']) + 1
        
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if mode == 0:
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    if mode == 1:
        iou = intersection_area / float(bb1_area)
    if mode == 2:
        iou = intersection_area / float(bb2_area)

    assert mode in [0, 1, 2]
    assert iou >= 0.0
    assert iou <= 1.0
    return iou