
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import copy
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm
import albumentations as A


def get_best_region(out, imgs_t):
    region_t = imgs_t[:, :, 0:int(imgs_t.shape[3]/2), 0:int(imgs_t.shape[2]/2)] # initialize in case no bboxes are detected
    best_side = 'topleft'  # initialize in case no bboxes are detected
    if out.shape[0] > 0:
        bboxes_target = copy.deepcopy(out)

        bboxes_target_topleft = bboxes_target[bboxes_target[:, 2]<imgs_t.shape[2]/2, :]
        bboxes_target_topleft = bboxes_target_topleft[bboxes_target_topleft[:, 3]<imgs_t.shape[3]/2, :]

        bboxes_target_bottomleft = bboxes_target[bboxes_target[:, 2]<imgs_t.shape[2]/2, :]
        bboxes_target_bottomleft = bboxes_target_bottomleft[bboxes_target_bottomleft[:, 3]>imgs_t.shape[3]/2, :]

        bboxes_target_bottomright = bboxes_target[bboxes_target[:, 2]>imgs_t.shape[2]/2, :]
        bboxes_target_bottomright = bboxes_target_bottomright[bboxes_target_bottomright[:, 3]>imgs_t.shape[3]/2, :]

        bboxes_target_topright = bboxes_target[bboxes_target[:, 2]>imgs_t.shape[2]/2, :]
        bboxes_target_topright = bboxes_target_topright[bboxes_target_topright[:, 3]<imgs_t.shape[3]/2, :]

        conf_topleft = np.mean(bboxes_target_topleft[:, -1]) if len(bboxes_target_topleft)>0 else 0
        conf_bottomleft = np.mean(bboxes_target_bottomleft[:, -1]) if len(bboxes_target_bottomleft)>0 else 0
        conf_bottomright = np.mean(bboxes_target_bottomright[:, -1]) if len(bboxes_target_bottomright)>0 else 0
        conf_topright = np.mean(bboxes_target_topright[:, -1]) if len(bboxes_target_topright)>0 else 0

        if bboxes_target.shape[0]>0:
            side = ['topleft', 'bottomleft', 'bottomright', 'topright']
            region_bboxes = [bboxes_target_topleft, bboxes_target_bottomleft, bboxes_target_bottomright, bboxes_target_topright]
            conf = [conf_topleft, conf_bottomleft, conf_bottomright, conf_topright]
            id_best = conf.index(max(conf))
            best_side = side[id_best]
            best_bboxes = region_bboxes[id_best]
        else:
            best_bboxes = []

        if best_bboxes.shape[0]>0:
            out = copy.deepcopy(best_bboxes)
            out = torch.from_numpy(out)

        if best_side == 'topleft' and best_bboxes.shape[0] > 0: 
            region_t = imgs_t[:, :, 0:int(imgs_t.shape[3]/2), 0:int(imgs_t.shape[2]/2)]
            # clip bboxes that exceed the selected area
            for o in range(len(out)):
                if out[o, 3] + out[o, 5]/2>imgs_t.shape[3]/2:
                    new_h = imgs_t.shape[3]/2 - out[o, 3] + out[o, 5]/2
                    new_cy =  imgs_t.shape[3]/2 - new_h/2
                    out[o, 3] = int(new_cy)
                    out[o, 5] = int(new_h)
                if out[o, 2] + out[o, 4]/2>imgs_t.shape[2]/2:
                    new_w = imgs_t.shape[2]/2 - out[o, 2] + out[o, 4]/2
                    new_cx =  imgs_t.shape[2]/2 - new_w/2
                    out[o, 2] = int(new_cx)
                    out[o, 4] = int(new_w)
            
        elif best_side == 'bottomleft' and best_bboxes.shape[0] > 0: 
            region_t = imgs_t[:, :, int(imgs_t.shape[3]/2):, 0:int(imgs_t.shape[2]/2)]
            # clip bboxes that exceed the selected area
            for o in range(len(out)):
                if out[o, 3] - out[o, 5]/2<imgs_t.shape[3]/2:
                    new_h = out[o, 3] - imgs_t.shape[3]/2 + out[o, 5]/2
                    new_cy =  imgs_t.shape[3]/2 + new_h/2
                    out[o, 3] = int(new_cy)
                    out[o, 5] = int(new_h)
                if out[o, 2] + out[o, 4]/2>imgs_t.shape[2]/2:
                    new_w = imgs_t.shape[2]/2 - out[o, 2] + out[o, 4]/2
                    new_cx =  imgs_t.shape[2]/2 - new_w/2
                    out[o, 2] = int(new_cx)
                    out[o, 4] = int(new_w)

        elif best_side == 'bottomright' and best_bboxes.shape[0] > 0: 
            region_t = imgs_t[:, :, int(imgs_t.shape[3]/2):, int(imgs_t.shape[2]/2):]                       
            # clip bboxes that exceed the selected area
            for o in range(len(out)):
                if out[o, 3] - out[o, 5]/2<imgs_t.shape[3]/2:
                    new_h = out[o, 3] - imgs_t.shape[3]/2 + out[o, 5]/2
                    new_cy =  imgs_t.shape[3]/2 + new_h/2
                    out[o, 3] = int(new_cy)
                    out[o, 5] = int(new_h)
                if out[o, 2] - out[o, 4]/2<imgs_t.shape[2]/2:
                    new_w = out[o, 2] - imgs_t.shape[2]/2 + out[o, 4]/2
                    new_cx =  imgs_t.shape[2]/2 + new_w/2
                    out[o, 2] = int(new_cx)
                    out[o, 4] = int(new_w)

        elif best_side == 'topright' and best_bboxes.shape[0] > 0: 
            region_t = imgs_t[:, :, 0:int(imgs_t.shape[3]/2), int(imgs_t.shape[2]/2):]
            # clip bboxes that exceed the selected area
            for o in range(len(out)):
                if out[o, 3] + out[o, 5]/2>imgs_t.shape[3]/2:
                    new_h = imgs_t.shape[3]/2 - out[o, 3] + out[o, 5]/2
                    new_cy =  imgs_t.shape[3]/2 - new_h/2
                    out[o, 3] = int(new_cy)
                    out[o, 5] = int(new_h)
                if out[o, 2] - out[o, 4]/2<imgs_t.shape[2]/2:
                    new_w = out[o, 2] - imgs_t.shape[2]/2 + out[o, 4]/2
                    new_cx =  imgs_t.shape[2]/2 + new_w/2
                    out[o, 2] = int(new_cx)
                    out[o, 4] = int(new_w)

    else:
        out = torch.empty([0,7]) 
    
    return region_t, out, best_side




def transform_img_bboxes(out, best_side, region_t, transform_):
    out_ = copy.deepcopy(out)
    
    # fit the coordinates into the region-level reference instead of whole image
    if best_side == 'bottomleft':
        out_[:, 3] -= region_t.shape[3]
    if best_side == 'bottomright':
        out_[:, 2] -= region_t.shape[2]
        out_[:, 3] -= region_t.shape[3]
    if best_side == 'topright':
        out_[:, 2] -= region_t.shape[2]  

    # convert to [0, 1]
    for jj in range(out_.shape[0]):
        if out_[jj, 2] - out_[jj, 4]/2 < 0:  
            out_[jj, 4] = 2*out_[jj, 2]
        if out_[jj, 2] + out_[jj, 4]/2 > region_t.shape[2]:
            out_[jj, 4] = 2*(region_t.shape[2] - out_[jj, 2])
        if out_[jj, 3] - out_[jj, 5]/2 < 0:  
            out_[jj, 5] = 2*out_[jj, 3]
        if out_[jj, 3] + out_[jj, 5]/2 > region_t.shape[3]:
            out_[jj, 5] = 2*(region_t.shape[3] - out_[jj, 3])

    bboxes_ = out_
    bboxes_[:, 2:6] /= region_t.shape[2]
    region_t_np = region_t.squeeze(0).cpu().numpy()
    region_t_np = np.transpose(region_t_np, (1,2,0))

    bboxes_ = bboxes_[bboxes_[:, 4] > 0]
    bboxes_ = bboxes_[bboxes_[:, 5] > 0]

    if bboxes_.shape[0]:
        category_ids = [0]*bboxes_.shape[0]
        transformed = transform_(image=region_t_np, bboxes= bboxes_[:, 2:6], category_ids=category_ids)
        transformed_img =  np.transpose(transformed['image'], (2,0,1))
        bboxes_transformed = transformed['bboxes']
        bboxes_t = [list(bb) for bb in bboxes_transformed]
        bboxes_t = torch.FloatTensor(bboxes_t)    
        bboxes_t[:, [0, 2]] *= transformed_img.shape[2]
        bboxes_t[:, [1, 3]] *= transformed_img.shape[1]
        bboxes_[:, 2:6] = bboxes_t
        return transformed_img, bboxes_
    else: 
        return region_t.squeeze(0).cpu().numpy(), np.ones((1, 7))
    