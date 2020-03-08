# coding: utf-8
# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified byYuqing Zhu, Xizhou Zhu
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import numpy as np

import profile
import cv2
import time
import copy
import cPickle as pickle
import os

from nms import nms
           
NMS_THRESH = 0.3
IOU_THRESH = 0.8
MAX_THRESH=1e-2


def append_tid(dets_all):
    cls_num = len(dets_all)
    frame_num = len(dets_all[0])
    for cid in range(cls_num):
        for fid in range(frame_num):
            frame_dets = dets_all[cid][fid]
            if len(frame_dets) > 0:
                tids = np.ones((len(frame_dets), 1)) * -1
                frame_dets = np.concatenate((frame_dets, tids), axis=1)
                dets_all[cid][fid] = frame_dets


def delete_individuals(dets):

    cls_num = len(dets)
    for cid in range(cls_num):

        frame_num = len(dets[cid])
        for fid in range(frame_num):

            frame_dets = dets[cid][fid]
            if len(frame_dets) > 0:
                ids = np.where(frame_dets[:, -1] >= 0)[0]
                dets[cid][fid] = frame_dets[ids, :]


def createLinks(dets_all):
    links_all = []

    frame_num = len(dets_all[0])
    cls_num = len(dets_all)
    for cls_ind in range(cls_num):
        links_cls = []
        for frame_ind in range(frame_num - 1):
            # 取当前帧 与相邻后一帧
            dets1 = dets_all[cls_ind][frame_ind]
            dets2 = dets_all[cls_ind][frame_ind + 1]
            box1_num = len(dets1)
            box2_num = len(dets2)

            # 当前帧所有det的面积
            if frame_ind == 0:
                areas1 = np.empty(box1_num)
                for box1_ind, box1 in enumerate(dets1):
                    areas1[box1_ind] = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
            else:
                areas1 = areas2

            # 下一帧所有det的面积
            areas2 = np.empty(box2_num)
            for box2_ind, box2 in enumerate(dets2):
                areas2[box2_ind] = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

            links_frame = []
            # 当前帧的连接（根据iou）
            # 每个元素是一个det
            for box1_ind, box1 in enumerate(dets1):
                # 当前帧的每一个det，与下一帧的所有det比较iou
                area1 = areas1[box1_ind]
                x1 = np.maximum(box1[0], dets2[:, 0])
                y1 = np.maximum(box1[1], dets2[:, 1])
                x2 = np.minimum(box1[2], dets2[:, 2])
                y2 = np.minimum(box1[3], dets2[:, 3])
                w = np.maximum(0.0, x2 - x1 + 1)
                h = np.maximum(0.0, y2 - y1 + 1)
                inter = w * h
                ovrs = inter / (area1 + areas2 - inter)

                # |w1/h1 - w2/h2| < 0.3
                w1 = box1[2] - box1[0] + 1
                h1 = max(box1[3] - box1[1] + 1, 0.01)
                w2s = dets2[:, 2] - dets2[:, 0] + 1
                h2s = dets2[:, 3] - dets2[:, 1] + 1
                h2s = np.maximum(0.001, h2s)
                whr1 = w1 * 1.0 / h1
                whrs2 = w2s * 1.0 / h2s

                if whr1 > 1:
                    whr1 = 1.0 / whr1
                    whrs2 = np.maximum(0.001, whrs2)
                    whrs2 = 1.0 / whrs2

                whr_diff = np.abs(whrs2 - whr1)

                # 为当前帧当前det，保存下一帧可连接的所有det的id
                links_box = [ovr_ind for ovr_ind, ovr in enumerate(ovrs) if
                             ovr >= IOU_THRESH and whr_diff[ovr_ind] < 0.1]
                links_frame.append(links_box)
            links_cls.append(links_frame)
        links_all.append(links_cls)
    return links_all


def maxPath(dets_all, links_all):
    tid = 0
    for cls_ind, links_cls in enumerate(links_all):
        # 对每个类别的所有det

        max_begin = time.time()
        # 保存每一帧要删掉的det id
        delete_sets=[[]for i in range(0,len(dets_all[0]))]
        delete_single_box=[]
        dets_cls = dets_all[cls_ind]

        num_path=0

        # compute the number of links
        # link 总数
        sum_links=0
        for frame_ind, frame in enumerate(links_cls):
            for box_ind,box in enumerate(frame):
                sum_links+=len(box)

        while True:

            num_path+=1

            # 找到得分最高的一条trajectory
            rootindex, maxpath, maxsum = findMaxPath(links_cls, dets_cls,delete_single_box)

            # rootindex是trajectory的起始帧
            # maxpath是一个det list，指示了这条路径上的每个det id
            # maxsum是路径得分总数

            if maxsum<MAX_THRESH or sum_links==0 or len(maxpath) <1:
                break

            # ==== sunx ====
            connect_dets(dets_cls, rootindex, maxpath, tid)
            tid += 1
            # ==== sunx ====

            if len(maxpath)==1:
                delete=[rootindex,maxpath[0]]
                delete_single_box.append(delete)
            # 重新打分，traj上所有det的得分改为traj的均分
            rescore(dets_cls, rootindex, maxpath, maxsum)
            t4=time.time()

            # 保存要删除的det
            delete_set,num_delete=deleteLink(dets_cls, links_cls, rootindex, maxpath, NMS_THRESH)
            sum_links-=num_delete
            for i, box_ind in enumerate(maxpath):
                delete_set[i].remove(box_ind)
                delete_single_box.append([[rootindex+i],box_ind])
                for j in delete_set[i]:
                    dets_cls[i+rootindex][j]=np.zeros(6)
                delete_sets[i+rootindex]=delete_sets[i+rootindex]+delete_set[i]

        # 删除抑制的det
        for frame_idx,frame in enumerate(dets_all[cls_ind]):

            a=range(0,len(frame))
            keep=list(set(a).difference(set(delete_sets[frame_idx])))
            dets_all[cls_ind][frame_idx]=frame[keep,:]

    return dets_all


def findMaxPath(links,dets,delete_single_box):

    len_dets=[len(dets[i]) for i in xrange(len(dets))]
    max_boxes=np.max(len_dets)
    num_frame=len(links)+1
    a=np.zeros([num_frame,max_boxes])
    new_dets=np.zeros([num_frame,max_boxes])
    for delete_box in delete_single_box:
        new_dets[delete_box[0],delete_box[1]]=1
    if(max_boxes==0):
        max_path=[]
        return 0,max_path,0

    b=np.full((num_frame,max_boxes),-1)
    for l in xrange(len(dets)):
        for j in xrange(len(dets[l])):
            if(new_dets[l,j]==0):
                a[l,j]=dets[l][j][4]



    for i in xrange(1,num_frame):
        l1=i-1;
        for box_id,box in enumerate(links[l1]):
            for next_box_id in box:

                weight_new=a[i-1,box_id]+dets[i][next_box_id][4]
                if(weight_new>a[i,next_box_id]):
                    a[i,next_box_id]=weight_new
                    b[i,next_box_id]=box_id

    i,j=np.unravel_index(a.argmax(),a.shape)

    maxpath=[j]
    maxscore=a[i,j]
    while(b[i,j]!=-1):

            maxpath.append(b[i,j])
            j=b[i,j]
            i=i-1


    rootindex=i
    maxpath.reverse()
    return rootindex, maxpath, maxscore


def connect_dets(dets, rootindex, maxpath, tid):
    for i, det_id in enumerate(maxpath):
        dets[rootindex + i][det_id][5] = tid


def rescore(dets, rootindex, maxpath, maxsum):
    newscore = maxsum / len(maxpath)

    for i, box_ind in enumerate(maxpath):
        dets[rootindex + i][box_ind][4] = newscore


def deleteLink(dets, links, rootindex, maxpath, thesh):

    delete_set=[]
    num_delete_links=0

    for i, box_ind in enumerate(maxpath):
        areas = [(box[2] - box[0] + 1) * (box[3] - box[1] + 1) for box in dets[rootindex + i]]
        area1 = areas[box_ind]
        box1 = dets[rootindex + i][box_ind]
        x1 = np.maximum(box1[0], dets[rootindex + i][:, 0])
        y1 = np.maximum(box1[1], dets[rootindex + i][:, 1])
        x2 = np.minimum(box1[2], dets[rootindex + i][:, 2])
        y2 = np.minimum(box1[3], dets[rootindex + i][:, 3])
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        inter = w * h

        ovrs = inter / (area1 + areas - inter)
        #saving the box need to delete
        deletes = [ovr_ind for ovr_ind, ovr in enumerate(ovrs) if ovr >= 0.3]
        delete_set.append(deletes)

        #delete the links except for the last frame
        if rootindex + i < len(links):
            for delete_ind in deletes:
                num_delete_links+=len(links[rootindex+i][delete_ind])
                links[rootindex + i][delete_ind] = []

        if i > 0 or rootindex > 0:

            #delete the links which point to box_ind
            for priorbox in links[rootindex + i - 1]:
                for delete_ind in deletes:
                    if delete_ind in priorbox:
                        priorbox.remove(delete_ind)
                        num_delete_links+=1

    return delete_set,num_delete_links


def seq_nms(dets):
    append_tid(dets)
    links = createLinks(dets)
    dets=maxPath(dets, links)
    delete_individuals(dets)
    return dets


def seq_nms_nms(video, thresh, id=None):
    dets_all = seq_nms(video)
    for j in range(len(dets_all)):
        for frame_ind, dets in enumerate(dets_all[j]):
            keep = nms(dets, thresh)
            dets_all[j][frame_ind] = dets[keep, :]
    if id is not None:
        print('-- seq-nms: %d' % id)
    return dets_all


