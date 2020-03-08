# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# py-faster-rcnn
# Copyright (c) 2016 by Contributors
# Licence under The MIT License
# https://github.com/rbgirshick/py-faster-rcnn
# --------------------------------------------------------

import numpy as np
from utils.iou import vIoU


def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def traj_nms(trajs, thresh=0.3):
    if len(trajs) == 0:
        return []

    scores = np.array([traj['scr'] for traj in trajs])
    ratios = np.zeros(len(scores))
    for traj_idx, traj in enumerate(trajs):
        det_cnt = 0
        for det in traj['trj']:
            if det is not None:
                det_cnt += 1
        ratios[traj_idx] = det_cnt * 1.0 / len(traj['trj'])

    scores = scores * ratios
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        curr_traj = trajs[i]

        reserved_order_idx = []
        for j, traj_idx in enumerate(order[1:]):
            viou = vIoU(curr_traj['trj'], trajs[traj_idx]['trj'])
            if viou <= thresh:
                reserved_order_idx.append(j+1)
        order = order[reserved_order_idx]
    return keep

