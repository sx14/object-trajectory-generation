import os
import time
import json
from copy import deepcopy

import cv2
from tqdm import tqdm


def split_gt_instance(inst):

    split_insts = []
    traj = inst['trajectory']
    stt_fid = int(inst['start_fid'])
    last_fid = int(inst['start_fid'])
    fids = sorted(traj.keys())
    fids.append('%06d' % (int(fids[-1]) + 1))   # exclusive end
    for curr_fid in fids[1:]:
        if (last_fid + 1) == int(curr_fid) and int(curr_fid) < (int(inst['end_fid']) + 1):
            last_fid += 1
        else:
            end_fid = last_fid
            split_inst = deepcopy(inst)
            split_traj = {'%06d' % fid: traj['%06d' % fid] for fid in range(stt_fid, end_fid+1)}
            split_inst['start_fid'] = stt_fid
            split_inst['end_fid'] = end_fid
            split_inst['trajectory'] = split_traj
            split_insts.append(split_inst)

            stt_fid = int(curr_fid)
            last_fid = int(curr_fid)
    return split_insts


def to_pred(gt, pred_root, data_root):
    print('Generating detection results with gt ...')
    time.sleep(2)
    out = {}
    for pid_vid in tqdm(gt):
        pid, vid = pid_vid.split('/')
        frame_path = os.path.join(data_root, pid, vid, '000000.JPEG')
        im = cv2.imread(frame_path)
        im_h, im_w = im.shape[:2]
        gt_insts = gt[pid_vid]
        gt2pr_insts = []
        for gt_inst in gt_insts:

            gt_inst['score'] = 1.0
            gt_inst['start_fid'] = min([int(fid_str) for fid_str in gt_inst['trajectory']])
            gt_inst['end_fid'] = max([int(fid_str) for fid_str in gt_inst['trajectory']])
            gt_inst['height'] = im_h
            gt_inst['width'] = im_w
            gt2pr_insts += split_gt_instance(gt_inst)

        out['%s/%s' % (pid, vid)] = gt2pr_insts
    out = {'version': 'VERSION 1.0', 'results': out}
    with open(pred_root, 'w') as f:
        json.dump(out, f)


dataset = ('vidor_hoid_mini', 'VidOR-HOID-mini')
split = 'val'

gt_path = '../data/%s/%s_%s_object_gt.json' % (dataset[0], dataset[0], split)
data_root = '../data/%s/Data/VID/%s' % (dataset[1], split)
pred_path = '../output/%s/object_trajectories_%s_gt2det.json' % (dataset[0], split)

print('Loading %s' % gt_path)
with open(gt_path) as f:
    gt = json.load(f)

to_pred(gt, pred_path, data_root)