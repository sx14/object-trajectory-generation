import os
import time
import json
from collections import defaultdict
import cv2
import numpy as np


def load_fgfa_raw_detection(raw_det_path):
    print('Loading ...')
    time.sleep(1)
    """ Load raw detections (without NMS) of FGFA """
    import cPickle
    with open(raw_det_path) as f:
        all_raw_dets, frame_ids = cPickle.load(f)
    # ignore background
    return all_raw_dets[1:], frame_ids


def load_txt_result(out_txt_path):
    fid2dets = defaultdict(list)
    with open(out_txt_path) as f:
        lines = f.readlines()
        dets = [line.strip().split(' ') for line in lines]

    for det in dets:
        fid, cls_idx, conf, x1, y1, x2, y2, tid = det
        fid2dets[int(fid)].append([float(x1), float(y1), float(x2), float(y2), int(cls_idx), float(conf)])
    return fid2dets


def output_txt_result(all_dets, frame_ids, output_path):
    print('Output ...')
    time.sleep(1)
    """ Output results with FGFA output format """
    with open(output_path, 'wt') as f:
        for im_ind in range(len(frame_ids)):
            for cls_ind in range(len(all_dets)):
                dets = all_dets[cls_ind][im_ind]
                if len(dets) == 0:
                    continue
                for k in range(len(dets)):
                    det = dets[k]
                    f.write('{:d} {:d} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f} {:d}\n'.
                            format(frame_ids[im_ind], cls_ind+1, det[4],
                                   det[0], det[1], det[2], det[3], int(det[-1])))


def load_vid_frame_nums(video_idx_path):
    with open(video_idx_path) as f:
        lines = f.readlines()
        video_ids = [line.split(' ')[0] for line in lines]
        video_len = [int(line.split(' ')[-1]) for line in lines]
    return video_ids, video_len


def load_vid_frame_idx(frame_idx_path):
    with open(frame_idx_path) as f:
        lines = f.readlines()
        frame_ids = [line.split(' ')[0] for line in lines]
        frame_idxs = [line.split(' ')[1] for line in lines]
    return frame_ids, frame_idxs


def output_json_result(imageset_path, res_paths, output_path, category_list, data_root):
    max_per_video = 50
    score_thr = 0.00

    # load frame-idx
    with open(imageset_path) as f:
        raw_frame_list = f.readlines()
        frame_list = [l.strip() for l in raw_frame_list]
        idx2frame = {}
        for frame_rec in frame_list:
            frame, idx = frame_rec.split(' ')
            idx2frame[int(idx)] = frame

    # load results
    res = []
    for res_path in res_paths:
        with open(res_path) as f:
            lines = f.readlines()
            line_splits = [line.strip().split(' ') for line in lines]
            res_part = [[float(v) for v in line_split]
                        for line_split in line_splits]
            res += res_part

    # output data
    pred_output = {'version': 'VERSION 1.0', 'results': {}}
    pred_results = pred_output['results']  # video_id -> dets

    for det in res:
        frame_idx, cls_idx, conf, x1, y1, x2, y2, tid = det
        frame_idx = int(frame_idx)
        cls_idx = int(cls_idx)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        tid = int(tid)

        frame_info = idx2frame[frame_idx].split(' ')[0].split('/')
        frame_id = frame_info[-1]
        video_id = '/'.join(frame_info[1:-1])
        frame_path = os.path.join(data_root, idx2frame[frame_idx].split(' ')[0]+'.JPEG')

        if video_id not in pred_results:
            im = cv2.imread(frame_path)
            im_h, im_w, _ = im.shape
            # new video
            video = {'trajectory': {}, 'height': im_h, 'width': im_w}
            pred_results[video_id] = video

        video = pred_results[video_id]
        trajs = video['trajectory']

        if tid in trajs:
            traj = trajs[tid]
        else:
            traj = {}   # fid -> det
            trajs[tid] = traj

        traj[frame_id] = [x1, y1, x2, y2, conf, cls_idx]

    det_num = 0
    for video_id in pred_results:
        video = pred_results[video_id]
        trajs = video['trajectory']
        for tid in trajs:
            traj = trajs[tid]
            det_num += len(traj.keys())
    print(det_num)

    for video_id in pred_results:
        det_num = 0
        video = pred_results[video_id]
        trajs = video['trajectory']

        video_dets = []
        for tid in trajs:
            traj = trajs[tid]
            # print('T[%d]: %d' % (tid, len(traj)))
            det_num += len(traj)
            conf_sum = 0.0  # for avg
            cls_count = np.zeros(len(category_list))   # voting

            for frame_id in traj:
                det = traj[frame_id]
                conf_sum += det[4]
                cls_count[det[5]] += 1
                traj[frame_id] = det[:4]    #[x1,y1,x2,y2]

            cls_ind = np.argmax(cls_count)
            category = category_list[cls_ind]
            score = conf_sum / len(traj.keys())

            if score < score_thr:
                continue

            fids = sorted([int(fid) for fid in traj.keys()])
            video_det = {
                'category': category,
                'score': score,
                'trajectory': traj,
                'org_start_fid': fids[0],
                'org_end_fid': fids[-1],
                'start_fid': fids[0],
                'end_fid': fids[-1],
                'width': video['width'],
                'height': video['height']
            }
            video_dets.append(video_det)

        video_dets = sorted(video_dets, key=lambda det: det['score'], reverse=True)
        video_dets = video_dets[:max_per_video]
        print('%s: %d %d' % (video_id, len(video_dets), det_num))
        pred_results[video_id] = video_dets

    with open(output_path, 'w') as f:
        json.dump(pred_output, f)