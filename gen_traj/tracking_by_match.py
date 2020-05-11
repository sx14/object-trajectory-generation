import pdb
from collections import defaultdict
import numpy as np

from nms.nms import nms


def tracking_by_match(vid_dets, thr=0.6, max_traj_num=50):

    # init: nms, add tids
    for cls_idx, cls_dets in enumerate(vid_dets):
        for frm_idx, frm_dets in enumerate(cls_dets):
            if frm_dets is None or len(frm_dets) == 0:
                continue
            keep = nms(frm_dets, 0.3)
            pdb.set_trace()
            frm_dets = frm_dets[keep]
            frm_dets_new = np.zeros((frm_dets.shape[0], frm_dets.shape[1]+1))
            frm_dets_new[:, :frm_dets.shape[1]] = frm_dets
            frm_dets_new[:, -1] = -1
            cls_dets[frm_idx] = frm_dets_new

    curr_tid = 0
    tid2scr = {}
    tid2cnt = {}

    for cls_idx, cls_dets in enumerate(vid_dets):
        for frm_idx in range(len(cls_dets) - 1):
            curr_frm_dets = cls_dets[frm_idx]
            next_frm_dets = cls_dets[frm_idx + 1]

            for det in curr_frm_dets:
                if det[-1] == -1:
                    det[-1] = curr_tid
                    tid2scr[curr_tid] = det[4]
                    tid2cnt[curr_tid] = 1
                    curr_tid += 1

                if next_frm_dets is None or len(next_frm_dets) == 0:
                    continue
                curr_x1, curr_y1, curr_x2, curr_y2, tid = det
                next_x1s = next_frm_dets[:, 0]
                next_y1s = next_frm_dets[:, 1]
                next_x2s = next_frm_dets[:, 2]
                next_y2s = next_frm_dets[:, 3]

                i_x1s = np.maximum(curr_x1, next_x1s)
                i_y1s = np.maximum(curr_y1, next_y1s)
                i_x2s = np.minimum(curr_x2, next_x2s)
                i_y2s = np.minimum(curr_y2, next_y2s)

                u_x1s = np.minimum(curr_x1, next_x1s)
                u_y1s = np.minimum(curr_y1, next_y1s)
                u_x2s = np.maximum(curr_x2, next_x2s)
                u_y2s = np.maximum(curr_y2, next_y2s)

                i_areas = np.maximum((i_x2s - i_x1s + 1), 0) * np.maximum((i_y2s - i_y1s + 1), 0)
                u_areas = (u_x2s - u_x1s + 1) * (u_y2s - u_y1s + 1) - i_areas
                ious = i_areas / u_areas

                best_det_id = np.argmax(ious)
                if ious[best_det_id] > thr:
                    next_frm_dets[best_det_id, -1] = tid
                    tid2scr[tid] += next_frm_dets[best_det_id, 4]
                    tid2cnt[tid] += 1

            if frm_idx == len(cls_dets) - 1:
                for det in next_frm_dets:
                    if det[-1] == -1:
                        det[-1] = curr_tid
                        tid2scr[curr_tid] = det[-2]
                        tid2cnt[curr_tid] = 1
                        curr_tid += 1

    tid2conf = {}
    for tid in tid2cnt:
        traj_conf = tid2scr[tid] / tid2cnt[tid]
        if traj_conf >= 0.01:
            # tid2conf[tid] = (tid2scr[tid] / tid2cnt[tid] + tid2iou[tid] / tid2cnt[tid]) / 2
            tid2conf[tid] = (tid2scr[tid] * 1.0 / tid2cnt[tid] + tid2cnt[tid] * 10.0 / len(vid_dets[0]))

    reserved_tid_conf_list = sorted(tid2conf.items(), key=lambda item: item[1], reverse=True)[:max_traj_num]
    reserved_tids = {tid: conf for tid, conf in reserved_tid_conf_list}

    all_boxes = []
    for cls_idx in range(len(vid_dets)):
        cls_boxes = []
        for frm_idx in range(len(vid_dets[0])):
            cls_boxes.append({})
        all_boxes.append(cls_boxes)

    for cls_idx, cls_dets in enumerate(vid_dets):
        for frm_idx, frm_dets in enumerate(cls_dets):
            for det in frm_dets:
                tid = det[-1]
                if tid not in reserved_tids:
                    continue
                else:
                    det[4] = tid2conf[tid]
                    all_boxes[cls_idx][frm_idx][tid] = det.tolist()

    for cls_id in range(len(all_boxes)):
        cls_boxes = all_boxes[cls_id]
        for frm_id in range(len(cls_boxes)):
            cls_boxes[frm_id] = np.array(cls_boxes[frm_id].values())
    return all_boxes