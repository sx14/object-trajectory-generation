import time
import numpy as np

from utils.iou import overlap_ratio


def greedy_merge(all_seg_trajs, num_cls, num_frm, num_seg_frm, max_traj_num=50):
    """ Merge trajectories of segments """
    print('Merging ...')
    time.sleep(1)

    # NMS
    # all_seg_trajs.append([])
    # seg_all_cls_trajs = defaultdict(list)
    # for seg_traj in seg_trajs:
    #     seg_all_cls_trajs[seg_traj['cls']].append(seg_traj)
    # for cls in seg_all_cls_trajs:
    #     seg_cls_trajs = seg_all_cls_trajs[cls]
    #     keep = traj_nms(seg_cls_trajs)
    #     for k in keep:
    #         all_seg_trajs[seg_idx].append(seg_cls_trajs[k])

    # === create links ===
    # init links
    # all_links[seg_id][curr_trj_id] = [[next_trj_id, overlap]]
    all_links = []
    for seg_idx in range(len(all_seg_trajs)):
        seg_traj_links = []
        for traj_idx in range(len(all_seg_trajs[seg_idx])):
            seg_traj_links.append([])
        all_links.append(seg_traj_links)

    for seg_idx in range(len(all_seg_trajs) - 1):
        seg1_trajs = all_seg_trajs[seg_idx]
        seg2_trajs = all_seg_trajs[seg_idx + 1]

        for traj1_idx, traj1 in enumerate(seg1_trajs):
            for traj2_idx, traj2 in enumerate(seg2_trajs):
                if traj1['cls'] == traj2['cls']:
                    # print('seg_id:%d, trj1_id:%d, trj2_id:%d' % (seg_idx, traj1_idx, traj2_idx))
                    overlap = overlap_ratio(traj1['trj'], traj2['trj'])
                    if overlap >= 0.5:
                        all_links[seg_idx][traj1_idx].append([traj2_idx, overlap])

    # === greedily link ===
    tid2iou = {}
    tid2scr = {}
    tid2cnt = {}

    # init linked trajectories
    next_linked_tid = 0
    for traj in all_seg_trajs[0]:
        traj['tid'] = next_linked_tid
        tid2iou[next_linked_tid] = 0
        tid2scr[next_linked_tid] = traj['scr']
        tid2cnt[next_linked_tid] = 1
        next_linked_tid += 1

    for seg_idx in range(len(all_links)-1):
        # curr -> next
        hit_next_seg_traj_idx = set()
        for seg_traj_idx in range(len(all_links[seg_idx])):
            seg_traj_tid = all_seg_trajs[seg_idx][seg_traj_idx]['tid']
            seg_traj_links = all_links[seg_idx][seg_traj_idx]
            if len(seg_traj_links) > 0:
                best_link = max(seg_traj_links, key=lambda link: link[1])
                best_next_traj_idx = best_link[0]
                best_next_traj_iou = best_link[1]

                # if best_next_traj_idx in hit_next_seg_traj_idx:
                #     continue

                best_next_traj = all_seg_trajs[seg_idx + 1][best_next_traj_idx]
                hit_next_seg_traj_idx.add(best_next_traj_idx)

                best_next_traj['tid'] = seg_traj_tid
                best_next_traj['iou'] = best_next_traj_iou

                tid2iou[seg_traj_tid] += best_next_traj_iou
                tid2scr[seg_traj_tid] += all_seg_trajs[seg_idx + 1][best_next_traj_idx]['scr']
                tid2cnt[seg_traj_tid] += 1

        # new tid for unlinked next segment trajs
        for next_seg_traj_idx in range(len(all_links[seg_idx + 1])):
            if next_seg_traj_idx not in hit_next_seg_traj_idx:
                new_traj = all_seg_trajs[seg_idx + 1][next_seg_traj_idx]
                new_traj['tid'] = next_linked_tid
                tid2iou[next_linked_tid] = 0
                tid2scr[next_linked_tid] = new_traj['scr']
                tid2cnt[next_linked_tid] = 1
                next_linked_tid += 1

    # === filter ===
    tid2conf = {}
    for tid in tid2cnt:
        traj_conf = tid2scr[tid] / tid2cnt[tid]
        if traj_conf >= 0.01:
            # tid2conf[tid] = (tid2scr[tid] / tid2cnt[tid] + tid2iou[tid] / tid2cnt[tid]) / 2
            tid2conf[tid] = (tid2scr[tid] * 1.0 / tid2cnt[tid] + tid2cnt[tid] / len(all_seg_trajs) * 10.0)

    reserved_tid_conf_list = sorted(tid2conf.items(), key=lambda item: item[1], reverse=True)[:max_traj_num]
    reserved_tids = {tid: conf for tid, conf in reserved_tid_conf_list}

    # === merge ===
    all_boxes = []
    for cls_idx in range(num_cls):
        cls_boxes = []
        for frm_idx in range(num_frm):
            cls_boxes.append({})
        all_boxes.append(cls_boxes)

    for seg_idx in range(len(all_seg_trajs)):
        stt_fid = seg_idx * (num_seg_frm / 2)
        for traj in all_seg_trajs[seg_idx]:
            trj = traj['trj']
            cls = traj['cls']
            tid = traj['tid']
            if tid not in reserved_tids:
                continue
            for seg_frm_idx, trj_det in enumerate(trj):
                if trj_det is not None:
                    tid2det = all_boxes[cls][stt_fid + seg_frm_idx]
                    if tid in tid2det:
                        exist_det = tid2det[tid]
                        merge_det = (exist_det + trj_det) / 2
                    else:
                        merge_det = trj_det
                    merge_det[4] = tid2conf[tid]
                    merge_det[5] = tid
                    tid2det[tid] = merge_det.tolist()

    for cls_id in range(len(all_boxes)):
        cls_boxes = all_boxes[cls_id]
        for frm_id in range(len(cls_boxes)):
            cls_boxes[frm_id] = np.array(cls_boxes[frm_id].values())
    return all_boxes