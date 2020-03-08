import os
import time
from nms.seq_nms import seq_nms_nms


# def seqnms_track(segs):
#     print('Tracking')
#     """ Generate short trajectories for segments """
#     for i in tqdm(range(len(segs))):
#         seg_dets = segs[i]
#         segs[i] = seq_nms_nms(seg_dets, 0.3)
#     return segs


def _gen_trajectories(seg_dets):

    tid2trj = {}
    tid2cls = {}
    tid2scr = {}
    tid2cnt = {}

    for cls_id in range(len(seg_dets)):
        for seg_fid in range(len(seg_dets[cls_id])):
            for det in seg_dets[cls_id][seg_fid]:
                tid = det[5]
                box = det[:5]
                if tid not in tid2trj:
                    tid2cls[tid] = cls_id
                    tid2trj[tid] = [None] * len(seg_dets[cls_id])
                    tid2scr[tid] = 0
                    tid2cnt[tid] = 0

                det[-1] = -1
                tid2trj[tid][seg_fid] = det
                tid2scr[tid] += box[4]
                tid2cnt[tid] += 1

    trajs = []
    for tid, traj in tid2trj.items():
        scr = tid2scr[tid] / tid2cnt[tid]
        if traj[len(traj) / 2] is not None and scr > 0.05:
            trajs.append({'trj': traj,
                          'cls': tid2cls[tid],
                          'scr': scr})
    return trajs


def seqnms_track(all_segs, frame_root):
    print('Tracking ...')
    time.sleep(1)
    """ Generate short trajectories for segments """

    from multiprocessing.pool import Pool as Pool
    from multiprocessing import cpu_count
    cpu_num = min(20, cpu_count())
    pool = Pool(processes=cpu_num)
    results = [pool.apply_async(seq_nms_nms, args=(seg_dets, 0.3, seg_idx))
               for seg_idx, seg_dets in enumerate(all_segs)]
    pool.close()
    pool.join()

    for r, res in enumerate(results):
        all_segs[r] = res.get()

    all_seg_trajs = []
    for seg_idx, seg_dets in enumerate(all_segs):
        seg_trajs = _gen_trajectories(seg_dets)
        all_seg_trajs.append(seg_trajs)
    return all_seg_trajs


