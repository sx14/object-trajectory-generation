import time


def _gen_segments(all_dets, seg_len):
    print('Segmenting ...')
    time.sleep(1)
    """ Split all detections into segments """
    segs = []

    num_frames = len(all_dets[0])
    for stt_fid in range(0, num_frames, seg_len / 2):
        end_fid = stt_fid + seg_len
        seg_dets = [all_dets[c][stt_fid: end_fid] for c in range(len(all_dets))]
        segs.append(seg_dets)
        if end_fid >= num_frames:
            break
    return segs


def track_in_video(all_dets_raw, seg_track_method, merge_method, frame_root, max_traj_num=50, num_seg_frm=30):
    """ Process one video """
    all_segs = _gen_segments(all_dets_raw, num_seg_frm)
    all_seg_trajs = seg_track_method(all_segs, frame_root)
    # trajs = [{'cls': cls, 'scr': scr, 'trj': [np.array([x1,x2,x3,x4,scr,-1])]}]

    num_cls = len(all_segs[0])
    num_frm = len(all_dets_raw[0])
    all_traj_dets = merge_method(all_seg_trajs, num_cls, num_frm, num_seg_frm, max_traj_num)
    return all_traj_dets



