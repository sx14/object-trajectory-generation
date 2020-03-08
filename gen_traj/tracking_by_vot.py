import os
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import dlib
from nms.nms import nms, traj_nms
from vis.show_frame import show_boxes


def _track_dlib(frames, frame_idx, init_det, vis=False):
    im_h, im_w = frames[frame_idx].shape[:2]

    # init traj
    traj = [None] * len(frames)
    traj[frame_idx] = np.array([init_det[0], init_det[1],
                                init_det[2], init_det[3],
                                init_det[4], -1])

    # create tracker
    tracker = dlib.correlation_tracker()
    init_scr = init_det[4]

    if vis:
        plt.figure(0)
    # forward
    tracker.start_track(frames[frame_idx], dlib.rectangle(int(init_det[0]), int(init_det[1]),
                                                          int(init_det[2]), int(init_det[3])))
    for f in range(frame_idx+1, len(frames)):
        frame = frames[f]
        tracker.update(frame)
        roi = tracker.get_position()

        box = [int(roi.left()),
               int(roi.top()),
               int(roi.right()),
               int(roi.bottom())]
        box = [max(0, box[0]),
               max(0, box[1]),
               max(0, box[2]),
               max(0, box[3])]
        box = [min(box[0], im_w - 1),
               min(box[1], im_h - 1),
               min(box[2], im_w - 1),
               min(box[3], im_h - 1), init_scr, -1]
        traj[f] = np.array(box)


        if vis:
            plt.ion()
            plt.axis('off')
            plt.imshow(frame[:,:,::-1])
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0],
                                 box[3] - box[1], fill=False, edgecolor=(1, 0, 0), linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.show()
            plt.pause(0.1)
            plt.cla()

    if vis:
        plt.close()

    if vis:
        plt.figure(0)

    # backward
    tracker.start_track(frames[frame_idx], dlib.rectangle(int(init_det[0]), int(init_det[1]),
                                                          int(init_det[2]), int(init_det[3])))
    for f in range(frame_idx-1, -1, -1):
        frame = frames[f]
        tracker.update(frame)
        roi = tracker.get_position()

        box = [int(roi.left()),
               int(roi.top()),
               int(roi.right()),
               int(roi.bottom())]
        box = [max(0, box[0]),
               max(0, box[1]),
               max(0, box[2]),
               max(0, box[3])]
        box = [min(box[0], im_w - 1),
               min(box[1], im_h - 1),
               min(box[2], im_w - 1),
               min(box[3], im_h - 1), init_scr, -1]
        traj[f] = np.array(box)

        if vis:
            plt.ion()
            plt.axis('off')
            plt.imshow(frame[:, :, ::-1])
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0],
                                 box[3] - box[1], fill=False, edgecolor=(1, 0, 0), linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.show()
            plt.pause(0.1)
            plt.cla()

    if vis:
        plt.close()

    return traj


def _track_kcf(frames, frame_idx, init_det, vis=False):
    im_h, im_w = frames[frame_idx].shape[:2]

    # init traj
    traj = [None] * len(frames)
    traj[frame_idx] = np.array([init_det[0], init_det[1],
                                init_det[2], init_det[3],
                                init_det[4], -1])

    # create tracker
    tracker = cv2.TrackerKCF_create()
    init_box = (init_det[0],
                init_det[1],
                init_det[2] - init_det[0] + 1,
                init_det[3] - init_det[1] + 1)
    init_scr = init_det[-1]

    if vis:
        plt.figure(0)
    # forward
    tracker.init(frames[frame_idx], init_box)
    for f in range(frame_idx+1, len(frames)):
        frame = frames[f]
        ok, box = tracker.update(frame)
        if ok:
            box = [int(box[0]),
                   int(box[1]),
                   int(box[0] + box[2]),
                   int(box[1] + box[3])]
            box = [max(0, box[0]),
                   max(0, box[1]),
                   max(0, box[2]),
                   max(0, box[3])]
            box = [min(box[0], im_w - 1),
                   min(box[1], im_h - 1),
                   min(box[2], im_w - 1),
                   min(box[3], im_h - 1), init_scr, -1]
            traj[f] = np.array(box)
        else:
            # print('F: %d fail!!!' % f)
            break

        if vis:
            plt.ion()
            plt.axis('off')
            plt.imshow(frame[:,:,::-1])
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0],
                                 box[3] - box[1], fill=False, edgecolor=(1, 0, 0), linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.show()
            plt.pause(0.1)
            plt.cla()

    if vis:
        plt.close()

    if vis:
        plt.figure(0)

    # backward
    tracker.init(frames[frame_idx], init_box)
    for f in range(frame_idx-1, -1, -1):
        frame = frames[f]
        ok, box = tracker.update(frame)
        if ok:
            box = [int(box[0]),
                   int(box[1]),
                   int(box[0] + box[2]),
                   int(box[1] + box[3])]
            box = [max(0, box[0]),
                   max(0, box[1]),
                   max(0, box[2]),
                   max(0, box[3])]
            box = [min(box[0], im_w - 1),
                   min(box[1], im_h - 1),
                   min(box[2], im_w - 1),
                   min(box[3], im_h - 1), init_scr, -1]
            traj[f] = np.array(box)

            if vis:
                plt.ion()
                plt.axis('off')
                plt.imshow(frame[:, :, ::-1])
                rect = plt.Rectangle((box[0], box[1]), box[2] - box[0],
                                     box[3] - box[1], fill=False, edgecolor=(1, 0, 0), linewidth=3.5)
                plt.gca().add_patch(rect)
                plt.show()
                plt.pause(0.1)
                plt.cla()

        else:
            # print('B: %d fail!!!' % f)
            break

    if vis:
        plt.close()

    return traj


def _merge(traj, cls_idx, seg_dets, seg_det_infos, seg_det_flags):
    cls_frm_dets = [[] for _ in range(len(traj))]
    cls_frm_det_idxs = [[] for _ in range(len(traj))]

    # collect segment detections
    for det_idx in range(len(seg_dets)):
        det_info = seg_det_infos[det_idx]
        if det_info['cls_idx'] == cls_idx:
            det_frm_idx = det_info['frm_idx']
            cls_frm_dets[det_frm_idx].append(seg_dets[det_idx])
            cls_frm_det_idxs[det_frm_idx].append(det_idx)

    traj_scr = 0.0
    for i in range(len(traj)):
        trj_box = traj[i]

        if len(cls_frm_dets[i]) == 0 or trj_box is None:
            continue

        txmin, tymin, txmax, tymax = trj_box[:4]
        tarea = (tymax - tymin + 1) * (txmax - txmin + 1)

        det_boxes = np.array(cls_frm_dets[i])
        dxmins = det_boxes[:, 0]
        dymins = det_boxes[:, 1]
        dxmaxs = det_boxes[:, 2]
        dymaxs = det_boxes[:, 3]
        dareas = (dxmaxs - dxmins + 1) * (dymaxs - dymins + 1)

        ixmins = np.maximum(dxmins, txmin)
        ixmaxs = np.minimum(dxmaxs, txmax)
        iymins = np.maximum(dymins, tymin)
        iymaxs = np.minimum(dymaxs, tymax)

        ws = np.maximum(0.0, ixmaxs - ixmins + 1)
        hs = np.maximum(0.0, iymaxs - iymins + 1)
        inters = ws * hs
        ovrs = inters / (dareas + tarea - inters)

        max_ovr_idx = np.argmax(ovrs)
        max_ovr = ovrs[max_ovr_idx]

        # update traj box
        if max_ovr >= 0.7:
            trj_box[:5] = det_boxes[max_ovr_idx][:5]
        traj[i] = trj_box
        traj_scr += traj[i][4]

        # remove detections overlapped
        rm_inds = np.where(ovrs > 0.3)[0]
        for rm_ind in rm_inds:
            seg_det_flags[cls_frm_det_idxs[i][rm_ind]] = False

    traj_scr = traj_scr / len(traj)
    return traj, traj_scr, seg_det_flags


def _nms_filter(all_seg_cls_frm_dets, max_per_frame=20):
    all_seg_frm_cls_dets = []
    for seg_idx in range(len(all_seg_cls_frm_dets)):
        seg_dets = all_seg_cls_frm_dets[seg_idx]

        all_seg_frm_cls_dets.append([])
        cls_num = len(seg_dets)
        seg_len = len(seg_dets[0])

        # init new container
        for seg_frm_idx in range(seg_len):
            all_seg_frm_cls_dets[seg_idx].append([])
            for j in range(cls_num):
                all_seg_frm_cls_dets[seg_idx][seg_frm_idx].append([])

        # perform NMS
        for cls_idx, cls_dets in enumerate(seg_dets):
            for seg_frm_idx, frm_dets in enumerate(cls_dets):
                keep = nms(frm_dets, 0.3)
                all_seg_frm_cls_dets[seg_idx][seg_frm_idx][cls_idx] = [frm_dets[i]
                                                                       for i in keep if frm_dets[i][4] >= 0.1]
        # reserve top N per frame
        for seg_frm_idx in range(seg_len):
            frm_dets = []
            for cls_idx in range(cls_num):
                seg_frm_cls_dets = all_seg_frm_cls_dets[seg_idx][seg_frm_idx][cls_idx]
                for det in seg_frm_cls_dets:
                    frm_dets.append({'det': det, 'scr': det[-1], 'cls': cls_idx})
            sorted_frm_dets = sorted(frm_dets, key=lambda item: item['scr'], reverse=True)[:max_per_frame]

            frm_cls_dets = [[] for _ in range(cls_num)]
            for frm_det in sorted_frm_dets:
                frm_cls_dets[frm_det['cls']].append(frm_det['det'])
            for cls_idx in range(cls_num):
                all_seg_frm_cls_dets[seg_idx][seg_frm_idx][cls_idx] = frm_cls_dets[cls_idx]

    return all_seg_frm_cls_dets


def vot_track(segs, frame_root):
    print('Tracking ...')

    # load frames
    sorted_frame_ids = sorted(os.listdir(frame_root))
    sorted_frames = [cv2.imread(os.path.join(frame_root, frame_id)) for frame_id in sorted_frame_ids]

    # resize
    frame_hw = sorted_frames[0].shape[:2]
    resize_ratio = 320.0 / max(frame_hw)
    resize_ratio = min(resize_ratio, 1.0)
    if resize_ratio < 1.0:
        print('resizing frames and boxes ...')
        sorted_frames = [cv2.resize(frame, (int(frame_hw[1] * resize_ratio),
                                            int(frame_hw[0] * resize_ratio))) for frame in sorted_frames]
        frame_hw = sorted_frames[0].shape[:2]

        # Attention: don't repeat resizing the boxes in overlapped segment
        segs = deepcopy(segs)
        for seg_idx, seg_dets in enumerate(segs):
            seg_len = len(seg_dets[0]) / 2
            if seg_idx == (len(segs) - 1):
                seg_len = seg_len * 2
            for cls_dets in seg_dets:
                for seg_frm_idx in range(seg_len):
                    frm_dets = cls_dets[seg_frm_idx]
                    for det in frm_dets:
                        det[:4] = det[:4] * resize_ratio

    # ====== debug ======
    # for s, seg_dets in enumerate(segs):
    #     seg_len = len(seg_dets[0])
    #     for seg_frm_idx in range(seg_len / 2):
    #         frm_dets = []
    #         frm_det_clses = []
    #         for cls_idx in range(len(seg_dets)):
    #             a = seg_dets[cls_idx][seg_frm_idx]
    #             frm_dets += a.tolist()
    #             frm_det_clses += [cls_idx] * len(a)
    #         frm_idx = s * seg_len / 2 + seg_frm_idx
    #         show_boxes(sorted_frames[frm_idx], frm_dets, frm_det_clses, [(1,0,0)] * len(frm_dets))
    # ===================

    print('frame size: %d x %d -- %.2f' % (frame_hw[0], frame_hw[1], resize_ratio))

    # collect, nms and filter
    all_seg_frm_cls_dets = _nms_filter(segs, max_per_frame=20)

    # ====== debug ======
    # for s, seg_dets in enumerate(all_seg_frm_cls_dets):
    #     seg_len = len(seg_dets)
    #     for seg_frm_idx in range(seg_len / 2):
    #         frm_dets = []
    #         frm_det_clses = []
    #         for cls_idx in range(len(seg_dets[0])):
    #             a = seg_dets[seg_frm_idx][cls_idx]
    #             frm_dets += a
    #             frm_det_clses += [cls_idx] * len(a)
    #         frm_idx = s * seg_len / 2 + seg_frm_idx
    #         print(frm_idx)
    #         show_boxes(sorted_frames[frm_idx], frm_dets, frm_det_clses, [(1,0,0)] * len(frm_dets))
    # ===================

    # generate segment trajs by tracking
    all_trajs = []
    past_frm_num = 0
    seg_num = len(all_seg_frm_cls_dets)
    for seg_idx in range(seg_num):

        seg_frm_cls_dets = all_seg_frm_cls_dets[seg_idx]
        seg_len = len(seg_frm_cls_dets)
        seg_frames = sorted_frames[past_frm_num: past_frm_num + seg_len]
        assert seg_len == len(seg_frames)

        seg_dets = []
        seg_det_infos = []
        for frm_idx, frm_dets in enumerate(seg_frm_cls_dets):
            for cls_idx, cls_dets in enumerate(frm_dets):
                seg_dets += cls_dets
                seg_det_infos += [{'frm_idx': frm_idx, 'cls_idx': cls_idx}] * len(cls_dets)

        seg_tck_cnt = 0
        seg_tck_cnt_org = len(seg_dets)
        seg_trajs = []

        if seg_tck_cnt_org > 0:

            seg_det_flags = [True] * len(seg_dets)
            seg_dets = np.array(seg_dets)
            seg_det_orders = np.argsort(seg_dets[:, 4])[::-1]
            seg_dets = seg_dets[seg_det_orders]
            seg_det_infos = np.array(seg_det_infos)
            seg_det_infos = seg_det_infos[seg_det_orders]



            for seg_det_idx in range(len(seg_dets)):
                det = seg_dets[seg_det_idx]
                det_info = seg_det_infos[seg_det_idx]
                det_flag = seg_det_flags[seg_det_idx]

                if not det_flag:
                    continue

                seg_tck_cnt += 1
                print('Frame %s | Cate %s | Score %s' % (str(det_info['frm_idx']).rjust(3),
                                                         str(det_info['cls_idx']).rjust(2), '%.2f' % det[4]))
                # traj = _track_kcf(seg_frames, seg_frm_idx, det, vis=False)
                traj = _track_dlib(seg_frames, det_info['frm_idx'], det, vis=False)
                trk_cnt_org = sum(seg_det_flags)
                traj, traj_scr, seg_det_flags = _merge(traj, det_info['cls_idx'], seg_dets, seg_det_infos, seg_det_flags)
                trk_cnt_cur = sum(seg_det_flags)
                print('  `--' + str(trk_cnt_org).rjust(5) + ' -> %d' % trk_cnt_cur)

                seg_trajs.append({
                    'trj': traj,
                    'scr': traj_scr,
                    'cls': det_info['cls_idx']
                })

        all_trajs.append(seg_trajs)
        print('[%d/%d] %.2f / %.2f' % (seg_idx + 1, seg_num,
                                       (seg_tck_cnt * 1.0 / seg_len),
                                       (seg_tck_cnt_org * 1.0 / seg_len)))
        past_frm_num += seg_len / 2

    # resize
    if resize_ratio < 1.0:
        for seg_trajs in all_trajs:
            for traj in seg_trajs:
                trj = traj['trj']
                for box in trj:
                    if box is not None:
                        box[:4] = box[:4] / resize_ratio

    return all_trajs


def vot_track2(segs, frame_root):
    print('Tracking ...')

    # load frames
    sorted_frame_ids = sorted(os.listdir(frame_root))
    sorted_frames = [cv2.imread(os.path.join(frame_root, frame_id)) for frame_id in sorted_frame_ids]

    # resize
    frame_hw = sorted_frames[0].shape[:2]
    resize_ratio = 320.0 / max(frame_hw)
    resize_ratio = min(resize_ratio, 1.0)
    if resize_ratio < 1.0:
        print('resizing frames and boxes ...')
        sorted_frames = [cv2.resize(frame, (int(frame_hw[1] * resize_ratio),
                                            int(frame_hw[0] * resize_ratio))) for frame in sorted_frames]
        frame_hw = sorted_frames[0].shape[:2]

        # Attention: don't repeat resizing the boxes in overlapped segment
        segs = deepcopy(segs)
        for seg_idx, seg_dets in enumerate(segs):
            seg_len = len(seg_dets[0]) / 2
            if seg_idx == (len(segs) - 1):
                seg_len = seg_len * 2
            for cls_dets in seg_dets:
                for seg_frm_idx in range(seg_len):
                    frm_dets = cls_dets[seg_frm_idx]
                    for det in frm_dets:
                        det[:4] = det[:4] * resize_ratio

    # ====== debug ======
    # for s, seg_dets in enumerate(segs):
    #     seg_len = len(seg_dets[0])
    #     for seg_frm_idx in range(seg_len / 2):
    #         frm_dets = []
    #         frm_det_clses = []
    #         for cls_idx in range(len(seg_dets)):
    #             a = seg_dets[cls_idx][seg_frm_idx]
    #             frm_dets += a.tolist()
    #             frm_det_clses += [cls_idx] * len(a)
    #         frm_idx = s * seg_len / 2 + seg_frm_idx
    #         show_boxes(sorted_frames[frm_idx], frm_dets, frm_det_clses, [(1,0,0)] * len(frm_dets))
    # ===================

    print('frame size: %d x %d -- %.2f' % (frame_hw[0], frame_hw[1], resize_ratio))

    # collect, nms and filter
    all_seg_frm_cls_dets = _nms_filter(segs, max_per_frame=20)

    # ====== debug ======
    # for s, seg_dets in enumerate(all_seg_frm_cls_dets):
    #     seg_len = len(seg_dets)
    #     for seg_frm_idx in range(seg_len / 2):
    #         frm_dets = []
    #         frm_det_clses = []
    #         for cls_idx in range(len(seg_dets[0])):
    #             a = seg_dets[seg_frm_idx][cls_idx]
    #             frm_dets += a
    #             frm_det_clses += [cls_idx] * len(a)
    #         frm_idx = s * seg_len / 2 + seg_frm_idx
    #         print(frm_idx)
    #         show_boxes(sorted_frames[frm_idx], frm_dets, frm_det_clses, [(1,0,0)] * len(frm_dets))
    # ===================

    # generate segment trajs by tracking
    all_trajs = []
    past_frm_num = 0
    seg_num = len(all_seg_frm_cls_dets)
    for seg_idx in range(seg_num):

        seg_frm_cls_dets = all_seg_frm_cls_dets[seg_idx]
        seg_len = len(seg_frm_cls_dets)
        seg_frames = sorted_frames[past_frm_num: past_frm_num + seg_len]
        assert seg_len == len(seg_frames)

        seg_dets = []
        seg_det_infos = []
        for frm_idx, frm_dets in enumerate(seg_frm_cls_dets):
            for cls_idx, cls_dets in enumerate(frm_dets):
                seg_dets += cls_dets
                seg_det_infos += [{'frm_idx': frm_idx, 'cls_idx': cls_idx}] * len(cls_dets)
        seg_det_flags = [True] * len(seg_dets)
        seg_dets = np.array(seg_dets)
        seg_det_orders = np.argsort(seg_dets[:, 4])[::-1]
        seg_dets = seg_dets[seg_det_orders]
        seg_det_infos = np.array(seg_det_infos)
        seg_det_infos = seg_det_infos[seg_det_orders]

        seg_tck_cnt = 0
        seg_tck_cnt_org = len(seg_dets)
        seg_trajs = []

        for seg_det_idx in range(len(seg_dets)):
            det = seg_dets[seg_det_idx]
            det_info = seg_det_infos[seg_det_idx]
            det_flag = seg_det_flags[seg_det_idx]

            if not det_flag:
                continue

            seg_tck_cnt += 1
            print('Frame %s | Cate %s | Score %s' % (str(det_info['frm_idx']).rjust(3),
                                                     str(det_info['cls_idx']).rjust(2), '%.2f' % det[4]))
            # traj = _track_kcf(seg_frames, seg_frm_idx, det, vis=False)
            traj = _track_dlib(seg_frames, det_info['frm_idx'], det, vis=False)
            trk_cnt_org = sum(seg_det_flags)
            traj, traj_scr, seg_det_flags = _merge(traj, det_info['cls_idx'], seg_dets, seg_det_infos, seg_det_flags)
            trk_cnt_cur = sum(seg_det_flags)
            print(str(trk_cnt_org).rjust(5) + ' -> %d' % trk_cnt_cur)

            seg_trajs.append({
                'trj': traj,
                'scr': traj_scr,
                'cls': det_info['cls_idx']
            })

        all_trajs.append(seg_trajs)
        print('[%d/%d] %.2f / %.2f' % (seg_idx + 1, seg_num,
                                       (seg_tck_cnt * 1.0 / seg_len),
                                       (seg_tck_cnt_org * 1.0 / seg_len)))
        past_frm_num += seg_len / 2

    # resize
    if resize_ratio < 1.0:
        for seg_trajs in all_trajs:
            for traj in seg_trajs:
                trj = traj['trj']
                for box in trj:
                    if box is not None:
                        box[:4] = box[:4] / resize_ratio

    return all_trajs



