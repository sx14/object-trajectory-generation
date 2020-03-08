import json
from collections import defaultdict
import numpy as np

from vis.show_trajactory import show_trajectories


def show_alphapose(res_path, image_dir):
    with open(res_path) as f:
        res = json.load(f)

    trajs = defaultdict(dict)
    for file_name, insts in res.items():
        fid = '%06d' % int(file_name.split('.')[0])

        for inst in insts:
            tid = inst['idx']
            scr = inst['scores']

            kps = inst['keypoints']
            kps = np.array(kps)
            kps = kps.reshape((-1, 3))
            box = [9999, 9999, 0, 0]
            for kp in kps:
                x, y, s = kp
                x = int(x)
                y = int(y)
                box[0] = min(box[0], x)
                box[1] = min(box[1], y)
                box[2] = max(box[2], x)
                box[3] = max(box[3], y)

            box_height = box[3] - box[1]
            box_width = box[2] - box[0]
            box_h_padding = int(box_height * 0.1)
            box_w_padding = int(box_width * 0.1)

            box[0] = max(0, box[0]-box_w_padding)
            box[1] = max(0, box[1]- 2 * box_h_padding)
            box[2] = box[2] + box_w_padding
            box[3] = box[3] + box_w_padding

            trajs[tid][fid] = [box, scr, 'h']

    long_trajs = []
    for traj in trajs.values():
        score_sum = sum([det[1] for det in traj.values()])
        score_avg = score_sum / len(traj.keys())
        if score_avg > 2 and len(traj) > 60:
            long_trajs.append(traj)

    show_trajectories(image_dir, long_trajs)


def show_lighttrack(res_path, image_dir):
    with open(res_path) as f:
        res = json.load(f)

    trajs = defaultdict(dict)

    for img_dets in res:
        fid = '%06d' % int(img_dets['image']['name'][5:10])

        for inst in img_dets['candidates']:
            if 'track_id' not in inst:
                continue
            tid = inst['track_id']
            scr = inst['track_score']
            box = inst['det_bbox']
            box = [int(max(box[0], 0)), int(max(box[1], 0)),
                   int(max(box[0], 0))+int(box[2]),
                   int(max(box[1], 0))+int(box[3])]

            trajs[tid][fid] = [box, scr, 'h']

    long_trajs = [traj for traj in trajs.values()]
    show_trajectories(image_dir, long_trajs)


if __name__ == '__main__':
    img_dir = '../data/demo_video/frames'
    res_path = '../data/demo_video/alphapose-results.json'
    show_alphapose(res_path, img_dir)

    #res_path = '../data/lighttrack-results.json'
    #show_lighttrack(res_path, img_dir)