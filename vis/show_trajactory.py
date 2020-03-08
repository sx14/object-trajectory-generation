import os
from collections import defaultdict
from show_frame import *


def show_trajectories(image_dir, trajectories, ext='JPEG'):
    """
    trajectories:
    [{img_id: [box, score, cls]}]
    """
    color_bank = get_colors(100)
    img_dets = defaultdict(dict)
    for tid, traj in enumerate(trajectories):
        for img_id in traj:
            img_dets[img_id][tid] = traj[img_id]

    plt.figure(0)
    for img_id in sorted(img_dets.keys()):
        dets = img_dets[img_id]
        img_path = os.path.join(image_dir, '%s.%s' % (img_id, ext))

        tids = dets.keys()
        bboxes = [dets[tid][0] for tid in tids]
        scores = ['%.2f' % dets[tid][1] for tid in tids]
        colors = [color_bank[tid] for tid in tids]

        plt.ion()
        plt.axis('off')
        show_boxes(img_path, bboxes, scores, colors)
        plt.pause(0.00001)
        plt.cla()
    plt.close()
