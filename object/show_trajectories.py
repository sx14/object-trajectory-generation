import os
import time
import json
from tqdm import tqdm
import shutil
import random
from pynput import keyboard
import numpy as np


def good_colors():
    colors = [
        [255, 255, 255],
        [255, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ]
    colors = np.array(colors)
    colors = colors / 255.0
    return colors.tolist()


def random_color():
    color = []
    for i in range(3):
        color.append(random.randint(0, 255) / 255.0)
    return color


def show_trajectories(frame_dir, frame_dets, tid2color, save=False):
    import matplotlib.pyplot as plt

    show = []
    def stop(key):
        if key != keyboard.Key.space:
            show.append(0)
        return False

    save_dir = 'temp'
    if save:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            os.mkdir(save_dir)
        else:
            os.mkdir(save_dir)

    fids = sorted([int(fid) for fid in frame_dets if fid != 'viou'])

    plt.figure(0)
    tqdm_bar = tqdm(total=len(fids))
    for i, fid in enumerate(fids):

        if i % 100 == 0 and i > 0:
            with keyboard.Listener(on_press=stop) as listener:
                listener.join()
                listener.stop()

            if len(show) > 0:
                break

        tqdm_bar.update(1)
        str_fid = '%06d' % fid
        plt.ion()
        plt.axis('off')

        im = plt.imread(os.path.join(frame_dir, str_fid + '.JPEG'))
        plt.imshow(im)

        boxes = frame_dets[str_fid]

        for tid in boxes:
            bbox = boxes[tid]
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=tid2color[tid], linewidth=3.5)
            plt.gca().add_patch(rect)
        plt.show()
        plt.pause(0.0000005)

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        if save:
            plt.savefig(os.path.join('temp', '%06d.JPEG' % fid), bbox_inches='tight')
        plt.cla()
        plt.cla()
    plt.close()
    tqdm_bar.close()


def show_prediction(video_root, pred_path, gt_path, vid=None):

    with open(pred_path) as f:
        pred_res = json.load(f)
        vid_res = pred_res['results']

    with open(gt_path) as f:
        gt = json.load(f)

    if vid is not None:
        vid_res = {vid: vid_res[vid]}

    for vid in vid_res:

        frame_dir = os.path.join(video_root, vid)
        frame_list = sorted(os.listdir(frame_dir))
        frame_num = len(frame_list)
        print('>>>> %s [%d] <<<<' % (vid, frame_num))

        video_gts = gt[vid]
        video_dets = vid_res[vid]
        video_dets = sorted(video_dets, key=lambda item: item['viou'], reverse=True)

        for gt in video_gts:
            gt_fids = [int(fid) for fid in gt['trajectory'].keys()]
            gt_stt_fid = min(gt_fids)
            gt_end_fid = max(gt_fids)
            for det in video_dets:
                if det['hit_tid'] == gt['tid']:
                    det_stt_fid = det['start_fid']
                    det_end_fid = det['end_fid']
                    print('%s\t[%d|%d -> %d|%d]  [vIoU: %.4f][score: %.4f]'
                          % (det['category'],
                             gt_stt_fid, det_stt_fid,
                             det_end_fid, gt_end_fid,
                             det['viou'], det['score']))

                    time.sleep(2)
                    nTrajShown = 1
                    frame_dets = {}
                    for fid, box in gt['trajectory'].items():
                        if fid in frame_dets:
                            frame_dets[fid][0] = box
                        else:
                            frame_dets[fid] = {0: box}

                    nTrajShown += 1
                    for fid, box in det['trajectory'].items():
                        if fid in frame_dets:
                            frame_dets[fid][1] = box
                        else:
                            frame_dets[fid] = {1: box}

                    tid2colors = {}
                    colors = good_colors()
                    for i in range(nTrajShown):
                        if i < len(colors):
                            tid2colors[i] = colors[i]
                        else:
                            tid2colors[i] = random_color()
                    show_trajectories(frame_dir, frame_dets, tid2colors)





if __name__ == '__main__':
    video_root = '../data/VidOR/Data/VID/val'
    res_path = '../output/demo_video_trajs.json'
    gt_path = '../data/demo_video/vidor_val_object_gt.json'
    vid = u'0004/11566980553'
    show_prediction(video_root, res_path, gt_path, vid)
