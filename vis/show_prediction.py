import os
import json
import random
import matplotlib.pyplot as plt


def show_boxes(im, dets, cls, confs, mode='single'):
    """Draw detected bounding boxes."""

    def random_color():
        color = []
        for i in range(3):
            color.append(random.randint(0, 255) / 255.0)
        return color

    if mode != 'single':
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

    for i in range(0, len(dets)):
        if mode == 'single':
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(im, aspect='equal')

        bbox = dets[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=random_color(), linewidth=5)
        )
        ax.text(bbox[0], bbox[1],
                '%s: %.2f' % (cls[i], confs[i]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

        if mode == 'single':
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    if mode != 'single':
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def show_trajectory(frame_paths, traj, colors):
    import matplotlib.pyplot as plt

    plt.figure(0)
    for i, frame_path in enumerate(frame_paths):
        plt.ion()
        plt.axis('off')

        im = plt.imread(frame_path)
        plt.imshow(im)

        bbox = traj[i]
        color = colors[i]

        if bbox is not None:
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
        plt.show()
        plt.pause(0.0000005)
        plt.cla()
    plt.close()


def show_prediction(video_root, pred_path, vid=None):

    with open(pred_path) as f:
        pred_res = json.load(f)
        vid_res = pred_res['results']

    if vid is not None:
        vid_res = {vid: vid_res[vid]}

    for vid in vid_res:

        frame_dir = os.path.join(video_root, vid)
        frame_list = sorted(os.listdir(frame_dir))
        frame_num = len(frame_list)
        print('>>>> %s [%d] <<<<' % (vid, frame_num))

        video_dets = vid_res[vid]
        video_dets = sorted(video_dets, key=lambda item: item['score'], reverse=True)
        print('%d dets' % len(video_dets))
        for tid, det in enumerate(video_dets):
            cls = det['category']

            traj = det['trajectory']
            score = det['score']
            org_stt_fid = det['org_start_fid']
            org_end_fid = det['org_end_fid']
            stt_fid = det['start_fid']
            end_fid = det['end_fid']

            print('T[%d] %s %.4f [%d| %d -> %d |%d]' % (tid, cls, score, stt_fid, org_stt_fid, org_end_fid, end_fid))

            blank_len = 30
            traj_boxes = [None] * len(frame_list)
            traj_colors = [None] * len(frame_list)

            org_color = (0.0, 1.0, 0.0)
            ext_color = (1.0, 0.0, 0.0)
            for fid in sorted(traj.keys()):
                fid_ind = int(fid)
                traj_boxes[fid_ind] = traj[fid]
                if fid_ind < org_stt_fid or fid_ind > org_end_fid:
                    traj_colors[fid_ind] = ext_color
                else:
                    traj_colors[fid_ind] = org_color

            seg_frames = frame_list[max(0, stt_fid-blank_len):min(end_fid+blank_len, frame_num)]
            traj_boxes = traj_boxes[max(0, stt_fid-blank_len):min(end_fid+blank_len, frame_num)]
            traj_colors = traj_colors[max(0, stt_fid-blank_len):min(end_fid+blank_len, frame_num)]
            seg_frame_paths = [os.path.join(frame_dir, frame_id) for frame_id in seg_frames]
            show_trajectory(seg_frame_paths, traj_boxes, traj_colors)


if __name__ == '__main__':
    video_root = '../data/VidOR/Data/VID/val'
    res_path = '../output/demo_video/fgfa_det_kcf.json'
    show_prediction(video_root, res_path)