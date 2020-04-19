import os
import json
from matplotlib import pyplot as plt


def show_boxes(im_path, dets, cls, colors=None):
    """Draw detected bounding boxes."""
    if colors is None:
        colors = ['red' for _ in range(len(dets))]
    im = plt.imread(im_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(0, len(dets)):

        bbox = dets[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=colors[i], linewidth=1.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{}'.format(cls[i]),
                bbox=dict(facecolor=colors[i], alpha=0.5),
                fontsize=14, color='white')
        plt.axis('off')
        plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ds = 'VidOR-HOID-mini'
    res_dir = 'vidor_hoid_mini'
    data_root = os.path.join('../data', res_dir)
    output_root = os.path.join('../output', res_dir)
    output_path = os.path.join(output_root, 'fgfa_det.json')
    video_root = '../data/%s/Data/VID' % ds
    res = json.load(open(output_path))
    for pid_vid in res:
        video_dir = os.path.join(video_root, 'val', pid_vid)
        for fid in res[pid_vid]:
            frame_path = os.path.join(video_dir, fid+'.JPEG')
            dets = res[pid_vid][fid]
            det_list = [[det['xmin'], det['ymin'], det['xmax'], det['ymax']] for det in dets]
            cls_list = [det['category'] for det in dets]
            show_boxes(frame_path, det_list, cls_list)