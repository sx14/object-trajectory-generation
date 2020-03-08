import os
from collections import defaultdict
from vis.show_frame import show_boxes, get_colors
from gen_traj.utils.io import *


def show_fgfa_detection(fid2dets, fid2fname, image_dir, categories, ext='JPEG'):

    for fid in sorted(fid2dets.keys()):
        print(fid)
        dets = fid2dets[fid]
        dets = sorted(dets, key=lambda det: det[5], reverse=True)
        dets = [det for det in dets if det[5] > 0.1][:10]
        boxes = [[det[0], det[1], det[2], det[3]] for det in dets]
        clses = ['%s[%.2f]' % (categories[det[4]], det[5]) for det in dets]
        img_path = os.path.join(image_dir, fid2fname[fid]+'.'+ext)
        show_boxes(img_path, boxes, clses, get_colors(len(boxes)))


if __name__ == '__main__':
    vidor_categories = ['__background__',  # always index 0
                        'bread', 'cake', 'dish', 'fruits',
                        'vegetables', 'backpack', 'camera', 'cellphone',
                        'handbag', 'laptop', 'suitcase', 'ball/sports_ball',
                        'bat', 'frisbee', 'racket', 'skateboard',
                        'ski', 'snowboard', 'surfboard', 'toy',
                        'baby_seat', 'bottle', 'chair', 'cup',
                        'electric_fan', 'faucet', 'microwave', 'oven',
                        'refrigerator', 'screen/monitor', 'sink', 'sofa',
                        'stool', 'table', 'toilet', 'guitar',
                        'piano', 'baby_walker', 'bench', 'stop_sign',
                        'traffic_light', 'aircraft', 'bicycle', 'bus/truck',
                        'car', 'motorcycle', 'scooter', 'train',
                        'watercraft', 'crab', 'bird', 'chicken',
                        'duck', 'penguin', 'fish', 'stingray',
                        'crocodile', 'snake', 'turtle', 'antelope',
                        'bear', 'camel', 'cat', 'cattle/cow',
                        'dog', 'elephant', 'hamster/rat', 'horse',
                        'kangaroo', 'leopard', 'lion', 'panda',
                        'pig', 'rabbit', 'sheep/goat', 'squirrel',
                        'tiger', 'adult', 'baby', 'child']

    res_dir = 'demo_video'
    data_root = os.path.join('../data', res_dir)
    output_root = os.path.join('../output', res_dir)
    video_root = '../data/VidOR/Data/VID'

    # load frame ids
    frame_info_path = os.path.join(data_root, 'frame_idx.txt')
    frame_names, frame_ids = load_vid_frame_idx(frame_info_path)
    fid2fname = {int(frame_ids[i]): frame_names[i] for i in range(len(frame_names))}

    out_txt_path = '../output/demo_video/fgfa_det_kcf_raw.txt'
    fid2dets = load_txt_result(out_txt_path)

    img_dir = '../data/VidOR/Data/VID'
    show_fgfa_detection(fid2dets, fid2fname, img_dir, vidor_categories)