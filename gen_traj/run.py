import numpy as np
from utils.io import *
from gen_traj import *
from tracking_by_link import *
from tracking_by_vot import *
from merge_strategies import *

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

vidor_hoid_mini_categories = ["__background__",  # always index 0
                              "adult", "aircraft", "baby", "baby_seat",
                              "baby_walker", "backpack", "ball/sports_ball",
                              "bat", "bench", "bicycle", "bird", "bottle",
                              "cake", "camera", "car", "cat", "cellphone",
                              "chair", "child", "cup", "dish", "dog", "duck",
                              "fruits", "guitar", "handbag", "horse", "laptop",
                              "piano", "rabbit", "racket", "refrigerator",
                              "scooter", "screen/monitor", "skateboard", "ski",
                              "snowboard", "sofa", "stool", "surfboard",
                              "table", "toy", "watercraft"]

ds2cates = {'VidOR': vidor_categories, 'VidOR-HOID-mini': vidor_hoid_mini_categories}


if __name__ == '__main__':

    ds = 'VidOR'
    res_dir = 'demo_video'
    tracking_method = 'seqnms'
    data_root = os.path.join('../data', res_dir)
    output_root = os.path.join('../output', res_dir)
    video_root = '../data/%s/Data/VID' % ds

    # load FGFA results (individual detections)
    raw_det_path = os.path.join(data_root, 'fgfa_det_raw.bin')
    all_dets_raw, frame_idx = load_fgfa_raw_detection(raw_det_path)

    # load video info
    video_info_path = os.path.join(data_root, 'video_idx.txt')
    video_names, video_frame_nums = load_vid_frame_nums(video_info_path)

    # init result container
    all_traj_dets = []
    for c in range(len(all_dets_raw)):
        all_traj_dets.append([])

    # process
    all_video_end_frm_idx = np.cumsum(np.array(video_frame_nums))
    for v in range(len(video_frame_nums)):
        print('Processing %d-th video ...' % v)
        end_frm_idx = all_video_end_frm_idx[v]
        stt_frm_idx = end_frm_idx - video_frame_nums[v]

        vid_frame_root = os.path.join(video_root, video_names[v])
        vid_dets_raw = [all_dets_raw[c][stt_frm_idx:end_frm_idx] for c in range(len(all_dets_raw))]
        if tracking_method == 'seqnms':
            vid_dets_trj = track_in_video(vid_dets_raw, seqnms_track, greedy_merge, vid_frame_root)
        elif tracking_method == 'kcf':
            vid_dets_trj = track_in_video(vid_dets_raw, vot_track, greedy_merge, vid_frame_root)

        for c in range(len(all_dets_raw)):
            all_traj_dets[c] += vid_dets_trj[c]

    # output txt
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    out_txt_path = os.path.join(output_root, 'fgfa_det_%s.txt' % tracking_method)
    output_txt_result(all_traj_dets, frame_idx, out_txt_path)

    # output json
    imageset_path = os.path.join(data_root, 'frame_idx.txt')
    out_json_path = os.path.join(output_root, 'fgfa_det_%s.json' % tracking_method)
    output_json_result(imageset_path, [out_txt_path], out_json_path, ds2cates[ds], video_root)