from utils.io import *
from tracking_by_link import *
from tracking_by_vot import *
from tracking_by_match import *
from merge_strategies import *


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


def gen_traj_by_seg(all_dets_raw, seg_track_method, merge_method, frame_root, max_traj_num=50, num_seg_frm=30):
    """ Process one video """
    all_segs = _gen_segments(all_dets_raw, num_seg_frm)
    all_seg_trajs = seg_track_method(all_segs, frame_root)
    # trajs = [{'cls': cls, 'scr': scr, 'trj': [np.array([x1,x2,x3,x4,scr,-1])]}]

    num_cls = len(all_segs[0])
    num_frm = len(all_dets_raw[0])
    all_traj_dets = merge_method(all_seg_trajs, num_cls, num_frm, num_seg_frm, max_traj_num)
    return all_traj_dets


def gen_traj(all_dets_raw, max_traj_num=50):
    """ Process one video """
    return tracking_by_match(all_dets_raw, max_traj_num=max_traj_num)


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
    stt_time = time.time()
    ds = 'VidOR'
    res_dir = 'vidor_hoid_mini'
    tracking_method = 'vot'
    data_root = os.path.join('../data', res_dir)
    output_root = os.path.join('../output', res_dir)
    video_root = '../data/%s/Data/VID' % ds

    # load FGFA results (individual detections)
    raw_det_path = os.path.join(data_root, 'fgfa_det_raw.bin')
    all_dets_raw, frame_idx = load_fgfa_raw_detection(raw_det_path)

    # load video info
    video_info_path = os.path.join(data_root, 'video_idx.txt')
    video_names, video_frame_nums = load_vid_frame_nums(video_info_path)
    all_frame_sum = sum(video_frame_nums)
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
            vid_dets_trj = gen_traj_by_seg(vid_dets_raw, seqnms_track, greedy_merge, vid_frame_root)
        elif tracking_method == 'vot':
            vid_dets_trj = gen_traj_by_seg(vid_dets_raw, vot_track, greedy_merge, vid_frame_root)
        elif tracking_method == 'iou':
            vid_dets_trj = gen_traj(vid_dets_raw)

        for c in range(len(all_dets_raw)):
            all_traj_dets[c] += vid_dets_trj[c]

    end_time = time.time()

    # output speed
    time_consume = end_time - stt_time
    spf = time_consume / all_frame_sum
    out_speed_path = os.path.join(output_root, 'speed_of_%s.txt' % tracking_method)
    output_speed(spf, out_speed_path)

    # output txt
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    out_txt_path = os.path.join(output_root, 'fgfa_det_%s.txt' % tracking_method)
    output_txt_result(all_traj_dets, frame_idx, out_txt_path)

    # output json
    imageset_path = os.path.join(data_root, 'frame_idx.txt')
    out_json_path = os.path.join(output_root, 'fgfa_det_%s.json' % tracking_method)
    output_json_result(imageset_path, [out_txt_path], out_json_path, ds2cates[ds], video_root)