import json
from utils.io import *
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


def proc_video_dets(vid_dets_raw, conf_thr=0.4, max_per_frame=20):
    frm_num = len(vid_dets_raw[0])
    all_frm_dets = [[] for _ in range(frm_num)]

    for cate_idx in range(len(vid_dets_raw)):
        cate_dets_raw = vid_dets_raw[cate_idx]
        for frm_idx in range(len(cate_dets_raw)):
            cate_frm_dets_raw = np.array(cate_dets_raw[frm_idx])
            keep = nms(cate_frm_dets_raw, 0.3)
            cate_frm_dets = cate_frm_dets_raw[keep]
            for det in cate_frm_dets:
                det = det.tolist()
                det.append(cate_idx + 1)
                all_frm_dets[frm_idx].append(det)

    for frm_idx in range(frm_num):
        frm_dets = all_frm_dets[frm_idx]
        all_frm_dets[frm_idx] = [frm_det for frm_det in frm_dets if frm_det[4] > conf_thr][:max_per_frame]

    return all_frm_dets


def output_json(all_video_dets, cates, output_path):
    print('Generating json result file ...')
    res = defaultdict(dict)
    for video_name in all_video_dets:
        all_frm_dets = all_video_dets[video_name]
        for frm_idx in range(len(all_frm_dets)):
            frm_dets = all_frm_dets[frm_idx]
            res[video_name]['%06d' % frm_idx] = []

            for frm_det in frm_dets:
                xmin, ymin, xmax, ymax, score, cate_idx = frm_det
                det = {'xmin': int(xmin), 'ymin': int(ymin),
                       'xmax': int(xmax), 'ymax': int(ymax),
                       'score': score, 'category': cates[cate_idx]}
                res[video_name]['%06d' % frm_idx].append(det)

    with open(output_path, 'w') as f:
        json.dump(res, f)
    print('results saved at: %s' % output_path)


if __name__ == '__main__':

    ds = 'VidOR-HOID-mini'
    res_dir = 'vidor_hoid_mini'
    data_root = os.path.join('../data', res_dir)
    output_root = os.path.join('../output', res_dir)

    # load FGFA results (individual detections)
    raw_det_path = os.path.join(data_root, 'fgfa_det_raw.bin')
    all_dets_raw, frame_idx = load_fgfa_raw_detection(raw_det_path)

    # load video info
    video_info_path = os.path.join(data_root, 'video_idx.txt')
    video_names, video_frame_nums = load_vid_frame_nums(video_info_path)

    # process
    print('Processing ... (NMS, filter)')
    time.sleep(2)
    all_video_end_frm_idx = np.cumsum(np.array(video_frame_nums))
    all_video_dets = {}
    for v in tqdm(range(len(video_frame_nums))):
        video_name = video_names[v]
        end_frm_idx = all_video_end_frm_idx[v]
        stt_frm_idx = end_frm_idx - video_frame_nums[v]
        vid_dets_raw = [all_dets_raw[c][stt_frm_idx:end_frm_idx] for c in range(len(all_dets_raw))]
        vid_dets = proc_video_dets(vid_dets_raw, 0.4)
        all_video_dets[video_name[4:]] = vid_dets

    output_path = os.path.join(output_root, 'fgfa_det.json')
    output_json(all_video_dets, ds2cates[ds], output_path)
