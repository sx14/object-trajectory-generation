

def IoU(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1[:4]
    xmin2, ymin2, xmax2, ymax2 = box2[:4]

    # ibox
    xmini = max(xmin1, xmin2)
    xmaxi = min(xmax1, xmax2)
    ymini = max(ymin1, ymin2)
    ymaxi = min(ymax1, ymax2)

    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    areai = max(0, (xmaxi - xmini + 1)) * max(0, (ymaxi - ymini + 1))

    return areai * 1.0 / (area1 + area2 - areai)


def vIoU(traj1, traj2, box_thresh=0.5):
    assert len(traj1) == len(traj2)

    union_count = 0
    inter_count = 0

    iou_sum = 0

    for i in range(len(traj1)):
        box1 = traj1[i]
        box2 = traj2[i]

        if box1 is not None or box2 is not None:
            union_count += 1
        if box1 is not None and box2 is not None:
            iou = IoU(box1, box2)
            iou_sum += iou
            if iou >= box_thresh:
                inter_count += 1

    viou = inter_count * 1.0 / union_count
    return viou


def overlap_ratio(traj1, traj2):
    # align two trajectories, fill None
    traj1_len = len(traj1)
    traj2_len = len(traj2)
    traj1 = traj1 + [None] * (traj2_len - traj1_len / 2)
    traj2 = [None] * (traj1_len / 2) + traj2
    assert len(traj1) == len(traj2)

    int_len = 0
    hit_len = 0
    for fid in range(len(traj1)):
        box1 = traj1[fid]
        box2 = traj2[fid]
        if box1 is not None and box2 is not None:
            int_len += 1
            iou = IoU(box1, box2)

            hit_len += iou
            # if iou > 0.5:
            #     hit_len += 1
    if int_len > 0:
        return hit_len * 1.0 / int_len
    else:
        return 0