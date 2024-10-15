import torch

def bbox_center_dim_to_topleft_botright(bb):
    """
    bb: <*, 4>
    last dimension is [x_center, y_center, w, h]
    outputs <*, 4> with last dimension [x1,y1,x2,y2]
    """
    bb_out = bb.clone()
    bb_out[..., 0] = bb[..., 0] - bb[..., 2] / 2
    bb_out[..., 1] = bb[..., 1] - bb[..., 3] / 2
    bb_out[..., 2] = bb[..., 0] + bb[..., 2] / 2
    bb_out[..., 3] = bb[..., 1] + bb[..., 3] / 2
    return bb_out


def iou(bb1, bb2):
    """
    bb1,bb2: <*,4>
    last dimension is [x1,y1,x2,y2]
    """
    assert bb1.shape == bb2.shape
    assert (bb1[..., 0] <= bb1[..., 2]).all()
    assert (bb1[..., 1] <= bb1[..., 3]).all()
    assert (bb2[..., 0] <= bb2[..., 2]).all()
    assert (bb2[..., 1] <= bb2[..., 3]).all()

    # compute areas as (x2-x1)*(y2-y1)
    area_bb1 = (bb1[..., 2] - bb1[..., 0]) * (bb1[..., 3] - bb1[..., 1])
    area_bb2 = (bb2[..., 2] - bb2[..., 0]) * (bb2[..., 3] - bb2[..., 1])

    # coordinates of the intersection rectangles
    int_x1 = torch.max(bb1[..., 0], bb2[..., 0])
    int_y1 = torch.max(bb1[..., 1], bb2[..., 1])
    int_x2 = torch.min(bb1[..., 2], bb2[..., 2])
    int_y2 = torch.min(bb1[..., 3], bb2[..., 3])

    # intersection area
    area_int = (int_x2 - int_x1) * (int_y2 - int_y1)

    # compute iou
    iou = area_int / (area_bb1 + area_bb2 - area_int)

    # replace iou to 0 when no intersection
    iou[(int_x2 < int_x1) + (int_y2 < int_y1)] = 0

    return iou

# Non-Max Suppression algorithm
def nms(bb, iou_threshold=0.5):
    # bb = <x_center,y_center,w,h,cf,*>
    original_bb = bb
    bb = bb.clone()

    # sort by cf
    idx = bb[:, 4].sort()[1]
    bb = bb[idx]

    bb[:, :4] = bbox_center_dim_to_topleft_botright(bb[:, :4])
    keep = []
    while bb.shape[0] > 1:
        bb, top = bb[:-1], bb[-1]
        top_ious = iou(top.unsqueeze(0).expand(bb.shape[0], -1), bb)
        mask = top_ious < iou_threshold
        bb = bb[mask]

        keep.append(idx[-1])
        idx = idx[:-1]
        idx = idx[mask]
    if bb.shape[0] == 1:
        keep.append(idx[0])

    idx = torch.stack(keep, dim=0)
    bb = original_bb[idx]
    return bb