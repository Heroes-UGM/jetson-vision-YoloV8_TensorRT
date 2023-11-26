from typing import List, Tuple, Union

from torch import Tensor
from torchvision.ops import nms

def det_postprocess(data: Tuple[Tensor, Tensor, Tensor, Tensor]):
    assert len(data) == 4
    iou_thres: float = 0.6
    num_dets, bboxes, scores, labels = data[0][0], data[1][0], data[2][0], data[3][0]
    nums = num_dets.item()
    if nums == 0:
        return bboxes.new_zeros((0, 4)), scores.new_zeros((0, )), labels.new_zeros((0, ))
    # check score negative
    scoresid = scores < 0
    scores[scoresid] = 1 + scores[scoresid]
    # add nms
    idx = nms(bboxes, scores, iou_thres)
    bboxes, scores, labels = bboxes[idx], scores[idx], labels[idx]
    numsnms = scores > 0
    bboxes = bboxes[numsnms]
    scores = scores[numsnms]
    labels = labels[numsnms]

    return bboxes, scores, labels
