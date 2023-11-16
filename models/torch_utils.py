from typing import List, Tuple, Union

from torch import Tensor
from torchvision.ops import nms

def det_postprocess(data: Tuple[Tensor, Tensor, Tensor]):
    assert len(data) == 3
    nums, bboxes, scores, labels = len(data[2][0]), data[2][0], data[0][0], data[1][0]
    if nums == 0:
        return bboxes.new_zeros((0, 4)), scores.new_zeros(
            (0, )), labels.new_zeros((0, ))
    # add nms
    idx = nms(bboxes, scores, float(0.65))
    idx = idx[scores[idx] > 0.25]
    bboxes, scores, labels = bboxes[idx], scores[idx], labels[idx]

    return bboxes, scores, labels
