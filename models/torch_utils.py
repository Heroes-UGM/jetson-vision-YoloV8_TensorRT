from typing import List, Tuple, Union

from torch import Tensor
from torchvision.ops import nms

def det_postprocess(data: Tuple[Tensor, Tensor, Tensor]):
    assert len(data) == 3
    bboxes, scores, labels = data[2][0], data[0][0], data[1][0]
    id_thresh = (scores) > 0.3
    bboxes = bboxes[id_thresh]
    labels = labels[id_thresh]
    scores = scores[id_thresh]
    if len(bboxes) == 0:
        return bboxes.new_zeros((0, 4)), scores.new_zeros(
            (0, )), labels.new_zeros((0, ))
    else:
        # add nms
        idx = nms(bboxes, scores, float(0.5))
        bboxes, scores, labels = bboxes[idx], scores[idx], labels[idx]
        return bboxes, scores, labels
