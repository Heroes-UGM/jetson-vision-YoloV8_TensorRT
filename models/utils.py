from typing import List, Tuple, Union

import numpy as np
from numpy import ndarray

# image suffixs
SUFFIXS = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff',
           '.webp', '.pfm')


def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)


def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    im = np.transpose(im,[2,0,1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    return im
        
def calculate_letterbox_offset(original_width, original_height, target_width, target_height):
    # Calculate aspect ratios
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # Check if letterboxing or pillarboxing is needed
    if original_aspect_ratio > target_aspect_ratio:
        # Letterboxing
        scaled_width = target_width
        scaling_factor = target_width / original_width
        scaled_height = original_height * scaling_factor
        delta_w = 0
        delta_h = (target_height - scaled_height) / 2
    else:
        # Pillarboxing
        scaled_height = target_height
        scaling_factor = target_height / original_height
        scaled_width = original_width * scaling_factor
        delta_w = (target_width - scaled_width) / 2
        delta_h = 0

    return (scaling_factor, delta_w, delta_h, target_width, target_height)

