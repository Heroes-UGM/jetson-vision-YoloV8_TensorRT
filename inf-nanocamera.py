from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch
import time
#import imutils
import nanocamera as nano

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox


device = torch.device("cuda:0")
Engine = TRTModule("last_300ep.engine", device)
H, W = Engine.inp_info[0].shape[-2:]

# set desired output names order
Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

camera = nano.Camera(camera_type=1, device_id=0)
prev_time = 0
new_time = 0

while camera.isReady():
    # Read a frame from the video
    frame = camera.read()
    if True:
        bgr, ratio, dwdh = letterbox(frame, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        # inference
        data = Engine(tensor)
        bboxes, scores, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(frame, bbox[:2], bbox[2:], color, 2)
            cv2.putText(frame,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
        new_time = time.time()
        fps = int(1/(new_time-prev_time))
        prev_time = new_time
        print("FPS: "+str(fps))
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
camera.release()
del camera
