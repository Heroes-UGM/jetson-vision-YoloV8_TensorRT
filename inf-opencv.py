from models import TRTModule  # isort:skip

import cv2
import torch
import time
import imutils

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox

pipeline = " ! ".join(["v4l2src device=/dev/video0",
                       "video/x-raw, width=640, height=480, format=(string)YUY2, framerate=30/1",
                       "videoconvert",
                       "video/x-raw, format=BGR",
                       "appsink drop=1"
                       ])
                       
gst = "appsrc ! queue ! videoconvert ! video/x-raw,format=RGBA ! nvvidconv ! nvegltransform ! nveglglessink "

                       

device = torch.device("cuda:0")
Engine = TRTModule("best_416_sim.engine", device)
H, W = Engine.inp_info[0].shape[-2:]

# set desired output names order
Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
vw = cv2.VideoWriter(gst, cv2.CAP_GSTREAMER, 0, 30, (640, 480))
prev_time = 0
new_time = 0

while video_capture.isOpened():
    # Read a frame from the video
    ret_val, frame = video_capture.read()
    if ret_val:
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
        vw.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
video_capture.release()
cv2.destroyAllWindows()
