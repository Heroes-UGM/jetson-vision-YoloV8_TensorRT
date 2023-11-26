from models import TRTModule  # isort:skip
from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, calculate_letterbox_offset
from jetson_utils import videoSource, videoOutput, cudaAllocMapped, cudaDeviceSynchronize, cudaOverlay, cudaResize, cudaDrawRect, cudaFont
import torch
#import time

engine_file = "last.engine"
# make sure to match hardware
widthcam = 640
heightcam = 480
fpscam = 30
# "display://0" for opengl display, "webrtc://@:8554/output" for web stream, "rtp://<remote-ip>:8554" for vlc stream
outputDisplay = "webrtc://@:8554/output"

device = torch.device("cuda:0")
Engine = TRTModule(engine_file, device)
H, W = Engine.inp_info[0].shape[-2:]

# set desired output names order
Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
#prev_time = 0
#new_time = 0

# Initiate source and display
input = videoSource("/dev/video0", options={'width': widthcam, 'height': heightcam, 'framerate': fpscam})
output = videoOutput(outputDisplay, ["--output-encoder=v4l2","--headless"])
offset = calculate_letterbox_offset(input.GetWidth(),input.GetHeight(),W,H)
font = cudaFont(size=16)

def lettering(im, scaling_factor, delta_w, delta_h, w, h):
    # Create cuda for resized image
    tensorresized = cudaAllocMapped(width=im.width * scaling_factor,
                                    height=im.height * scaling_factor,
                                    format=im.format)
    cudaResize(im, tensorresized)
    # Create cuda for background
    imgOutput = cudaAllocMapped(width=w, height=h,format='rgb8')
    cudaOverlay(tensorresized, imgOutput, delta_w, delta_h)
    return imgOutput

while True:
    # Read a frame from the video
    cam = input.Capture(format='rgb8')
    if cam is not None:
        
        # Matching res and convert to tensor
        img = lettering(cam,*offset)
        cudaDeviceSynchronize()
        tensor = blob(img)
        tensor = torch.as_tensor(tensor, device=device)

        # Inference
        data = Engine(tensor)
        bboxes, scores, labels = det_postprocess(data)
        
        # Drawing box and calculate distance
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            distance = 5744 * pow((bbox[2]-bbox[0])*(bbox[3] - bbox[1]), -0.5)
            cudaDrawRect(img, (bbox[0],bbox[1],bbox[2],bbox[3]), (color[0],color[1],color[2],125))
            font.OverlayText(img, img.width, img.height, f'{int(distance)}', bbox[0], bbox[1], font.White, font.Gray40)
            font.OverlayText(img, img.width, img.height, f'{int(score*100)} %', bbox[2], bbox[3], font.White, font.Gray40)
        output.Render(img)
        # Calculate FPS
        #new_time = time.time()
        #fps = int(1/(new_time-prev_time))
        #prev_time = new_time
        #print("FPS: "+str(fps))
