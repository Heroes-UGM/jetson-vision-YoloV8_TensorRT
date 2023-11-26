from jetson_utils import videoSource, videoOutput, cudaAllocMapped, cudaDeviceSynchronize, cudaOverlay, cudaResize, cudaDrawRect, cudaFont
from models.utils import blob, calculate_letterbox_offset

# make sure to match hardware
widthcam = 640
heightcam = 480
fpscam = 30

# "display://0" for opengl display, "webrtc://@:8554/output" for web stream, "rtp://<remote-ip>:8554" for vlc stream
outputDisplay = "data.mp4"
W = 416
H = 416

input = videoSource("/dev/video0", options={'width': widthcam, 'height': heightcam, 'framerate': fpscam})
output = videoOutput(outputDisplay, ["--headless"])
offset = calculate_letterbox_offset(input.GetWidth(),input.GetHeight(),W,H)

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
        img = lettering(cam,*offset)
        output.Render(img)
