import asyncio
import cv2
import cupy as cp
# import numpy as np
import time
from models.experimental import attempt_load
from utils.general import non_max_suppression
from flask import Flask, Response, render_template

# Load the YOLOv5 model
weights_path = "/home/tcom/shubham/yolov5/yolov5s.pt"
model = attempt_load(weights_path, device="cuda")
model.eval()

# Preprocess the input
def preprocess_input(images):
    processed_images = []
    for image in images:
        # cv2.rectangle()
        # cv2.cuda.rectangle()

        img = cv2.cuda.resize(image, (640, 640))
        img = img.transpose(2, 0, 1)  # Transpose dimensions (HWC to CHW)
        img = img[cp.newaxis, ...]  # Add batch dimension
        img = cp.ascontiguousarray(img, dtype=cp.float32)  # Ensure contiguous memory layout
        img /= 255.0  # Normalize pixel values
        img = cp.asarray(img)  # Move the data to the GPU memory (CuPy array)
        processed_images.append(img)
    return processed_images

# Run inference and display on GPU
async def process_frames(frames):
    # Perform inference on the batch of frames
    imgs = preprocess_input(frames)
    imgs = cp.stack(imgs)  # Stack the frames along the batch dimension

    # Inference
    with cp.cuda.Device(0):  # Specify the GPU device for inference
        preds = model(imgs)

    # Post-processing
    preds = preds.astype(cp.float32)
    preds = non_max_suppression(preds, conf_thres=0.5, iou_thres=0.5)

    # Draw bounding boxes on the frames
    processed_frames = []
    for i, det in enumerate(preds):
        frame = frames[i]  # Use the original CPU frame
        for d in det:
            x1, y1, x2, y2, conf, cls = d.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        processed_frames.append(frame)

    # Convert the frames to JPEG format
    buffers = []
    for frame in processed_frames:
        _, buffer = cv2.imencode('.jpg', frame)
        buffers.append(buffer.tobytes())

    # Release GPU memory
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    return buffers

async def process_video():
    cap = cv2.VideoCapture(video_path)
    frame_buffer = []
    last_batch_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)

        # Check if the batch size is reached or a specific time interval has elapsed
        if len(frame_buffer) == batch_size or time.time() - last_batch_time >= time_interval:
            # Start asynchronous processing of the batch of frames
            image_buffers = await process_frames(frame_buffer)

            # Update the global variable with the latest processed frames
            global processed_frames
            processed_frames = image_buffers

            frame_buffer.clear()
            last_batch_time = time.time()

    cap.release()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        # Yield the latest processed frames as multipart JPEG data
        for frame in processed_frames:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    video_path = "rtsp://admin:Admin1357@192.168.0.190:554/profile2/media.smp"  # Replace with your video file path
    batch_size = 4  # Specify the desired batch size
    time_interval = 0.1  # Specify the desired time interval in seconds

    # Initialize the global variable for the processed frames
    processed_frames = []

    loop = asyncio.get_event_loop()
    loop.create_task(process_video())

    app.run()
