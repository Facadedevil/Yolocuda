import asyncio
import cv2
import cupy as cp
from models.experimental import attempt_load
from utils.general import non_max_suppression

# Load the YOLOv5 model
weights_path = "path/to/weights.pt"
model = attempt_load(weights_path, map_location="cuda")
model.eval()

# Preprocess the input
def preprocess_input(image):
    img = cv2.resize(image, (640, 640))
    img = img.transpose(2, 0, 1)  # Transpose dimensions (HWC to CHW)
    img = img[np.newaxis, ...]  # Add batch dimension
    img = np.ascontiguousarray(img, dtype=np.float32)  # Ensure contiguous memory layout
    img /= 255.0  # Normalize pixel values
    img = cp.asarray(img)  # Move the data to the GPU memory (CuPy array)
    return img

# Run inference and display on GPU
async def process_frame(frame):
    # Perform inference on the frame
    img = preprocess_input(frame)
    img = cp.asarray(img)  # Transfer the frame to the GPU memory

    # Inference
    with cp.cuda.Device(0):  # Specify the GPU device for inference
        pred = model(img)

    # Transfer the predictions to the CPU memory if necessary
    pred_cpu = cp.asnumpy(pred)  # Uncomment this line if you want to transfer predictions to CPU memory

    # Post-processing
    pred = pred.astype(cp.float32)
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

    # Display the frame with bounding boxes (GPU-accelerated drawing)
    # ... (Add code here to draw bounding boxes on the frame)

    # Stream the processed frame (optional, if needed)
    # ... (Add code here to stream the processed frame)

async def process_video():
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Start asynchronous processing of the frame
        await process_frame(frame)

        # Display the frame (if desired)

    cap.release()

# Run the video processing asynchronously
asyncio.run(process_video())
