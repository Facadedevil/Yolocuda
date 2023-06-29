# yolocuda

This repository contains code for performing real-time object detection using YOLOv5. It uses a pre-trained YOLOv5 model to detect objects in video streams from an RTSP source.

## Requirements

- Python 3.x
- OpenCV
- PyTorch
- torchvision
- cupy

## Installation

1. Clone the repository:
git clone [https://github.com/Facadedevil/Yolocuda](https://github.com/Facadedevil/Yolocuda)




2. Install the required dependencies:
pip install opencv-python torch torchvision cupy

3. Download the YOLOv5 weights file:
wget [https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt)



## Usage

1. Run the script with the RTSP URL as a command-line argument:  
python optimized-yolo.py rtsp://your_rtsp_url



3. Customize the settings:

- Batch Size: Specify the desired batch size by modifying the `batch_size` variable.
- Time Interval: Specify the desired time interval in seconds by modifying the `time_interval` variable.
- Caching: Specify the maximum number of frames to keep in the buffer by modifying the `caching` variable.

## Output

The script will display the video stream with bounding boxes and class labels for detected objects in real-time.


# Repository Under Development

**Note: This repository is currently under development, and as a result, you may encounter bugs or unexpected behavior.**

We are actively working on improving and refining the codebase to provide a more stable and reliable experience. However, during this development phase, it is possible that you may encounter issues or errors while using the code.

We appreciate your understanding and patience as we work towards resolving any bugs and making the necessary improvements. If you encounter any issues, please feel free to report them to us, and we will do our best to address them promptly.

Thank you for your interest and support!


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.





