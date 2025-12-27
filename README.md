# SSD Object Detection with Qt

An application for object detection using SSD (Single Shot MultiBox Detector) models with OpenCV DNN module.

## Features

- **Multi-Object Detection**: Detect and visualize multiple objects in images
- **Bounding Box Visualization**: Color-coded rectangles around detected objects
- **Confidence Scores**: Shows detection confidence percentage for each object

## Usage

1. **Load Image**: Select an image file for detection
2. **Load Model**: 
   - First, select class names file (e.g., `coco.names`)
   - Then, select the SSD model file (.pb, .onnx, .caffemodel)
   - Optionally, select config file (for Caffe models)
3. **Detect Objects**: Run SSD object detection on the loaded image
4. **Reset Image**: Return to original image without detection markings


## Output Format
- **Detection Matrix**: [batchId, classId, confidence, x_min, y_min, x_max, y_max]

## Requirements

- OpenCV 4.x with DNN module
- Qt6
- CMake 3.12+
- C++17 compatible compiler
