# SSD Object Detection with Qt

A Qt-based GUI application for real-time object detection using SSD (Single Shot MultiBox Detector) models with OpenCV DNN module.

## Features

- **SSD-Style Detection**: Optimized for Single Shot MultiBox Detector architecture
- **Multi-Object Detection**: Detect and visualize multiple objects in images
- **Bounding Box Visualization**: Color-coded rectangles around detected objects
- **Confidence Scores**: Shows detection confidence percentage for each object
- **Normalized Coordinates**: Proper handling of SSD's normalized output format
- **Interactive GUI**: Easy-to-use interface with file dialogs
- **Class Names**: Support for custom class label files

## Supported SSD Model Formats

- **TensorFlow**: `.pb` files (SSD MobileNet, SSD Inception, etc.)
- **ONNX**: `.onnx` files (cross-platform SSD models)
- **Caffe**: `.caffemodel` + `.prototxt` files (original SSD implementation)
- **PyTorch**: `.t7` files (Torch7 format)

## SSD Architecture Benefits

- **Single Forward Pass**: Faster than two-stage detectors
- **Multiple Scale Detection**: Detects objects at different scales
- **Direct Bounding Box Prediction**: No region proposals needed
- **Normalized Output**: Coordinates in [0,1] range for resolution independence

## Usage

1. **Load Image**: Select an image file for detection
2. **Load Model**: 
   - First, select class names file (e.g., `coco.names`)
   - Then, select the SSD model file (.pb, .onnx, .caffemodel)
   - Optionally, select config file (for Caffe models)
3. **Detect Objects**: Run SSD object detection on the loaded image
4. **Reset Image**: Return to original image without detection markings

## SSD Parameters

- **Input Size**: 300x300 (standard for SSD MobileNet)
- **Preprocessing**: Mean subtraction with ImageNet values (103.94, 116.78, 123.68)
- **Scale Factor**: 0.017 (1/255 * 4.4 for proper normalization)
- **Confidence Threshold**: 0.5 (adjustable in code)
- **NMS Threshold**: 0.4 (adjustable in code)

## Output Format

- **Detection Matrix**: [batchId, classId, confidence, x_min, y_min, x_max, y_max]
- **Normalized Coordinates**: All coordinates are in [0,1] range
- **Visual**: Color-coded bounding boxes with class labels and confidence percentages
- **Text**: Detailed detection results in the results panel
- **Log**: Process information and status messages

## Popular SSD Models

- **SSD MobileNet v1/v2**: Fast detection for mobile/embedded devices
- **SSD Inception v2**: Better accuracy with reasonable speed
- **SSD ResNet**: Higher accuracy for complex scenes
- **Custom SSD**: Fine-tuned models for specific use cases

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Running

```bash
./run.sh
# or
cd build && ./OpenCvDNN
```

## Requirements

- OpenCV 4.x with DNN module
- Qt6
- CMake 3.12+
- C++17 compatible compiler