#include "OpenCvDnnWidget.h"
#include <QtCore/QString>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMessageBox>
#include <fstream>
#include <opencv2/highgui.hpp>

OpenCvDNNWidget::OpenCvDNNWidget(QWidget *parent) : QMainWindow(parent) {
  setupUI();
  printOpenCVInfo();
}

QPixmap OpenCvDNNWidget::matToQPixmap(const cv::Mat &mat) {
  cv::Mat rgbMat;
  if (mat.channels() == 3) {
    cv::cvtColor(mat, rgbMat, cv::COLOR_BGR2RGB);
  } else if (mat.channels() == 1) {
    cv::cvtColor(mat, rgbMat, cv::COLOR_GRAY2RGB);
  } else {
    rgbMat = mat;
  }

  QImage qimg(rgbMat.data, rgbMat.cols, rgbMat.rows, rgbMat.step,
              QImage::Format_RGB888);
  return QPixmap::fromImage(qimg);
}

bool OpenCvDNNWidget::loadImage() {
  QString fileName = QFileDialog::getOpenFileName(
      this, tr("Open Image"), "",
      tr("Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"));

  if (fileName.isEmpty()) {
    return false;
  }
  currentImage = cv::imread(fileName.toStdString());
  if (!currentImage.empty()) {
    // Resize image for display if too large
    cv::Mat displayImage = currentImage.clone();
    if (displayImage.cols > 600 || displayImage.rows > 400) {
      cv::resize(displayImage, displayImage, cv::Size(600, 400));
    }

    imageLabel->setPixmap(matToQPixmap(displayImage));
    logTextEdit->append(QString("Image loaded: %1").arg(fileName));
    classifyButton->setEnabled(!dnnNet.empty());
    resetImageButton->setEnabled(true);
  } else {
    logTextEdit->append("Failed to load image!");
    return false;
  }

  return true;
}

bool OpenCvDNNWidget::loadClassNames() {

  QMessageBox::information(
      this, "Info", "Please select class names file (e.g., coco.names) ");

  classNames.clear();

  QString classNamePath =
      QFileDialog::getOpenFileName(this, tr("Open Class Names File (Optional)"),
                                   "", tr("Text Files (*.txt)"));
  if (classNamePath.isEmpty()) {
    return false;
  }
  classNames.clear();
  std::ifstream ifs(classNamePath.toStdString());
  if (ifs.is_open()) {
    std::string line;
    while (std::getline(ifs, line)) {
      classNames.push_back(line);
    }
    logTextEdit->append(QString("Loaded %1 class names from %2")
                            .arg(classNames.size())
                            .arg(classNamePath));
  } else {
    logTextEdit->append(
        QString("Failed to open class names file: %1").arg(classNamePath));
    return false;
  }
  return true;
}

bool OpenCvDNNWidget::loadModel() {

  if (!loadClassNames()) {
    logTextEdit->append("Class names can't be loaded.");
    return false;
  }

  QMessageBox::information(this, "Info",
                           "Please select model file (e.g., model.pb) ");
  QString modelPath = QFileDialog::getOpenFileName(
      this, tr("Open SSD Detection Model"), "",
      tr("Model Files (*.pb *.onnx *.caffemodel *.t7)"));

  if (!modelPath.isEmpty()) {
    QMessageBox::information(
        this, "Info", "Please select configuration file (e.g., config.pbtxt) ");
    QString configPath = QFileDialog::getOpenFileName(
        this, tr("Open Config File (Optional for SSD)"), "",
        tr("Config Files (*.prototxt *.pbtxt)"));

    try {
      if (configPath.isEmpty()) {
        // For models that don't require config (ONNX, some TensorFlow models)
        dnnNet = cv::dnn::readNet(modelPath.toStdString());
      } else {
        // For YOLO/Darknet models that need config files
        dnnNet =
            cv::dnn::readNet(modelPath.toStdString(), configPath.toStdString());
      }

      if (!dnnNet.empty()) {
        logTextEdit->append(
            QString("SSD detection model loaded successfully: %1")
                .arg(modelPath));
        if (!configPath.isEmpty()) {
          logTextEdit->append(QString("Config file: %1").arg(configPath));
        }
        classifyButton->setEnabled(!currentImage.empty());
      } else {
        logTextEdit->append("Failed to load SSD detection model!");
        return false;
      }
    } catch (const cv::Exception &e) {
      logTextEdit->append(QString("Model loading error: %1").arg(e.what()));
      return false;
    }
  }
  return true;
}

void OpenCvDNNWidget::detectObjects() {
  if (currentImage.empty() || dnnNet.empty()) {
    logTextEdit->append("Please load both image and model first!");
    return;
  }

  try {
    lastDetections.clear();

    // Create blob for SSD models (typically 300x300 input)
    cv::Mat blob =
        cv::dnn::blobFromImage(currentImage, .1, cv::Size(300, 300),
                               cv::Scalar(127.5, 127.5, 127.5), true, false);

    // Set input to the network
    dnnNet.setInput(blob);

    // Forward pass - SSD models typically have one output layer
    cv::Mat output = dnnNet.forward();

    // Parse SSD detections - output format: [1, 1, N, 7]
    // Each detection: [batchId, classId, confidence, x_min, y_min, x_max,
    // y_max]
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Get image dimensions for coordinate conversion
    int imgWidth = currentImage.cols;
    int imgHeight = currentImage.rows;

    // Process SSD output format
    cv::Mat detectionMat(output.size[2], output.size[3], CV_32F,
                         output.ptr<float>());

    for (int i = 0; i < detectionMat.rows; ++i) {
      int classId = static_cast<int>(detectionMat.at<float>(i, 1));
      float confidence = detectionMat.at<float>(i, 2);

      if (confidence > confidenceThreshold) {

        // Extract normalized coordinates and convert to absolute
        float x_min = detectionMat.at<float>(i, 3) * imgWidth;
        float y_min = detectionMat.at<float>(i, 4) * imgHeight;
        float x_max = detectionMat.at<float>(i, 5) * imgWidth;
        float y_max = detectionMat.at<float>(i, 6) * imgHeight;

        // Create bounding box
        cv::Rect box(static_cast<int>(x_min), static_cast<int>(y_min),
                     static_cast<int>(x_max - x_min),
                     static_cast<int>(y_max - y_min));

        // Clamp to image boundaries
        box.x = std::max(0, box.x);
        box.y = std::max(0, box.y);
        box.width = std::min(box.width, imgWidth - box.x);
        box.height = std::min(box.height, imgHeight - box.y);

        // Only add valid detections
        if (box.width > 0 && box.height > 0) {
          classIds.push_back(classId);
          confidences.push_back(confidence);
          boxes.push_back(box);
        }
      }
    }

    // Apply Non-Maximum Suppression (optional for SSD, but helpful for
    // overlapping detections)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold,
                      indices);

    // Create a copy of the original image for drawing
    cv::Mat displayImageWithResults = currentImage.clone();

    // Generate different colors for different classes
    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),   // Red
        cv::Scalar(0, 0, 255),   // Blue
        cv::Scalar(0, 255, 255), // Yellow
        cv::Scalar(128, 0, 128), // Purple
        cv::Scalar(0, 128, 128), // Teal
    };

    QString resultText = QString("SSD Detection Results:\nFound %1 objects:\n")
                             .arg(indices.size());

    // Draw bounding boxes and labels
    for (size_t i = 0; i < indices.size(); ++i) {
      int idx = indices[i];
      cv::Rect box = boxes[idx];
      int classId = classIds[idx];
      float confidence = confidences[idx];

      // Store detection
      Detection detection;
      detection.classId = classId;
      detection.confidence = confidence;
      detection.box = box;
      detection.className =
          (!classNames.empty() && classId < static_cast<int>(classNames.size()))
              ? classNames[classId - 1]
              : ("Class " + std::to_string(classId));
      lastDetections.push_back(detection);

      // Choose color based on class ID
      cv::Scalar color = colors[classId % colors.size()];

      // Draw bounding box
      cv::rectangle(displayImageWithResults, box, color, 2);

      // Prepare label text
      std::string labelText = detection.className + ": " +
                              std::to_string(confidence * 100).substr(0, 4) +
                              "%";

      // Calculate text size and draw background
      int fontFace = cv::FONT_HERSHEY_SIMPLEX;
      double fontScale = 0.6;
      int thickness = 1;
      cv::Size textSize =
          cv::getTextSize(labelText, fontFace, fontScale, thickness, nullptr);

      cv::Point labelPos(box.x, box.y - 5);
      if (labelPos.y < textSize.height) {
        labelPos.y = box.y + box.height + textSize.height + 5;
      }

      // Draw text background
      cv::Rect textBackground(labelPos.x, labelPos.y - textSize.height,
                              textSize.width, textSize.height + 5);
      cv::rectangle(displayImageWithResults, textBackground, color, cv::FILLED);

      // Draw text
      cv::putText(displayImageWithResults, labelText, labelPos, fontFace,
                  fontScale, cv::Scalar(0, 0, 0), thickness);

      // Add to result text
      resultText += QString("- %1: %2%\n")
                        .arg(QString::fromStdString(detection.className))
                        .arg(QString::number(confidence * 100, 'f', 1));
    }

    if (indices.empty()) {
      resultText =
          "SSD Detection Results:\nNo objects detected above threshold";
      logTextEdit->append("No objects detected above confidence threshold.");
    } else {
      logTextEdit->append(
          QString("SSD detected %1 objects successfully!").arg(indices.size()));
    }

    // Resize for display if too large
    cv::Mat resizedDisplay = displayImageWithResults.clone();
    if (resizedDisplay.cols > 600 || resizedDisplay.rows > 400) {
      cv::resize(resizedDisplay, resizedDisplay, cv::Size(600, 400));
    }

    // Update the image display with the detection results
    imageLabel->setPixmap(matToQPixmap(resizedDisplay));

    // Update result label
    resultLabel->setText(resultText);

  } catch (const cv::Exception &e) {
    logTextEdit->append(
        QString("SSD object detection error: %1").arg(e.what()));
  }
}

void OpenCvDNNWidget::setupUI() {
  setWindowTitle("SSD Object Detection");
  setMinimumSize(800, 600);

  QWidget *centralWidget = new QWidget(this);
  setCentralWidget(centralWidget);

  QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);

  // Left panel for image display
  QVBoxLayout *leftLayout = new QVBoxLayout();

  imageLabel = new QLabel();
  imageLabel->setMinimumSize(400, 300);
  imageLabel->setStyleSheet(
      "border: 2px solid gray; background-color: lightgray;");
  imageLabel->setAlignment(Qt::AlignCenter);
  imageLabel->setText("No Image Loaded");
  leftLayout->addWidget(imageLabel);

  // Buttons
  QHBoxLayout *buttonLayout = new QHBoxLayout();
  loadImageButton = new QPushButton("Load Image");
  loadModelButton = new QPushButton("Load Model");
  classifyButton = new QPushButton("Detect Objects");
  resetImageButton = new QPushButton("Reset Image");
  classifyButton->setEnabled(false);
  resetImageButton->setEnabled(false);

  connect(loadImageButton, &QPushButton::clicked, this,
          &OpenCvDNNWidget::loadImage);
  connect(loadModelButton, &QPushButton::clicked, this,
          &OpenCvDNNWidget::loadModel);
  connect(classifyButton, &QPushButton::clicked, this,
          &OpenCvDNNWidget::detectObjects);
  connect(resetImageButton, &QPushButton::clicked, this,
          &OpenCvDNNWidget::resetImage);

  buttonLayout->addWidget(loadImageButton);
  buttonLayout->addWidget(loadModelButton);
  buttonLayout->addWidget(classifyButton);
  buttonLayout->addWidget(resetImageButton);
  leftLayout->addLayout(buttonLayout);

  mainLayout->addLayout(leftLayout);

  // Right panel for results and logs
  QVBoxLayout *rightLayout = new QVBoxLayout();

  resultLabel = new QLabel("Detection Results:\nNo detections yet");
  //   resultLabel->setStyleSheet(
  //       "border: 1px solid gray; padding: 10px; background-color: white;");
  resultLabel->setAlignment(Qt::AlignTop);
  resultLabel->setMinimumHeight(150);
  rightLayout->addWidget(resultLabel);

  logTextEdit = new QTextEdit();
  logTextEdit->setMaximumHeight(200);
  rightLayout->addWidget(logTextEdit);

  mainLayout->addLayout(rightLayout);
}

void OpenCvDNNWidget::printOpenCVInfo() {
  logTextEdit->append(QString("OpenCV Version: %1").arg(CV_VERSION));
  logTextEdit->append("OpenCV DNN module loaded successfully!");
  logTextEdit->append("Ready to load images and SSD detection models.");
  logTextEdit->append("Supported formats: SSD MobileNet (.pb), SSD Caffe "
                      "(.caffemodel + .prototxt), ONNX (.onnx)");
}

void OpenCvDNNWidget::resetImage() {
  if (currentImage.empty()) {
    logTextEdit->append("No image to reset!");
    return;
  }

  // Reset to original image without any markings
  cv::Mat displayImage = currentImage.clone();
  if (displayImage.cols > 600 || displayImage.rows > 400) {
    cv::resize(displayImage, displayImage, cv::Size(600, 400));
  }

  imageLabel->setPixmap(matToQPixmap(displayImage));
  logTextEdit->append("Image reset to original state.");
}