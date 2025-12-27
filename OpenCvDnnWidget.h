#ifndef OPENCVDNNWIDGET_H
#define OPENCVDNNWIDGET_H

#include <QtGui/QImage>
#include <QtGui/QPixmap>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class OpenCvDNNWidget : public QMainWindow {
    Q_OBJECT

private:
    QLabel *imageLabel;
    QLabel *resultLabel;
    QTextEdit *logTextEdit;
    QPushButton *loadImageButton;
    QPushButton *loadModelButton;
    QPushButton *classifyButton;
    QPushButton *resetImageButton;

    cv::Mat currentImage;
    cv::dnn::Net dnnNet;
    std::vector<std::string> classNames;
    
    // Object detection parameters
    float confidenceThreshold = 0.4f;
    float nmsThreshold = 0.4f;
    
    // Detection results
    struct Detection {
        int classId;
        float confidence;
        cv::Rect box;
        std::string className;
    };
    std::vector<Detection> lastDetections;

    // Convert OpenCV Mat to QPixmap for display
    QPixmap matToQPixmap(const cv::Mat &mat);

public:
    explicit OpenCvDNNWidget(QWidget *parent = nullptr);

private slots:
    bool loadImage();
    bool loadModel();
    bool loadClassNames();
    void detectObjects();
    void resetImage();

private:
    void setupUI();
    void printOpenCVInfo();
};

#endif // OPENCVDNNWIDGET_H