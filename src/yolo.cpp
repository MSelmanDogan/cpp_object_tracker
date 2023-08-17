// detection.rows neden -1 geliyor boş gelmesi ?
// detections kısmını data'ya çevir for'u güncelle
#include <opencv2/opencv.hpp>
#include <fstream>

struct Detection {
  int classId;
  float confidence;
  cv::Rect box;
};

std::vector<std::string> load_class_list() {
  std::vector<std::string> class_list;
  std::ifstream ifs("classes2.txt");
  std::string line;
  while (getline(ifs, line)) {
    class_list.push_back(line);
  }
  return class_list;
}

int main() {
  const float confidence_threshold = 0.01;

  cv::dnn::Net net = cv::dnn::readNet("yolov5n.onnx");
  std::vector<cv::String> outputNames = net.getUnconnectedOutLayersNames();

  cv::VideoCapture video("ball.mp4");
  if (!video.isOpened()) {
    std::cout << "Video not opened." << std::endl;
    return -1;
  }

  std::vector<std::string> class_list = load_class_list();

  const float INPUT_WIDTH = 640.0;
  const float INPUT_HEIGHT = 640.0;

  while (true) {
    cv::Mat frame;
    video >> frame;
    if (frame.empty()) {
      break;
    }

    cv::Mat blob = cv::dnn::blobFromImage(
      frame, 1.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
      cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> detections;
    net.forward(detections, outputNames);
    if (detections.empty()) {
      std::cout << "No detections found." << std::endl;
      continue;  // Sonraki kareye geç
    }

    std::vector<Detection> detected_objects;

    for (const cv::Mat& detection : detections) {
      // std::cout << "Detection matrix size: " << detection.rows << std::endl;
      for (int i = 0; i < detection.rows; ++i) {
        float confidence = detection.at<float>(i, 4);
        std::cout << "Confidence: " << confidence << std::endl;
        if (confidence > confidence_threshold) {
          int classId = static_cast<int>(detection.at<float>(i, 1));
          int left = static_cast<int>(detection.at<float>(i, 0) * frame.cols);
          int top = static_cast<int>(detection.at<float>(i, 1) * frame.rows);
          int right = static_cast<int>(detection.at<float>(i, 2) * frame.cols);
          int bottom = static_cast<int>(detection.at<float>(i, 3) * frame.rows);
          
          Detection detected_object;
          detected_object.classId = classId;
          detected_object.confidence = confidence;
          detected_object.box = cv::Rect(left, top, right - left, bottom - top);
          detected_objects.push_back(detected_object);
        }
      }
    }
    
    for (const Detection& detected_object : detected_objects) {
      std::cout << "data\n";
      std::cout << class_list[detected_object.classId];
      cv::rectangle(frame, detected_object.box, cv::Scalar(0, 255, 0), 2);
      cv::putText(frame, "Class: " + class_list[detected_object.classId], 
            cv::Point(detected_object.box.x, detected_object.box.y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Object Detection", frame);

    if (cv::waitKey(1) == 27) {
      break;
    }
  }

  video.release();
  cv::destroyAllWindows();

  return 0;
}
