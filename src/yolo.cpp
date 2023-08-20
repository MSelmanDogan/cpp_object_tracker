// detection.rows neden -1 geliyor boş gelmesi ?
// detections kısmını data'ya çevir for'u güncelle
#include <opencv2/opencv.hpp>
#include <fstream>

struct Detection {
  int classId;
  float confidence;
  cv::Rect box;
};
cv::Mat format_yolov5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}
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
  const float CONFIDENCE_THRESHOLD = 0.4;
  const float INPUT_WIDTH = 640.0;
  const float INPUT_HEIGHT = 640.0;
  const float SCORE_THRESHOLD = 0.6;
  const float NMS_THRESHOLD = 0.4;

  cv::dnn::Net net = cv::dnn::readNet("yolov5n.onnx");
  std::vector<cv::String> outputNames = net.getUnconnectedOutLayersNames();

  cv::VideoCapture video("ball.mp4");
  if (!video.isOpened()) {
    std::cout << "Video not opened." << std::endl;
    return -1;
  }

  std::vector<std::string> class_list = load_class_list();

  std::vector<Detection> output;

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
    auto input_image = format_yolov5(frame);
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    
    float *data = (float *)detections[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    // std::vector<Detection> detected_objects;

    // for (const cv::Mat& detection : detections) {
      // std::cout << "Detection matrix size: " << detection.rows << std::endl;
    for (int i = 0; i < rows; ++i) {
      float confidence = data[4];
      // std::cout << "Confidence: " << confidence << std::endl;
      if (confidence > CONFIDENCE_THRESHOLD) {
          float * classes_scores = data + 5;
          cv::Mat scores(1, class_list.size(), CV_32FC1, classes_scores);
          cv::Point class_id;
          double max_class_score;
          minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
          if (max_class_score > SCORE_THRESHOLD) {

              confidences.push_back(confidence);

              class_ids.push_back(class_id.x);

              float x = data[0];
              float y = data[1];
              float w = data[2];
              float h = data[3];
              int left = int((x - 0.5 * w) * x_factor);
              int top = int((y - 0.5 * h) * y_factor);
              int width = int(w * x_factor);
              int height = int(h * y_factor);
              boxes.push_back(cv::Rect(left, top, width, height));
          }

      }

      data += 85;

  }
    // }
    
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.classId = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
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
