#ifndef OBJECTTRACKER_H

#define OBJECTTRACKER_H
#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"

class ObjectTracker {
public:
    ObjectTracker(int minBlobArea = 200);
    void processVideo(std::string& videoPath);

private:
    int minBlobArea;
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2;
    KalmanFilter kalmanFilter;
    cv::Point2f predictedPosition;
    cv::Point2f measuredPosition;
    std::vector<cv::Point2f> previousMeasuredPosition; // to store measured values
    std::vector<cv::Point2f> previousPredictedPosition; // to store predicted values
    // void trackObjects(cv::Mat& frame);
    // void drawTrackedObjects(cv::Mat& frame);
};

#endif // OBJECTTRACKER_H