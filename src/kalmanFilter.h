#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include <opencv2/opencv.hpp>

class KalmanFilter {
    public:
    KalmanFilter();
    void intialize(float x, float y);
    void correct(const cv::Point2f& measuredPosition);
    void predict(cv::Point2f& predictedPosition);
    cv::Point2f getPredictedPosition() const;

    private:
    cv::KalmanFilter kalmanFilter;
    cv::Point2f predictedPosition;
    cv::Point2f measuredPosition;
    cv::Mat measurement;
    cv::KalmanFilter kalmanFilter_obj;
};

#endif // KALMANFILTER_H