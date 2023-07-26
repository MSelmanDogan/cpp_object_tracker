#include "kalmanFilter.h"

KalmanFilter::KalmanFilter(){
    // Kalman Filter'in kurulumu burada yapılır
    // (transition matrix, measurement matrix, process noise covariance,
    // measurement noise covariance, ...)
    kalmanFilter_obj = cv::KalmanFilter(6, 2, 0);
    // 4 states (x, y, vx, vy) measurements (x,y)


    kalmanFilter_obj.transitionMatrix = cv::Mat::eye(6, 6, CV_32F);
    kalmanFilter_obj.measurementMatrix = cv::Mat::zeros(2, 6, CV_32F);

    kalmanFilter_obj.measurementMatrix.at<float>(0) = 1.0f;
    kalmanFilter_obj.measurementMatrix.at<float>(7) = 1.0f;
    kalmanFilter_obj.measurementMatrix.at<float>(16) = 1.0f;
    kalmanFilter_obj.measurementMatrix.at<float>(23) = 1.0f;

    kalmanFilter_obj.transitionMatrix.at<float>(2) = 1.0f; // x hızı
    kalmanFilter_obj.transitionMatrix.at<float>(9) = 1.0f; // y hızı
    // initialize other Kalman Filter parameters: process noise covariance, 
    // measurement noise covariance, error covariance

    //You can experiment with different values to get better tracking results

    cv::setIdentity(kalmanFilter_obj.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kalmanFilter_obj.measurementNoiseCov, cv::Scalar::all(1e-1));
    // cv::setIdentity(kalmanFilter_obj.errorCovPost, cv::Scalar::all(1));
}

void KalmanFilter::intialize(float x, float y) {
    // Kalman Filter'ı başlatmak için kullanılır (ilk tahmin ve ölçüm)
}

void KalmanFilter::predict(cv::Point2f& predictedPosition) {
    // Kalman Filter öngörme adımı
    cv::Mat prediction = kalmanFilter_obj.predict();
    predictedPosition.x = prediction.at<float>(0);
    predictedPosition.y = prediction.at<float>(1);
}

void KalmanFilter::correct(const cv::Point2f& measuredPosition) {
    cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F); // Measurement matrix
    // Kalman filter düzeltme adımı (ölçümü içerir)
    measurement.at<float>(0) = static_cast<float>(measuredPosition.x);
    measurement.at<float>(1) = static_cast<float>(measuredPosition.y);
    kalmanFilter_obj.correct(measurement);
}

cv::Point2f KalmanFilter::getPredictedPosition() const {
    // Kalman Filter tarafından tahmin edilen pozisyonu döndürür
    
    cv::Point2f predictedPosition(
        kalmanFilter_obj.statePost.at<float>(0), 
        kalmanFilter_obj.statePost.at<float>(1)
        );

    return predictedPosition;
}