#include "objectTracker.h"

ObjectTracker::ObjectTracker(int minBlobArea): minBlobArea(minBlobArea) {
    // ObjectTracker'ı başlatmak için gerekli işlemileri burada yap
    mog2 = cv::createBackgroundSubtractorMOG2();
    KNN = cv::createBackgroundSubtractorKNN();
    // kalmanFilter = KalmanFilter();
    std::vector<cv::Point2f> previousMeasuredPosition; // to store measured values
    std::vector<cv::Point2f> previousPredictedPosition; // to store predicted values
    bool isObjectDetected = false;
}

void ObjectTracker::processVideo(std::string& videoPath) {
    cv::VideoCapture video(videoPath);
    cv::Mat frame;


    if (!video.isOpened()) {
        std::cout << "Video not opened." << std::endl;
        return;
    }

    while (video.read(frame))
    {
        cv::Mat fgMask;
        // mog2->apply(frame,fgMask);
        KNN->apply(frame, fgMask);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        bool isObjectDetected = false; // Reset object detection flag for each frame
        // draw blobs for each contours
        for (size_t i = 0; i < contours.size(); i++) {
            cv::Rect bbox = cv::boundingRect(contours[i]);
            int blobArea = bbox.width * bbox.height;
            if (blobArea >= minBlobArea && blobArea <= 10000) {
                
                isObjectDetected = true;
                cv::drawContours(
                    frame, 
                    contours, 
                    static_cast<int>(i), 
                    cv::Scalar(0, 0, 255), 
                    2);
                cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);

                // Update measurement with the centroid of the selected blob
                measuredPosition.x = bbox.x + bbox.width / 2;
                measuredPosition.y = bbox.y + bbox.height / 2;
                
                kalmanFilter.correct(measuredPosition);
                kalmanFilter.predict(predictedPosition);
                predictedPosition = kalmanFilter.getPredictedPosition();

                // Kalman Filter update step
                // measurement.at<float>(0) = static_cast<float>(measuredPosition.x);
                // measurement.at<float>(1) = static_cast<float>(measuredPosition.y);
                // kalmanFilter.correct(measurement);

                // Kalman Filter prediction step

                // cv::Mat prediction = kalmanFilter.predict();
                // predictedPosition.x = prediction.at<float>(0);
                // predictedPosition.y = prediction.at<float>(1);

                // Draw predicted and measure position
                

                previousMeasuredPosition.push_back(measuredPosition);
                previousPredictedPosition.push_back(predictedPosition);
                for (size_t j = 0; j < previousMeasuredPosition.size(); j++) {
                    cv::circle(
                        frame, 
                        previousMeasuredPosition[j], 
                        5, 
                        cv::Scalar(255, 255, 0), 
                        -1);
                    cv::circle(
                        frame, 
                        previousPredictedPosition[j], 
                        5, 
                        cv::Scalar(0, 0, 255), 
                        -1);
                }
            }
            else if (
                blobArea < minBlobArea && 
                !previousPredictedPosition.empty() && 
                !isObjectDetected) {
                // There is no object now, but it was here
                isObjectDetected = true;
                measuredPosition = previousPredictedPosition.back();

                // measurement.at<float>(0) = static_cast<float>(measuredPosition.x);
                // measurement.at<float>(1) = static_cast<float>(measuredPosition.y);
                // kalmanFilter.correct(measurement);

                // // Kalman Filter prediction step
                // cv::Mat prediction = kalmanFilter.predict();
                // predictedPosition.x = prediction.at<float>(0);
                // predictedPosition.y = prediction.at<float>(1);

                // kalmanFilter.predict();
                kalmanFilter.correct(measuredPosition);
                kalmanFilter.predict(predictedPosition);
                predictedPosition = kalmanFilter.getPredictedPosition();
                // predictedPosition.x = kalmanFilter.statePost.at<float>(0);
                // predictedPosition.y = kalmanFilter.statePost.at<float>(1);

                previousMeasuredPosition.push_back(measuredPosition);
                previousPredictedPosition.push_back(predictedPosition);

                for (size_t j = 0; j < previousMeasuredPosition.size(); j++) {
                    cv::circle(
                        frame, 
                        previousMeasuredPosition[j], 
                        5, 
                        cv::Scalar(255, 0, 0),
                        -1);
                    cv::circle(
                        frame, 
                        previousPredictedPosition[j], 
                        5, 
                        cv::Scalar(255, 255, 255), 
                        -1);
                }
            }
            else {
                // Bu kısıma başka bir işlem eklemeyi unutmayın veya bu else 
                // bloğunu kaldırın
            }
        }
        // trackObjects(frame);
        // drawTrackedObjects(frame);

        cv::imshow("Frame", frame);
        cv::imshow("Foreground Mask", fgMask);

        if (cv::waitKey(600) == 27) 
            break;
    }

    video.release();
    cv::destroyAllWindows();

    // void ObjectTracker::trackObjects(cv::Mat& frame) {
    //     // Nesneleri takip etmek için gerekli işlemleri burada yapabilirsiniz
    // }
    
    // void ObjectTracker::drawTrackedObjects(cv::Mat& frame) {
    //     // Takip edilen nesneleri çizmek için gerekli işlemleri burada yap
    // }
}