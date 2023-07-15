#include <opencv2/opencv.hpp>

int main(){

    int minBlobArea = 200; 
    // int maxBlobArea = 1000;
    cv::VideoCapture video("./singleball.mp4");
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = cv::createBackgroundSubtractorMOG2();
    cv::KalmanFilter kalmanFilter(6,2,0);// 4 states (x, y, vx, vy) 2 measurements (x,y)

    cv::Mat measurement = cv::Mat::zeros(2,1,CV_32F); // Measurement matrix


    kalmanFilter.transitionMatrix = cv::Mat::eye(6,6,CV_32F);
    kalmanFilter.measurementMatrix = cv::Mat::zeros(2,6, CV_32F);

    kalmanFilter.measurementMatrix.at<float>(0) = 1.0f;
    kalmanFilter.measurementMatrix.at<float>(7) = 1.0f;
    kalmanFilter.measurementMatrix.at<float>(16) = 1.0f;
	kalmanFilter.measurementMatrix.at<float>(23) = 1.0f;

    kalmanFilter.transitionMatrix.at<float>(2) = 1.0f; // x hızı
    kalmanFilter.transitionMatrix.at<float>(9) = 1.0f; // y hızı
    // initialize other Kalman Filter parameters: process noise covariance, 
    // measurement noise covariance, error covariance

    //You can experiment with different values to get better tracking results

    cv::setIdentity(kalmanFilter.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kalmanFilter.measurementNoiseCov, cv::Scalar::all(1e-1));
    // cv::setIdentity(kalmanFilter.errorCovPost, cv::Scalar::all(1));

    cv::Point2f predictedPosition;
    cv::Point2f measuredPosition;
    bool isObjectDetected = false; // Track if the object is detected in the current
    // frame
    std::vector<cv::Point2f> previousMeasuredPosition; // to store measured values
    std::vector<cv::Point2f> previousPredictedPosition; // to store predicted values
    if (!video.isOpened()){
        std::cout << "Video not oppened." << std::endl;
        return -1;
    }

    cv::Mat frame;

    while (video.read(frame))
    {
        cv::Mat fgMask; // foreground mask

        // calculate foreground mask with mog2

        mog2->apply(frame,fgMask);
        // Finding contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        isObjectDetected = false; // Reset object detection flag for each frame
        // draw blobs for each contour
        for (size_t i = 0; i< contours.size(); i++){
            cv::Rect bbox = cv::boundingRect(contours[i]);
            int blobArea = bbox.width * bbox.height;

            if (blobArea >= minBlobArea) {
                cv::drawContours(
                    frame, contours, static_cast<int>(i),
                    cv::Scalar(0,0,255), 2
                    );
                cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
                
                // Update measurement with the centroid of the selected blob
                measuredPosition.x = bbox.x + bbox.width / 2;
                measuredPosition.y = bbox.y + bbox.height / 2;
                isObjectDetected = true;

                // Kalman Filter update step
                measurement.at<float>(0) = static_cast<float>(measuredPosition.x);
                measurement.at<float>(1) = static_cast<float>(measuredPosition.y);
                kalmanFilter.correct(measurement);
                
                // Kalman Filter prediction step
                cv::Mat prediction = kalmanFilter.predict();
                predictedPosition.x = prediction.at<float>(0);
                predictedPosition.y = prediction.at<float>(1);

                // Draw predicted and measure position
                
                cv::circle(frame, measuredPosition, 5, cv::Scalar(0, 255, 255), -1);
                cv::circle(frame, predictedPosition, 5, cv::Scalar(255, 255, 255), -1);

                // Update previous positions for the next frame
                previousMeasuredPosition.push_back(measuredPosition);
                previousPredictedPosition.push_back(predictedPosition);

                for (size_t j = 0; j < previousMeasuredPosition.size(); j++){
                    cv::circle(frame, previousMeasuredPosition[j], 5, cv::Scalar(0, 255, 255), -1);
                    cv::circle(frame, previousPredictedPosition[j], 5, cv::Scalar(255,255,255), -1);
                }
            }
            // else{
            //     // if no valid blobs are found, update Kalman Filter with 
            //     // no measurement

            //     kalmanFilter.predict();
            //     predictedPosition.x = kalmanFilter.statePost.at<float>(0);
            //     predictedPosition.y = kalmanFilter.statePost.at<float>(1);
            //     cv::circle(frame, predictedPosition, 5, cv::Scalar(255, 0, 0), -1);
            // }
    //     if (!isObjectDetected){
    //     // If no valid blobs are found, update Kalman Filter with no measurement 
    //     // and continue prediction
    //     kalmanFilter.predict();
    //     predictedPosition.x = kalmanFilter.statePost.at<float>(0);
    //     predictedPosition.y = kalmanFilter.statePost.at<float>(1);

    //     // Draw predicted position using the previous predicted position
    //     cv::line(frame, previousPredictedPosition, predictedPosition, cv::Scalar(255, 0, 0), 2);

    //     // Update the previous predicted position with the current predicted position
    //     previousPredictedPosition = predictedPosition;
    // }


        }

        cv::imshow("Frame", frame);
        cv::imshow("Foreground Mask", fgMask);

        if (cv::waitKey(600) == 27) // exit with ESC 
        //waitkey(600) because play on i wanna slow motion for my video 
            break;

    }

    video.release();
    cv::destroyAllWindows();
    
}