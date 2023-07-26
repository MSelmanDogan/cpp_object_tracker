#include "objectTracker.h"

int main() {
    // int minBlobArea = 200;
    ObjectTracker objectTracker;
    std::string videoPath = "../singleball.mp4";
    objectTracker.processVideo(videoPath);
    return 0;
}