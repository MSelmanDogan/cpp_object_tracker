# cpp_object_tracker
g++ main.cpp -o my_tracker objectTracker.cpp kalmanFilter.cpp $(pkg-config --cflags --libs opencv4)