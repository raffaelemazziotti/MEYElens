#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main() {
    cv::VideoCapture cap0(0);
    cv::VideoCapture cap1(1);

    if (!cap0.isOpened() || !cap1.isOpened()) {
        std::cerr << "Error: One or both cameras not available!" << std::endl;
        return -1;
    }

    cv::Mat frame0, frame1;
    auto last_time0 = std::chrono::steady_clock::now();
    auto last_time1 = last_time0;

    while (true) {
        cap0.read(frame0);
        cap1.read(frame1);

        if (frame0.empty() || frame1.empty()) break;

        // Compute FPS for camera 0
        auto now0 = std::chrono::steady_clock::now();
        double fps0 = 1.0 / std::chrono::duration<double>(now0 - last_time0).count();
        last_time0 = now0;

        // Compute FPS for camera 1
        auto now1 = std::chrono::steady_clock::now();
        double fps1 = 1.0 / std::chrono::duration<double>(now1 - last_time1).count();
        last_time1 = now1;

        // Overlay FPS text
        cv::putText(frame0, "FPS: " + std::to_string(int(fps0)), {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);
        cv::putText(frame1, "FPS: " + std::to_string(int(fps1)), {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);

        cv::imshow("Camera 0", frame0);
        cv::imshow("Camera 1", frame1);

        // Press ESC to quit
        if (cv::waitKey(1) == 27) break;
    }

    cap0.release();
    cap1.release();
    cv::destroyAllWindows();

    return 0;
}
