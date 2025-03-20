#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Couldn't open camera.\n";
        return -1;
    }

    cv::namedWindow("Harris Corners", cv::WINDOW_AUTOSIZE);

    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int thresh = 200;

    cv::createTrackbar("Threshold: ", "Harris Corners", &thresh, 255);

    while (true)
    {
        cv::Mat frame, gray, dst, dst_norm, dst_norm_scaled;
        cap >> frame;
        if (frame.empty())
            break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::cornerHarris(gray, dst, blockSize, apertureSize, k);

        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
        cv::convertScaleAbs(dst_norm, dst_norm_scaled);

        for (int i = 0; i < dst_norm.rows; i++)
        {
            for (int j = 0; j < dst_norm.cols; j++)
            {
                if ((int)dst_norm.at<float>(i, j) > thresh)
                {
                    cv::circle(frame, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 2, 8, 0);
                }
            }
        }

        cv::imshow("Harris Corners", frame);

        char key = cv::waitKey(1);
        if (key == 27 || cv::getWindowProperty("Harris Corners", cv::WND_PROP_VISIBLE) < 1)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
