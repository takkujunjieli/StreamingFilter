#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

cv::Mat processFrame(cv::Mat &frame, const std::string &currentMode, cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &magnitudeImage);
int greyscale(cv::Mat &src, cv::Mat &dst);
int sepia(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

#endif // FILTER_H