#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core.hpp>
#include <string>

// Image processing functions
int greyscale(cv::Mat &src, cv::Mat &dst);
cv::Mat processFrame(cv::Mat &input, const std::string &mode);

#endif // FILTER_H