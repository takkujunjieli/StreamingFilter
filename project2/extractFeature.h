#ifndef EXTRACT_FEATURE_H
#define EXTRACT_FEATURE_H

#include <string>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

using FeatureExtractor = std::function<void(const cv::Mat &, std::vector<float> &)>;

void extractFeature(const std::string &imagePath, std::vector<float> &features, FeatureExtractor featureMethod);
void feature_method1(const cv::Mat &image, std::vector<float> &image_data);

#endif // EXTRACT_FEATURE_H