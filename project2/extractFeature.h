/*
   Junjie Li

    This file contains the declaration of the function extractFeature, which is used to extract features from an image.
*/

#ifndef EXTRACT_FEATURE_H
#define EXTRACT_FEATURE_H

#include <string>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

using FeatureExtractor = std::function<void(const std::string &, std::vector<float> &)>;

void extractFeature(const std::string &imagePath, std::vector<float> &features, FeatureExtractor featureMethod);
void extractCentralSquareFeature(const std::string &imagePath, std::vector<float> &image_data);
void extractColorHistogramFeature(const std::string &imagePath, std::vector<float> &image_data);

#endif // EXTRACT_FEATURE_H