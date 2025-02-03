#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <iostream>
#include "extractFeature.h"

// Alias for feature extraction function type
using FeatureExtractor = std::function<void(const cv::Mat &, std::vector<float> &)>;

// General feature extraction function
void extractFeature(const std::string &image_filename, std::vector<float> &image_data, FeatureExtractor featureMethod)
{
    // Load the image in grayscale
    cv::Mat image = cv::imread(image_filename, cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        std::cerr << "Error: Unable to load image " << image_filename << std::endl;
        return;
    }

    // Apply the given feature extraction method
    featureMethod(image, image_data);
}

// feature extraction function which use 7x7 square in the middle of the image as a feature vector
void feature_method1(const cv::Mat &image, std::vector<float> &image_data)
{
    // Check if the image is large enough
    if (image.rows < 7 || image.cols < 7)
    {
        std::cerr << "Error: Image is too small for feature extraction" << std::endl;
        return;
    }

    // Extract the 7x7 square in the middle of the image
    cv::Rect roi(image.cols / 2 - 3, image.rows / 2 - 3, 7, 7);
    cv::Mat feature = image(roi);

    // Convert the feature to a 1D vector
    image_data.clear();
    for (int i = 0; i < feature.rows; i++)
    {
        for (int j = 0; j < feature.cols; j++)
        {
            image_data.push_back(feature.at<uchar>(i, j));
        }
    }
}
