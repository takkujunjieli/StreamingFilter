

#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <iostream>
#include "matchDistance.h"

// Alias for matching function type
using MatchingMethod = std::function<double(const std::vector<float> &, const std::vector<float> &)>;

// General matching function
double matchDistance(const std::vector<float> &target, const std::vector<float> &img2, MatchingMethod matchingMethod)
{
    // Ensure the images are of the same size
    if (target.size() != img2.size())
    {
        throw std::invalid_argument("Input images must have the same size.");
    }

    // Apply the given matching method
    return matchingMethod(target, img2);
}

// ssd matching method
double sum_of_squared_differences(const std::vector<float> &target, const std::vector<float> &img2)
{

    // Ensure the images are of the same size
    if (target.size() != img2.size())
    {
        throw std::invalid_argument("Input images must have the same size.");
    }
    double ssd = 0.0;

    // Compute the SSD
    for (size_t i = 0; i < target.size(); ++i)
    {
        double diff = target[i] - img2[i];
        ssd += diff * diff;
    }

    return ssd;
}