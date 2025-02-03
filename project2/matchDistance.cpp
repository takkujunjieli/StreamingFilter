

#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <iostream>
#include "matchDistance.h"

using namespace cv;
using namespace std;

// Alias for matching function type
using MatchingMethod = function<double(const vector<float> &, const vector<float> &)>;

/**
 * Abstract matching function.
 * @param target The target image.
 * @param img2 The second image.
 * @param matchingMethod Matching method.
 * @return The distance between the two images.
 */
double matchDistance(const vector<float> &target, const vector<float> &img2, MatchingMethod matchingMethod)
{
    // Ensure the images are of the same size
    if (target.size() != img2.size())
    {
        throw invalid_argument("Input images must have the same size.");
    }

    // Apply the given matching method
    return matchingMethod(target, img2);
}

/**
 * Distance function that computes the sum of squared differences (SSD) between two images.
 * The result ranges from 0 to infinity, where 0 means perfect similarity.
 * @param target The target image.
 * @param img2 The second image.
 * @return The sum of squared differences between the two images.
 */
double sum_of_squared_differences(const vector<float> &target, const vector<float> &img2)
{

    // Ensure the images are of the same size
    if (target.size() != img2.size())
    {
        throw invalid_argument("Input images must have the same size.");
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

/**
 * Computes the histogram intersection similarity between two histograms.
 *
 * @param target The first histogram (normalized vector).
 * @param img2 The second histogram (normalized vector).
 * @return The histogram intersection similarity (higher means more similar).
 */
double histogramIntersection(const vector<float> &target, const vector<float> &img2)
{
    if (target.size() != img2.size())
    {
        cerr << "Error: Histograms must have the same size for comparison!" << endl;
        return -1.0;
    }

    double intersection = 0.0;

    // Compute sum of minimum values bin-wise
    for (size_t i = 0; i < target.size(); i++)
    {
        intersection += min(target[i], img2[i]);
    }

    return intersection; // Higher value means more similar
}
