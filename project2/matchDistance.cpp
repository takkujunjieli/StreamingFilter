

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
 * @return one minus The histogram intersection similarity.
 */
double histogramIntersection(const vector<float> &target, const vector<float> &img2)
{
    if (target.size() != img2.size())
    {
        cerr << "Error: Histograms must have the same size for comparison!" << endl;
        return -1.0;
    }

    double intersection = 0.0, sumTarget = 0.0;

    // Compute sum of minimum values bin-wise
    for (size_t i = 0; i < target.size(); i++)
    {
        intersection += min(target[i], img2[i]);
        sumTarget += target[i];
    }

    return (sumTarget > 0) ? (1.0 - (intersection / sumTarget)) : 1.0;
}

/**
 * Computes the histogram intersection similarity between two histograms
 * and combines the distances using weighted averaging.
 *
 * @param target The first histogram (normalized vector).
 * @param img2 The second histogram (normalized vector).
 * @return one minus double The histogram intersection similarity.
 */
double twoHistogram(const vector<float> &target, const vector<float> &img2)
{
    if (target.size() != img2.size() || target.empty())
    {
        cerr << "Error: Histograms must have the same size and not be empty." << endl;
        return 0.0;
    }

    int featureSize = target.size();
    int halfSize = featureSize / 2; // Each image has two histograms (top and bottom)

    double intersectionTop = 0.0, intersectionBottom = 0.0;
    double sumTop = 0.0, sumBottom = 0.0;

    // Compute histogram intersection for top half
    for (int i = 0; i < halfSize; i++)
    {
        intersectionTop += min(target[i], img2[i]);
        sumTop += target[i];
    }

    // Compute histogram intersection for bottom half
    for (int i = halfSize; i < featureSize; i++)
    {
        intersectionBottom += min(target[i], img2[i]);
        sumBottom += target[i];
    }

    // Normalize by total weight (avoid division by zero)
    double similarityTop = (sumTop > 0) ? intersectionTop / sumTop : 0.0;
    double similarityBottom = (sumBottom > 0) ? intersectionBottom / sumBottom : 0.0;

    // Weighted combination (adjust weights if needed)
    double weightTop = 0.5, weightBottom = 0.5;
    double finalSimilarity = (weightTop * similarityTop) + (weightBottom * similarityBottom);

    return 1.0 - finalSimilarity; // Higher values mean more similarity (max 1.0)
}

/**
 * Computes the histogram intersection similarity between two feature vectors.
 * Weights color and texture histograms equally.
 *
 * @param target The first histogram (normalized vector).
 * @param img2 The second histogram (normalized vector).
 * @return one minus The histogram intersection similarity.
 */
double oneColorOneTexture(const vector<float> &target, const vector<float> &img2)
{
    if (target.size() != img2.size() || target.empty())
    {
        cerr << "Error: Histogram vectors must be of the same size and non-empty." << endl;
        return -1.0f; // Return invalid score
    }

    int half_size = target.size() / 2; // since the two histograms are concatenated using same bins

    // Compute intersection sum for color histogram
    double color_intersection = 0.0;
    for (int i = 0; i < half_size; i++)
    {
        color_intersection += min(target[i], img2[i]);
    }

    // Compute intersection sum for texture histogram
    double texture_intersection = 0.0;
    for (int i = half_size; i < target.size(); i++)
    {
        texture_intersection += min(target[i], img2[i]);
    }

    // Average the two intersections to weight them equally
    return 1.0 - 0.5 * (color_intersection + texture_intersection);
}

/**
 * Computes the cosine similarity between two histograms.
 * Normalizes each vector by its L2-norm (Euclidean norm) before computing the dot product.
 *
 * @param target The first histogram (normalized vector).
 * @param img2 The second histogram (normalized vector).
 * @return one minus double similarity.
 */
double cosineSimilarity(const vector<float> &target, const vector<float> &img2)
{
    if (target.size() != img2.size() || target.empty())
    {
        cerr << "Error: Histogram vectors must be of the same size and non-empty." << endl;
        return -1.0; // Return invalid similarity
    }

    double dot_product = 0.0, norm_target = 0.0, norm_img2 = 0.0;

    // Compute dot product and L2 norms
    for (size_t i = 0; i < target.size(); i++)
    {
        dot_product += target[i] * img2[i];
        norm_target += target[i] * target[i];
        norm_img2 += img2[i] * img2[i];
    }

    // Compute L2 norms
    norm_target = sqrt(norm_target);
    norm_img2 = sqrt(norm_img2);

    // Avoid division by zero
    if (norm_target == 0.0 || norm_img2 == 0.0)
    {
        cerr << "Warning: One of the histograms has zero magnitude." << endl;
        return -1.0; // Return invalid similarity
    }

    // Compute cosine similarity
    return 1.0 - dot_product / (norm_target * norm_img2);
}
