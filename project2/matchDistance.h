/*
  Junjie Li

  This file contains the declaration of the function matchDistance, which is used to calculate the distance between two vectors.
 */

#ifndef MATCH_DISTANCE_H
#define MATCH_DISTANCE_H

#include <string>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

using namespace std;
using MatchingMethod = function<double(const vector<float> &, const vector<float> &)>;

/**
 * Abstract matching function.
 * @param target The target image.
 * @param img2 The second image.
 * @param matchingMethod Matching method.
 * @return The distance between the two images.
 */
double matchDistance(const vector<float> &target, const vector<float> &img2, MatchingMethod matchingMethod);

/**
 * Distance function that computes the sum of squared differences (SSD) between two images.
 * The result ranges from 0 to infinity, where 0 means perfect similarity.
 * @param target The target image.
 * @param img2 The second image.
 * @return The sum of squared differences between the two images.
 */
double sum_of_squared_differences(const vector<float> &target, const vector<float> &img2);

/**
 * Computes the histogram intersection similarity between two histograms.
 *
 * @param target The first histogram (normalized vector).
 * @param img2 The second histogram (normalized vector).
 * @return one minus The histogram intersection similarity.
 */
double histogramIntersection(const vector<float> &target, const vector<float> &img2);

/**
 * Computes the histogram intersection similarity between two histograms
 * and combines the distances using weighted averaging.
 *
 * @param target The first histogram (normalized vector).
 * @param img2 The second histogram (normalized vector).
 * @return one minus double The histogram intersection similarity.
 */
double twoHistogram(const vector<float> &target, const vector<float> &img2);

/**
 * Computes the histogram intersection similarity between two feature vectors.
 * Weights color and texture histograms equally.
 *
 * @param target The first histogram (normalized vector).
 * @param img2 The second histogram (normalized vector).
 * @return one minus The histogram intersection similarity.
 */
double oneColorOneTexture(const vector<float> &target, const vector<float> &img2);

/**
 * Computes the cosine similarity between two histograms.
 * Normalizes each vector by its L2-norm (Euclidean norm) before computing the dot product.
 *
 * @param target The first histogram (normalized vector).
 * @param img2 The second histogram (normalized vector).
 * @return one minus double similarity.
 */
double cosineSimilarity(const vector<float> &target, const vector<float> &img2);
#endif