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

using MatchingMethod = std::function<double(const std::vector<float> &, const std::vector<float> &)>;

double matchDistance(const std::vector<float> &target, const std::vector<float> &img2, MatchingMethod matchingMethod);
double sum_of_squared_differences(const std::vector<float> &target, const std::vector<float> &img2);
double histogramIntersection(const std::vector<float> &target, const std::vector<float> &img2);

#endif