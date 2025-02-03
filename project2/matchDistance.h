/*
  Bruce A. Maxwell

  Utility functions for reading and writing CSV files with a specific format

  Each line of the csv file is a filename in the first column, followed by numeric data for the remaining columns
  Each line of the csv file has to have the same number of columns
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

#endif