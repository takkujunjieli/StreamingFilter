/*
  Junjie Li
  Spring 2025

  This file contains the function prototypes for the image processing functions.
*/

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"

using namespace cv;
using namespace std;

Mat processFrame(Mat &frame, const string &currentMode);
int greyscale(Mat &src, Mat &dst);
int blur5x5_2(Mat &src, Mat &dst);
inline double euclideanDist(double a, double b);
void kmeansCustom(vector<float> &samples, int K, vector<float> &centers, vector<int> &labels, int maxIterations = 10);
void convertBGRtoHSV(const Mat &src, Mat &dst);
int threshold(Mat &src, Mat &dst);
int threshold_with_clean(Mat &src, Mat &dst);
int connectedComponentsWithStats(const Mat &binary, Mat &labels, Mat &stats, Mat &centroids);
int analyze_and_display_regions(Mat &src, Mat &dst);
vector<vector<double>> computeFeaturesForRegions(Mat &src, Mat &dst);
int collectAndStoreFeatures(Mat &src, const string &label, const string &dbFilePath);
int classifyImages(Mat &src, Mat &dst);

#endif // FILTER_H