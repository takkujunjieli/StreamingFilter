/*
  Junjie Li
  Spring 2025

  This file contains the function prototypes for the image processing functions.
*/

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"

cv::Mat processFrame(cv::Mat &frame, const std::string &currentMode, cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &magnitudeImage);
int greyscale(cv::Mat &src, cv::Mat &dst);
int sepia(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
int bright(cv::Mat &src, cv::Mat &dst, int brightness);
int emboss(cv::Mat &src, cv::Mat &dst);
int colorfulFaces(cv::Mat &src, cv::Mat &dst);
int estimateDepth(cv::Mat &src, cv::Mat &dst);
int dynamicFilmGrain(cv::Mat &src, cv::Mat &dst, double intensity);
int vignette(cv::Mat &src, cv::Mat &dst);
int filmFlicker(cv::Mat &src, cv::Mat &dst, double intensity);
int oldDocumentary(cv::Mat &src, cv::Mat &dst);

#endif // FILTER_H