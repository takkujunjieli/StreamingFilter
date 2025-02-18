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

using namespace cv;
using namespace std;

using FeatureExtractor = function<void(const string &, vector<float> &)>;

/*
    Given an image file path, this function extracts features from the image and stores them in the vector features.
    The feature extraction method is specified by the function pointer featureMethod.
*/
void extractFeature(const string &imagePath, vector<float> &features, FeatureExtractor featureMethod);

/*
    Given an image file path, this function extracts the color histogram feature from the image and stores it in the vector image_data.
*/
void extractCentralSquareFeature(const string &imagePath, vector<float> &image_data);

/*
    Given an image file path, this function extracts the color histogram feature from the image and stores it in the vector image_data.
*/
void extractHSFeature(const string &imagePath, vector<float> &image_data);

/*
    Given an image file path, this function extracts the color histogram feature from the image and stores it in the vector image_data.
*/
void extractTwoHSVHistFeature(const string &image_filename, vector<float> &image_data);

/*
    Given an image file path, this function extracts the color histogram feature from the image and stores it in the vector image_data.
*/
void extractColorTextureFeature(const string &image_filename, vector<float> &image_data);

/*
    Given an image file path, this function extracts the color histogram feature from the image and stores it in the vector image_data.
*/
void extractOutdoors(const string &image_filename, vector<float> &image_data);

/*
    Given an image file path, this function extracts the color histogram feature from the image and stores it in the vector image_data.
*/
void extractBanana(const string &image_filename, vector<float> &image_data);

#endif // EXTRACT_FEATURE_H