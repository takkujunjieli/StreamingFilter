#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <iostream>
#include "extractFeature.h"

using namespace cv;
using namespace std;

// Alias for feature extraction function type
using FeatureExtractor = function<void(const string &, vector<float> &)>;

/**
 * Abstract feature extraction function.
 * @param image_filename Input image filename.
 * @param image_data Output 1D vector containing the feature.
 * @param featureMethod Feature extraction method.
 * @return void
 */
void extractFeature(const string &image_filename, vector<float> &image_data, FeatureExtractor featureMethod)
{
    // Apply the given feature extraction method
    featureMethod(image_filename, image_data);
}

/**
 * Extracts the 7x7 square in the middle of the image as a feature.
 * The feature is converted to a 1D vector.
 * @param image Input image in grayscale.
 * @param image_data Output 1D vector containing the feature.
 * @return void
 */
void extractCentralSquareFeature(const string &image_filename, vector<float> &image_data)
{
    // Load the image
    Mat image = imread(image_filename, IMREAD_GRAYSCALE);

    if (image.empty())
    {
        cerr << "Error: Unable to load image " << image_filename << endl;
        return;
    }

    // Check if the image is large enough
    if (image.rows < 7 || image.cols < 7)
    {
        cerr << "Error: Image is too small for feature extraction" << endl;
        return;
    }

    // Extract the 7x7 square in the middle of the image
    Rect roi(image.cols / 2 - 3, image.rows / 2 - 3, 7, 7);
    Mat feature = image(roi);

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

/**
 * Extracts a normalized 2D Hue-Saturation color histogram as a feature.
 * @param image Input image in grayscale.
 * @param image_data Output 1D vector containing the feature.
 * @return void
 */
void extractColorHistogramFeature(const string &image_filename, vector<float> &image_data)
{
    // Load the image
    Mat image = imread(image_filename, IMREAD_COLOR);

    if (image.empty())
    {
        cerr << "Error: Unable to load image " << image_filename << endl;
        return;
    }

    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    int h_bins = 16, s_bins = 16; // Number of bins
    Mat hist = Mat::zeros(h_bins, s_bins, CV_32F);
    int total_pixels = hsvImage.rows * hsvImage.cols;

    for (int i = 0; i < hsvImage.rows; i++)
    {
        for (int j = 0; j < hsvImage.cols; j++)
        {
            Vec3b pixel = hsvImage.at<Vec3b>(i, j);
            int h = pixel[0], s = pixel[1];

            int h_idx = (h * h_bins) / 180;
            int s_idx = (s * s_bins) / 256;

            hist.at<float>(h_idx, s_idx)++;
        }
    }

    hist /= total_pixels;

    // Convert histogram to 1D vector
    image_data.clear();
    for (int i = 0; i < hist.rows; i++)
    {
        for (int j = 0; j < hist.cols; j++)
        {
            image_data.push_back(hist.at<float>(i, j));
        }
    }
}