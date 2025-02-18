#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <functional>
#include <numeric>
#include <iostream>
#include "extractFeature.h"
#include "DA2Network.hpp"
#include <corecrt_math_defines.h>

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
 * @param image Input image filename.
 * @param image_data Output 1D vector containing the feature.
 * @return void
 */
void extractCentralSquareFeature(const string &image_filename, vector<float> &image_data)
{
    Mat image = imread(image_filename, IMREAD_GRAYSCALE);

    // Extract the 7x7 square in the middle of the image
    Rect roi(image.cols / 2 - 3, image.rows / 2 - 3, 7, 7);
    Mat feature = image(roi);

    // Convert the feature to a 1D vector
    feature.convertTo(feature, CV_32F);
    image_data.assign((float *)feature.datastart, (float *)feature.dataend);
}

/**
 * Extracts a normalized 2D Hue-Saturation color histogram as a feature.
 * @param image Input image in grayscale.
 * @param image_data Output 1D vector containing the feature.
 * @return void
 */
void extractHSFeature(const string &image_filename, vector<float> &image_data)
{
    Mat image = imread(image_filename, IMREAD_COLOR);

    int r_bins = 8, g_bins = 8, b_bins = 8; // Number of bins
    int histSize[] = {r_bins, g_bins, b_bins};
    float r_ranges[] = {0, 256};
    float g_ranges[] = {0, 256};
    float b_ranges[] = {0, 256};
    const float *ranges[] = {r_ranges, g_ranges, b_ranges};
    int channels[] = {0, 1, 2}; // B, G, R channels

    // Compute histogram
    Mat hist;
    calcHist(&image, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);

    // Normalize histogram
    hist /= (image.rows * image.cols);

    // Convert histogram to 1D vector
    image_data.clear();
    for (int r = 0; r < r_bins; r++)
    {
        for (int g = 0; g < g_bins; g++)
        {
            for (int b = 0; b < b_bins; b++)
            {
                image_data.push_back(hist.at<float>(r, g, b));
            }
        }
    }
}

/**
 * Extracts a feature vector using two color histograms of HSV.
 * The image is divided into top and bottom halves, and an 8-bin 2D histogram
 * is computed separately for each half.
 *
 * @param image_filename Input image filename.
 * @param image_data Output 1D vector containing the feature.
 * @return void
 */
void extractTwoHSVHistFeature(const string &image_filename, vector<float> &image_data)
{
    Mat image = imread(image_filename, IMREAD_COLOR);

    // Convert image to HSV color space
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    // Determine image halves
    int mid_row = hsvImage.rows / 2;
    Mat topHalf = hsvImage(Rect(0, 0, hsvImage.cols, mid_row));                          // Top half
    Mat bottomHalf = hsvImage(Rect(0, mid_row, hsvImage.cols, hsvImage.rows - mid_row)); // Bottom half

    // Define histogram parameters
    int h_bins = 8, s_bins = 8;
    int histSize[] = {h_bins, s_bins};
    float h_ranges[] = {0, 180}; // Hue ranges from 0 to 180
    float s_ranges[] = {0, 256}; // Saturation ranges from 0 to 255
    const float *ranges[] = {h_ranges, s_ranges};
    int channels[] = {0, 1}; // Hue (channel 0) and Saturation (channel 1)

    // Compute histograms
    Mat hist_top, hist_bottom;
    calcHist(&topHalf, 1, channels, Mat(), hist_top, 2, histSize, ranges, true, false);
    calcHist(&bottomHalf, 1, channels, Mat(), hist_bottom, 2, histSize, ranges, true, false);

    // Normalize histograms
    int total_pixels_top = topHalf.rows * topHalf.cols;
    int total_pixels_bottom = bottomHalf.rows * bottomHalf.cols;
    hist_top /= total_pixels_top;
    hist_bottom /= total_pixels_bottom;

    // Flatten histograms into the feature vector
    image_data.clear();
    for (int h = 0; h < h_bins; h++)
    {
        for (int s = 0; s < s_bins; s++)
        {
            image_data.push_back(hist_top.at<float>(h, s)); // Top half features
        }
    }
    for (int h = 0; h < h_bins; h++)
    {
        for (int s = 0; s < s_bins; s++)
        {
            image_data.push_back(hist_bottom.at<float>(h, s)); // Bottom half features
        }
    }
}

/**
 * Extracts a feature vector using a whole image color histogram (HSV)
 * and a whole image texture histogram (Sobel gradient magnitudes).
 *
 * @param image_filename Input image filename.
 * @param image_data Output 1D vector containing the feature.
 * @return void
 */
void extractColorTextureFeature(const string &image_filename, vector<float> &image_data)
{
    Mat image = imread(image_filename);

    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    int h_bins = 8, s_bins = 8;
    int histSize[] = {h_bins, s_bins};
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    const float *ranges[] = {h_ranges, s_ranges};
    int channels[] = {0, 1};

    // Compute color histogram
    Mat color_hist;
    calcHist(&hsvImage, 1, channels, Mat(), color_hist, 2, histSize, ranges, true, false);

    color_hist /= (hsvImage.rows * hsvImage.cols);

    Mat grayImage, grad_x, grad_y, grad_mag;

    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    Sobel(grayImage, grad_x, CV_32F, 1, 0, 3);
    Sobel(grayImage, grad_y, CV_32F, 0, 1, 3);

    magnitude(grad_x, grad_y, grad_mag);

    // Histogram parameters for gradient magnitudes
    int t_bins = 8; // Number of bins for texture
    int histSizeTexture[] = {t_bins};
    float t_ranges[] = {0, 255}; // Gradient magnitude range
    const float *rangesTexture[] = {t_ranges};

    // Compute gradient magnitude histogram
    Mat texture_hist;
    calcHist(&grad_mag, 1, 0, Mat(), texture_hist, 1, histSizeTexture, rangesTexture, true, false);

    texture_hist /= (grad_mag.rows * grad_mag.cols);

    image_data.clear();

    // Flatten color histogram into feature vector
    for (int h = 0; h < h_bins; h++)
    {
        for (int s = 0; s < s_bins; s++)
        {
            image_data.push_back(color_hist.at<float>(h, s));
        }
    }

    // Flatten texture histogram into feature vector
    for (int t = 0; t < t_bins; t++)
    {
        image_data.push_back(texture_hist.at<float>(t));
    }
}

/**
 * Extracts a set of features from the image to determine if it represents an outdoor scene.
 *
 * @param image_filename Input image filename.
 * @param image_data Output 1D vector containing the extracted features.
 * @return void
 */
void extractOutdoors(const std::string &image_filename, std::vector<float> &image_data)
{
    Mat image = imread(image_filename, IMREAD_COLOR);

    // Convert to grayscale
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Compute brightness distribution
    int height = gray.rows;
    int width = gray.cols;

    // Define regions for upper and lower halves
    Mat upper_half = gray(Range(0, height / 2), Range(0, width));
    Mat lower_half = gray(Range(height / 2, height), Range(0, width));

    // Calculate mean brightness
    double upper_mean = mean(upper_half)[0];
    double lower_mean = mean(lower_half)[0];

    // Feature: Ratio of upper to lower brightness
    float brightness_ratio = static_cast<float>(upper_mean / (lower_mean + 1e-5));

    // Additional features
    Scalar mean_scalar, stddev_scalar;

    // Overall image contrast (standard deviation of grayscale values)
    meanStdDev(gray, mean_scalar, stddev_scalar);
    float contrast = static_cast<float>(stddev_scalar[0]);

    // Edge density using Canny edge detector
    Mat edges;
    Canny(gray, edges, 100, 200);
    float edge_density = static_cast<float>(countNonZero(edges)) / (height * width);

    // Store features in output vector
    image_data.clear();
    image_data.push_back(brightness_ratio);
    image_data.push_back(contrast);
    image_data.push_back(edge_density);
}

std::vector<float> computeSpatialVarianceFeatures(const cv::Mat &image)
{
    cv::Mat hsv, lab;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);

    int grid_size = 8; // 8x8 grid = 64 regions
    int step_x = image.cols / grid_size;
    int step_y = image.rows / grid_size;

    std::vector<float> spatial_variance_features;

    for (int i = 0; i < grid_size; ++i)
    {
        for (int j = 0; j < grid_size; ++j)
        {
            cv::Rect region(j * step_x, i * step_y, step_x, step_y);
            cv::Mat hsv_region = hsv(region);
            cv::Mat lab_region = lab(region);

            cv::Scalar mean_hsv, stddev_hsv, mean_lab, stddev_lab;
            cv::meanStdDev(hsv_region, mean_hsv, stddev_hsv);
            cv::meanStdDev(lab_region, mean_lab, stddev_lab);

            // Extract 8 features per region
            spatial_variance_features.push_back(stddev_hsv[0]); // Hue variance
            spatial_variance_features.push_back(mean_hsv[0]);   // Hue mean
            spatial_variance_features.push_back(stddev_lab[1]); // A variance
            spatial_variance_features.push_back(mean_lab[1]);   // A mean
            spatial_variance_features.push_back(stddev_lab[2]); // B variance
            spatial_variance_features.push_back(mean_lab[2]);   // B mean
            spatial_variance_features.push_back(stddev_hsv[1]); // Saturation variance
            spatial_variance_features.push_back(mean_hsv[1]);   // Saturation mean
        }
    }

    // Normalize the feature vector using Standardization
    float mean_val = std::accumulate(spatial_variance_features.begin(), spatial_variance_features.end(), 0.0f) / spatial_variance_features.size();
    float sq_sum = std::inner_product(spatial_variance_features.begin(), spatial_variance_features.end(), spatial_variance_features.begin(), 0.0f);
    float stdev_val = std::sqrt(sq_sum / spatial_variance_features.size() - mean_val * mean_val);

    for (float &val : spatial_variance_features)
    {
        val = (val - mean_val) / stdev_val;
    }

    return spatial_variance_features; // Total: 64 × 8 = 512 features
}

void extractBanana(const std::string &image_filename, std::vector<float> &image_data)
{
    cv::Mat image = cv::imread(image_filename);

    // Convert to HSV color space
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Define yellow color range
    cv::Scalar lower_yellow(20, 100, 100); // Lower bound for yellow
    cv::Scalar upper_yellow(30, 255, 255); // Upper bound for yellow

    // Threshold the HSV image to get only yellow colors
    cv::Mat mask;
    cv::inRange(hsv, lower_yellow, upper_yellow, mask);

    // Find contours in the masked image
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find the largest contour (assuming it's the banana)
    double max_area = 0;
    std::vector<cv::Point> banana_contour;
    cv::Rect banana_bbox;
    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);
        cv::Rect bbox = cv::boundingRect(contour);
        float aspect_ratio = static_cast<float>(bbox.width) / bbox.height;

        // Filter out large yellow backgrounds or non-banana shapes
        if (area > max_area && aspect_ratio > 1.5 && aspect_ratio < 3.5 && area < (image.cols * image.rows * 0.3))
        {
            max_area = area;
            banana_contour = contour;
            banana_bbox = bbox;
        }
    }

    // Extract feature (aspect ratio)
    float aspect_ratio = static_cast<float>(banana_bbox.width) / banana_bbox.height;

    // Store feature in image_data
    image_data.clear();
    image_data.push_back(aspect_ratio);
}
