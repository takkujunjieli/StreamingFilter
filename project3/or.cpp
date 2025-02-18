/*
  Junjie Li
  January 2025

  This file offers all filters and effects that can be applied to an image/video.
*/

#include "or.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <random>
#include <vector>
#include "DA2Network.hpp"

using namespace std;
using namespace cv;

static const string dbFilePath = "C:/Users/alvin/Desktop/takkugit/projects/project3/build/Debug/object_db.csv";

// Process the frame based on the current mode
Mat processFrame(Mat &frame, const string &currentMode)
{
    Mat processedFrame;
    if (currentMode == "threshold")
    {
        if (threshold(frame, processedFrame) != 0)
        {
            processedFrame = frame.clone();
        }
    }
    else if (currentMode == "depthEstimation")
    {
        if (estimateDepth(frame, processedFrame) != 0)
        {
            processedFrame = frame.clone();
        }
    }
    else if (currentMode == "original")
    {
        processedFrame = frame.clone();
    }
    else if (currentMode == "threshold_with_clean")
    {
        if (threshold_with_clean(frame, processedFrame) != 0)
        {
            processedFrame = frame.clone();
        }
    }
    else if (currentMode == "analyze_and_display_regions")
    {
        analyze_and_display_regions(frame, processedFrame);
    }

    else if (currentMode == "computeFeaturesForRegions")
    {
        computeFeaturesForRegions(frame, processedFrame);
    }
    else if (currentMode == "classifyImages")
    {
        classifyImages(frame, processedFrame);
    }
    else
    {
        processedFrame = frame.clone();
    }
    return processedFrame;
}

// Convert an image to greyscale
int greyscale(Mat &src, Mat &dst)
{
    if (src.empty() || src.channels() != 3)
    {
        return -1;
    }

    vector<double> weights = {0.4, 0.45, 0.15};

    vector<Mat> channels(3);
    split(src, channels);

    Mat result;
    channels[0].convertTo(channels[0], CV_32F);
    channels[1].convertTo(channels[1], CV_32F);
    channels[2].convertTo(channels[2], CV_32F);

    result = channels[0] * weights[0] + channels[1] * weights[1] + channels[2] * weights[2];

    result *= 0.9;

    result.convertTo(result, CV_8U);

    merge(vector<Mat>{result, result, result}, dst);

    return 0;
}

// Apply a more efficient 5x5 blur to an image
int blur5x5_2(Mat &src, Mat &dst)
{
    // Ensure the source image is of type CV_8UC3
    if (src.type() != CV_8UC3)
    {
        return -1;
    }

    dst = src.clone();

    int kernel[5] = {1, 2, 4, 2, 1};
    int kernelSum = 16; // Sum of all kernel values

    Mat temp = src.clone();

    // First pass: vertical filter
    for (int y = 2; y < src.rows - 2; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int ky = -2; ky <= 2; ky++)
            {
                Vec3b pixel = src.ptr<Vec3b>(y + ky)[x];
                int kernelValue = kernel[ky + 2];

                sumB += pixel[0] * kernelValue;
                sumG += pixel[1] * kernelValue;
                sumR += pixel[2] * kernelValue;
            }

            temp.ptr<Vec3b>(y)[x][0] = sumB / kernelSum;
            temp.ptr<Vec3b>(y)[x][1] = sumG / kernelSum;
            temp.ptr<Vec3b>(y)[x][2] = sumR / kernelSum;
        }
    }

    // Second pass: horizontal filter
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 2; x < src.cols - 2; x++)
        {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int kx = -2; kx <= 2; kx++)
            {
                Vec3b pixel = temp.ptr<Vec3b>(y)[x + kx];
                int kernelValue = kernel[kx + 2];

                sumB += pixel[0] * kernelValue;
                sumG += pixel[1] * kernelValue;
                sumR += pixel[2] * kernelValue;
            }

            dst.ptr<Vec3b>(y)[x][0] = sumB / kernelSum;
            dst.ptr<Vec3b>(y)[x][1] = sumG / kernelSum;
            dst.ptr<Vec3b>(y)[x][2] = sumR / kernelSum;
        }
    }

    return 0;
}

// estimate the depth of an image using a depth-aware network
int estimateDepth(Mat &src, Mat &dst)
{
    if (src.empty())
    {
        printf("Input image is empty\n");
        return -1;
    }

    Mat dst_vis;
    const float reduction = 0.5;

    // make a DANetwork object
    DA2Network da_net("C:/Users/alvin/Desktop/takkugit/projects/project1/model_fp16.onnx");

    // for speed purposes, reduce the size of the input frame by half
    Mat resized_src;
    resize(src, resized_src, Size(), reduction, reduction);
    float scale_factor = 256.0 / (resized_src.rows * reduction);

    // set the network input
    da_net.set_input(resized_src, scale_factor);

    // run the network
    da_net.run_network(dst, resized_src.size());

    // apply a color map to the depth output to get a good visualization
    applyColorMap(dst, dst_vis, COLORMAP_INFERNO);

    return 0;
}

// Function to compute Euclidean distance
inline double euclideanDist(double a, double b)
{
    return abs(a - b);
}

// Custom K-means implementation for grayscale pixel clustering
void kmeansCustom(vector<float> &samples, int K, vector<float> &centers, vector<int> &labels, int maxIterations)
{
    // Initialize cluster centers randomly from the samples
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, samples.size() - 1);

    centers.resize(K);
    for (int i = 0; i < K; ++i)
    {
        centers[i] = samples[dis(gen)];
    }

    vector<float> newCenters(K, 0);
    vector<int> counts(K, 0);

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        labels.assign(samples.size(), 0);
        fill(newCenters.begin(), newCenters.end(), 0);
        fill(counts.begin(), counts.end(), 0);

        // Assign each sample to the closest cluster
        for (size_t i = 0; i < samples.size(); ++i)
        {
            double minDist = euclideanDist(samples[i], centers[0]);
            int clusterIndex = 0;

            for (int j = 1; j < K; ++j)
            {
                double dist = euclideanDist(samples[i], centers[j]);
                if (dist < minDist)
                {
                    minDist = dist;
                    clusterIndex = j;
                }
            }
            labels[i] = clusterIndex;
            newCenters[clusterIndex] += samples[i];
            counts[clusterIndex]++;
        }

        // Update cluster centers
        bool converged = true;
        for (int j = 0; j < K; ++j)
        {
            if (counts[j] > 0)
            {
                double newMean = newCenters[j] / counts[j];
                if (euclideanDist(centers[j], newMean) > 1e-3)
                {
                    converged = false;
                }
                centers[j] = newMean;
            }
        }

        if (converged)
            break; // Stop if converged
    }
}

// Function to convert BGR to HSV
void convertBGRtoHSV(const Mat &src, Mat &dst)
{
    dst.create(src.size(), src.type());

    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            Vec3b bgr = src.at<Vec3b>(y, x);
            float b = bgr[0] / 255.0;
            float g = bgr[1] / 255.0;
            float r = bgr[2] / 255.0;

            float maxVal = max(r, max(g, b));
            float minVal = min(r, min(g, b));
            float delta = maxVal - minVal;

            float h = 0, s = 0, v = maxVal;

            if (delta != 0)
            {
                s = delta / maxVal;

                if (r == maxVal)
                {
                    h = (g - b) / delta;
                }
                else if (g == maxVal)
                {
                    h = 2 + (b - r) / delta;
                }
                else
                {
                    h = 4 + (r - g) / delta;
                }

                h *= 60;
                if (h < 0)
                {
                    h += 360;
                }
            }

            v *= (1 - s);
            dst.at<Vec3b>(y, x) = Vec3b(static_cast<uchar>(h / 2), static_cast<uchar>(s * 255), static_cast<uchar>(v * 255));
        }
    }
}

int threshold(Mat &src, Mat &dst)
{
    if (src.empty())
    {
        return -1;
    }

    Mat blurred;
    blur5x5_2(src, blurred);

    Mat hsv;
    convertBGRtoHSV(blurred, hsv);

    // Extract the saturation channel
    vector<Mat> hsvChannels;
    split(hsv, hsvChannels);
    Mat saturation = hsvChannels[1];

    // Calculate histogram
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    Mat hist;
    calcHist(&saturation, 1, 0, Mat(), hist, 1, &histSize, &histRange);

    // Normalize the histogram
    hist /= saturation.total();

    // Compute cumulative sums and cumulative means
    vector<double> cumulativeSum(histSize, 0);
    vector<double> cumulativeMean(histSize, 0);
    cumulativeSum[0] = hist.at<float>(0);
    cumulativeMean[0] = 0;
    for (int i = 1; i < histSize; ++i)
    {
        cumulativeSum[i] = cumulativeSum[i - 1] + hist.at<float>(i);
        cumulativeMean[i] = cumulativeMean[i - 1] + i * hist.at<float>(i);
    }

    // Compute global mean
    double globalMean = cumulativeMean[histSize - 1];

    // Find the threshold that maximizes the between-class variance
    double maxVariance = 0;
    int optimalThreshold = 0;
    for (int t = 0; t < histSize; ++t)
    {
        double weightBackground = cumulativeSum[t];
        double weightForeground = 1 - weightBackground;
        if (weightBackground == 0 || weightForeground == 0)
        {
            continue;
        }
        double meanBackground = cumulativeMean[t] / weightBackground;
        double meanForeground = (globalMean - cumulativeMean[t]) / weightForeground;
        double betweenClassVariance = weightBackground * weightForeground * pow(meanBackground - meanForeground, 2);
        if (betweenClassVariance > maxVariance)
        {
            maxVariance = betweenClassVariance;
            optimalThreshold = t;
        }
    }

    // Apply the optimal threshold
    dst = Mat::zeros(saturation.size(), CV_8U);
    for (int y = 0; y < saturation.rows; ++y)
    {
        for (int x = 0; x < saturation.cols; ++x)
        {
            dst.at<uchar>(y, x) = (saturation.at<uchar>(y, x) > optimalThreshold) ? 255 : 0;
        }
    }

    return 0;
}

// Function to apply threshold and clean up the binary image using morphological operations
int threshold_with_clean(Mat &src, Mat &dst)
{
    // Call the existing threshold function
    if (threshold(src, dst) != 0)
    {
        return -1;
    }

    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

    // Apply morphological opening (erosion followed by dilation)
    morphologyEx(dst, dst, MORPH_OPEN, element);

    // Apply morphological closing (dilation followed by erosion)
    morphologyEx(dst, dst, MORPH_CLOSE, element);

    return 0;
}

int connectedComponentsWithStats(const Mat &binary, Mat &labels, Mat &stats, Mat &centroids)
{
    if (binary.empty() || binary.type() != CV_8UC1)
    {
        return -1;
    }

    labels = Mat::zeros(binary.size(), CV_32S);

    int label = 1;
    int rows = binary.rows;
    int cols = binary.cols;

    // Offsets for 8-connectivity
    int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

    // Vector to store stats and centroids
    vector<vector<int>> componentStats;
    vector<Point2d> componentCentroids;

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            if (binary.at<uchar>(y, x) == 255 && labels.at<int>(y, x) == 0)
            {
                // Start a new component
                queue<Point> q;
                q.push(Point(x, y));
                labels.at<int>(y, x) = label;

                int left = x, right = x, top = y, bottom = y, area = 0;
                double sumX = 0, sumY = 0;

                while (!q.empty())
                {
                    Point p = q.front();
                    q.pop();

                    int px = p.x;
                    int py = p.y;

                    // Update stats
                    left = min(left, px);
                    right = max(right, px);
                    top = min(top, py);
                    bottom = max(bottom, py);
                    area++;
                    sumX += px;
                    sumY += py;

                    // Check 8-connectivity
                    for (int i = 0; i < 8; ++i)
                    {
                        int nx = px + dx[i];
                        int ny = py + dy[i];

                        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows && binary.at<uchar>(ny, nx) == 255 && labels.at<int>(ny, nx) == 0)
                        {
                            q.push(Point(nx, ny));
                            labels.at<int>(ny, nx) = label;
                        }
                    }
                }

                // Store stats and centroids
                componentStats.push_back({left, top, right - left + 1, bottom - top + 1, area});
                componentCentroids.push_back(Point2d(sumX / area, sumY / area));

                label++;
            }
        }
    }

    // Convert stats and centroids to Mat
    stats = Mat(componentStats.size(), 5, CV_32S);
    centroids = Mat(componentCentroids.size(), 2, CV_64F);

    for (size_t i = 0; i < componentStats.size(); ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            stats.at<int>(i, j) = componentStats[i][j];
        }
        centroids.at<double>(i, 0) = componentCentroids[i].x;
        centroids.at<double>(i, 1) = componentCentroids[i].y;
    }

    return label;
}

int analyze_and_display_regions(Mat &src, Mat &dst)
{
    // Call the existing threshold_with_clean function
    Mat binary;
    if (threshold_with_clean(src, binary) != 0)
    {
        return -1;
    }

    // Perform connected components analysis
    Mat labels, stats, centroids;
    int numComponents = connectedComponentsWithStats(binary, labels, stats, centroids);

    // Initialize the dst image with the same size as src and type CV_8UC3
    dst = Mat::zeros(src.size(), CV_8UC3);

    // Filter out small regions and create a color map
    int minRegionSize = 500;                             // Adjust this value as needed
    vector<Vec3b> colors(numComponents, Vec3b(0, 0, 0)); // Initialize with black color

    // Generate random colors for each component
    RNG rng(12345);
    for (int i = 1; i < numComponents; ++i)
    {
        if (stats.at<int>(i, CC_STAT_AREA) >= minRegionSize)
        {
            colors[i] = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        }
    }

    // Create the region map directly on the dst image
    for (int y = 0; y < labels.rows; ++y)
    {
        for (int x = 0; x < labels.cols; ++x)
        {
            int label = labels.at<int>(y, x);
            if (label > 0 && stats.at<int>(label, CC_STAT_AREA) >= minRegionSize)
            {
                dst.at<Vec3b>(y, x) = colors[label];
            }
        }
    }

    return 0;
}

// Function to compute Haralick texture features
vector<double> computeHaralickFeatures(Mat &gray)
{
    vector<double> haralickFeatures(4, 0); // [Contrast, Energy, Homogeneity, Correlation]
    Mat glcm = Mat::zeros(256, 256, CV_32F);

    for (int i = 0; i < gray.rows - 1; i++)
    {
        for (int j = 0; j < gray.cols - 1; j++)
        {
            int intensity1 = gray.at<uchar>(i, j);
            int intensity2 = gray.at<uchar>(i, j + 1);
            glcm.at<float>(intensity1, intensity2)++;
        }
    }

    glcm /= sum(glcm)[0]; // Normalize

    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < 256; j++)
        {
            double p = glcm.at<float>(i, j);
            if (p > 0)
            {
                haralickFeatures[0] += (i - j) * (i - j) * p; // Contrast
                haralickFeatures[1] += p * p;                 // Energy
                haralickFeatures[2] += p / (1 + abs(i - j));  // Homogeneity
                haralickFeatures[3] += (i * j * p);           // Correlation
            }
        }
    }

    return haralickFeatures;
}

vector<vector<double>> computeFeaturesForRegions(Mat &src, Mat &dst)
{
    vector<vector<double>> featureVectors;

    // Convert to grayscale if the source image is not already grayscale
    Mat gray;
    if (src.channels() == 3)
    {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = src;
    }

    // Apply Gaussian blur to reduce noise
    GaussianBlur(gray, gray, Size(5, 5), 0);

    // Apply Otsu's thresholding for better segmentation
    Mat binary;
    // threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);

    // Apply morphological closing to remove small gaps
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);

    // Clone the source image for visualization
    dst = src.clone();

    // Find contours
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Filter out small contours by setting a minimum area threshold
    double minAreaThreshold = src.cols * src.rows * 0.005; // Ignore very small objects
    vector<vector<Point>> filteredContours;
    for (const auto &contour : contours)
    {
        if (contourArea(contour) > minAreaThreshold)
        {
            filteredContours.push_back(contour);
        }
    }

    // Iterate over each filtered contour and compute features
    for (const auto &contour : filteredContours)
    {
        // Compute moments
        Moments m = moments(contour);
        if (m.m00 == 0)
            continue; // Prevent division by zero

        double cx = m.m10 / m.m00;
        double cy = m.m01 / m.m00;

        // Compute covariance matrix components
        double mu20 = m.mu20 / m.m00;
        double mu02 = m.mu02 / m.m00;
        double mu11 = m.mu11 / m.m00;

        // Compute orientation angle
        double angle = 0.5 * atan2(2 * mu11, mu20 - mu02);

        // Draw the axis of least central moment
        int length = 150; // Increased length for better visualization
        Point p1(cx - length * cos(angle), cy - length * sin(angle));
        Point p2(cx + length * cos(angle), cy + length * sin(angle));
        line(dst, p1, p2, Scalar(0, 255, 0), 2); // Green line for orientation

        // Compute the oriented bounding box
        RotatedRect rect = minAreaRect(contour);
        Point2f boxPoints[4];
        rect.points(boxPoints);
        for (int j = 0; j < 4; j++)
        {
            line(dst, boxPoints[j], boxPoints[(j + 1) % 4], Scalar(0, 0, 255), 2); // Red box
        }

        // aspectRatio most be less than one
        double aspectRatio =
            rect.size.width > rect.size.height ? rect.size.width / rect.size.height : rect.size.height / rect.size.width;
        vector<Point> hull;
        convexHull(contour, hull);
        double solidity = contourArea(contour) / contourArea(hull);
        double extent = contourArea(contour) / (rect.size.width * rect.size.height);
        // double eccentricity = sqrt(1 - (mu20 / mu02));

        // Compute Hu Moments (scale and rotation invariant)
        vector<double> huMoments(7);
        HuMoments(m, huMoments);
        for (auto &h : huMoments)
            h = -1 * copysign(1.0, h) * log10(abs(h)); // Scale for better handling

        // Compute Haralick texture features
        vector<double> haralickFeatures = computeHaralickFeatures(gray);

        // Final feature vector (Scale-Invariant)
        vector<double> featureVector = {
            aspectRatio, extent, solidity,
            huMoments[0], huMoments[1], huMoments[2], huMoments[3],
            haralickFeatures[0], haralickFeatures[1], haralickFeatures[2], haralickFeatures[3]};

        featureVectors.push_back(featureVector);

        // Debug output for each contour
        cout << "Contour area: " << contourArea(contour) << endl;
        cout << "Bounding box center: (" << rect.center.x << ", " << rect.center.y << ")" << endl;
        cout << "Bounding box size: (" << rect.size.width << ", " << rect.size.height << ")" << endl;
    }

    return featureVectors;
}

// Function to collect scale-invariant features
int collectAndStoreFeatures(Mat &src, const string &label, const string &dbFilePath)
{
    Mat dst;
    vector<vector<double>> featureVectors = computeFeaturesForRegions(src, dst);

    if (featureVectors.empty())
    {
        cout << "No valid contours after filtering!" << endl;
        return -1;
    }

    ofstream dbFile(dbFilePath, ios::app);
    if (!dbFile.is_open())
    {
        cerr << "Error: Could not open database file!" << endl;
        return -1;
    }

    for (const auto &featureVector : featureVectors)
    {
        // Write feature vector to database
        dbFile << label;
        for (const auto &feature : featureVector)
        {
            dbFile << "," << feature;
        }
        dbFile << endl;
    }

    dbFile.close();
    return 0;
}

// Helper function to load feature vectors from the database
vector<pair<string, vector<double>>> loadFeatureVectors(const string &dbFilePath)
{
    vector<pair<string, vector<double>>> featureVectors;
    ifstream dbFile(dbFilePath);
    if (!dbFile.is_open())
    {
        cerr << "Error: Could not open database file!" << endl;
        return featureVectors;
    }

    string line;
    while (getline(dbFile, line))
    {
        stringstream ss(line);
        string label;
        getline(ss, label, ',');

        vector<double> features;
        string feature;
        while (getline(ss, feature, ','))
        {
            features.push_back(stod(feature));
        }

        featureVectors.push_back(make_pair(label, features));
    }

    dbFile.close();
    return featureVectors;
}

// Helper function to compute the scaled Euclidean distance between two feature vectors
double scaledEuclidean(const vector<double> &v1, const vector<double> &v2)
{
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i)
    {
        double diff = v1[i] - v2[i];
        sum += diff * diff / (v1[i] * v2[i]);
    }
    return sqrt(sum);
}

// Cosine Distance
double cosineDistance(const vector<double> &v1, const vector<double> &v2)
{
    double dot_product = 0.0, norm_v1 = 0.0, norm_v2 = 0.0;

    for (size_t i = 0; i < v1.size(); ++i)
    {
        dot_product += v1[i] * v2[i];
        norm_v1 += v1[i] * v1[i];
        norm_v2 += v2[i] * v2[i];
    }

    if (norm_v1 == 0 || norm_v2 == 0)
        return 1.0; // To handle zero-vector cases.

    return 1.0 - (dot_product / (sqrt(norm_v1) * sqrt(norm_v2)));
}

// Scaled L-infinity Distance
double scaledLInfDistance(const vector<double> &v1, const vector<double> &v2)
{
    double max_scaled_diff = 0.0;

    for (size_t i = 0; i < v1.size(); ++i)
    {
        double max_val = max(v1[i], v2[i]);
        if (max_val > 0) // Avoid division by zero
        {
            double scaled_diff = abs(v1[i] - v2[i]) / max_val;
            max_scaled_diff = max(max_scaled_diff, scaled_diff);
        }
    }
    return max_scaled_diff;
}

int classifyImages(Mat &src, Mat &dst)
{
    // Compute feature vectors for the input image
    vector<vector<double>> featureVectors = computeFeaturesForRegions(src, dst);

    if (featureVectors.empty())
    {
        cout << "No valid contours after filtering!" << endl;
        return -1;
    }

    // Load the feature vectors from the database
    vector<pair<string, vector<double>>> dbFeatureVectors = loadFeatureVectors(dbFilePath);

    // Define text position (Top-Right Corner)
    int marginRight = 100; // Adjust for padding from the right edge
    int marginTop = 30;    // Adjust for padding from the top

    // Iterate over each feature vector and classify the region
    for (size_t i = 0; i < featureVectors.size(); i++)
    {
        const auto &features = featureVectors[i];

        // Find the closest matching feature vector in the database
        string closestLabel;
        double minDistance = numeric_limits<double>::max();
        for (const auto &entry : dbFeatureVectors)
        {
            double distance = scaledEuclidean(features, entry.second);
            if (distance < minDistance)
            {
                minDistance = distance;
                closestLabel = entry.first;
            }
        }

        // Draw the label on the top-right of the image
        Point textPosition(dst.cols - marginRight, marginTop + (i * 30)); // Offset for multiple labels
        putText(dst, closestLabel, textPosition, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);

        // Store the feature vector back to the database with the generated label
        ofstream dbFile(dbFilePath, ios::app);
        if (!dbFile.is_open())
        {
            cerr << "Error: Could not open database file!" << endl;
            return -1;
        }

        dbFile << closestLabel;
        for (const auto &feature : features)
        {
            dbFile << "," << feature;
        }
        dbFile << endl;
        dbFile.close();
    }

    return 0;
}
