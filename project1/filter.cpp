/*
  Junjie Li
  January 2025

  This file offers all filters and effects that can be applied to an image/video.
*/

#include "filter.h"
#include "faceDetect.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>

// Process the frame based on the current mode
cv::Mat processFrame(cv::Mat &frame, const std::string &currentMode, cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &magnitudeImage)
{
    cv::Mat processedFrame;
    if (currentMode == "opencv_gray")
    {
        cv::cvtColor(frame, processedFrame, cv::COLOR_BGR2GRAY);
        cv::cvtColor(processedFrame, processedFrame, cv::COLOR_GRAY2BGR);
    }
    else if (currentMode == "custom_gray")
    {
        if (greyscale(frame, processedFrame) != 0)
        {
            processedFrame = frame.clone();
        }
    }
    else if (currentMode == "sepia")
    {
        if (sepia(frame, processedFrame) != 0)
        {
            processedFrame = frame.clone();
        }
    }
    else if (currentMode == "blur")
    {
        if (blur5x5_2(frame, processedFrame) != 0)
        {
            processedFrame = frame.clone();
        }
    }
    else if (currentMode == "sobelX3x3")
    {
        sobelX3x3(frame, sobelX);
        cv::convertScaleAbs(sobelX, processedFrame);
    }
    else if (currentMode == "sobelY3x3")
    {
        sobelY3x3(frame, sobelY);
        cv::convertScaleAbs(sobelY, processedFrame);
    }
    else if (currentMode == "magnitude")
    {
        sobelX3x3(frame, sobelX);
        sobelY3x3(frame, sobelY);
        magnitude(sobelX, sobelY, magnitudeImage);
        processedFrame = magnitudeImage.clone();
    }
    else if (currentMode == "blurQuantize")
    {
        if (blurQuantize(frame, processedFrame, 10) != 0)
        {
            processedFrame = frame.clone();
        }
    }
    else if (currentMode == "faceDetect")
    {
        cv::Mat grey;
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        if (detectFaces(grey, faces) == 0)
        {
            drawBoxes(frame, faces);
        }
        processedFrame = frame.clone();
    }
    else if (currentMode == "brightness")
    {
        if (bright(frame, processedFrame, 50) != 0)
        {
            processedFrame = frame.clone();
        }
    }
    else if (currentMode == "emboss")
    {
        if (emboss(frame, processedFrame) != 0)
        {
            processedFrame = frame.clone();
        }
    }
    else if (currentMode == "colorfulFaces")
    {
        if (colorfulFaces(frame, processedFrame) != 0)
        {
            processedFrame = frame.clone();
        }
    }
    else if (currentMode == "oldDocumentary")
    {
        if (oldDocumentary(frame, processedFrame) != 0)
        {
            processedFrame = frame.clone();
        }
    }
    else
    {
        processedFrame = frame.clone();
    }
    return processedFrame;
}

// Convert an image to greyscale
int greyscale(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty() || src.channels() != 3)
    {
        return -1;
    }

    std::vector<double> weights = {0.4, 0.45, 0.15};

    std::vector<cv::Mat> channels(3);
    cv::split(src, channels);

    cv::Mat result;
    channels[0].convertTo(channels[0], CV_32F);
    channels[1].convertTo(channels[1], CV_32F);
    channels[2].convertTo(channels[2], CV_32F);

    result = channels[0] * weights[0] + channels[1] * weights[1] + channels[2] * weights[2];

    result *= 0.9;

    result.convertTo(result, CV_8U);

    cv::merge(std::vector<cv::Mat>{result, result, result}, dst);

    return 0;
}

// Convert an image to sepia
int sepia(cv::Mat &input, cv::Mat &output)
{
    if (input.empty() || input.channels() != 3)
    {
        return -1;
    }

    // (blue, green, red) based on requirment
    std::vector<double> weights = {0.272, 0.534, 0.131,
                                   0.349, 0.686, 0.168,
                                   0.393, 0.769, 0.189};

    std::vector<cv::Mat> channels(3);
    cv::split(input, channels);

    cv::Mat result;
    channels[0].convertTo(channels[0], CV_32F);
    channels[1].convertTo(channels[1], CV_32F);
    channels[2].convertTo(channels[2], CV_32F);

    result = channels[0] * weights[0] + channels[1] * weights[1] + channels[2] * weights[2];

    result.forEach<float>([](float &pixel, const int *position) -> void
                          {
        if (pixel > 255.0f)
        {
            pixel = 255.0f;
        }
        else if (pixel < 0.0f)
        {
            pixel = 0.0f;
        } });

    result.convertTo(result, CV_8U);

    cv::merge(std::vector<cv::Mat>{result, result, result}, output);

    return 0;
}

// Apply a 5x5 pixel-wise process blur to an image
int blur5x5_1(cv::Mat &src, cv::Mat &dst)
{
    // Ensure the source image is of type CV_8UC3
    if (src.type() != CV_8UC3)
    {
        return -1;
    }

    dst = src.clone();

    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}};
    int kernelSum = 256; // Sum of all kernel values

    // Iterate over each pixel in the image, excluding the outer two rows and columns
    for (int y = 2; y < src.rows - 2; y++)
    {
        for (int x = 2; x < src.cols - 2; x++)
        {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int ky = -2; ky <= 2; ky++)
            {
                for (int kx = -2; kx <= 2; kx++)
                {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(y + ky, x + kx);
                    int kernelValue = kernel[ky + 2][kx + 2];

                    sumB += pixel[0] * kernelValue;
                    sumG += pixel[1] * kernelValue;
                    sumR += pixel[2] * kernelValue;
                }
            }

            dst.at<cv::Vec3b>(y, x)[0] = sumB / kernelSum;
            dst.at<cv::Vec3b>(y, x)[1] = sumG / kernelSum;
            dst.at<cv::Vec3b>(y, x)[2] = sumR / kernelSum;
        }
    }

    return 0;
}

// Apply a more efficient 5x5 blur to an image
int blur5x5_2(cv::Mat &src, cv::Mat &dst)
{
    // Ensure the source image is of type CV_8UC3
    if (src.type() != CV_8UC3)
    {
        return -1;
    }

    dst = src.clone();

    int kernel[5] = {1, 2, 4, 2, 1};
    int kernelSum = 16; // Sum of all kernel values

    cv::Mat temp = src.clone();

    // First pass: vertical filter
    for (int y = 2; y < src.rows - 2; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int ky = -2; ky <= 2; ky++)
            {
                cv::Vec3b pixel = src.ptr<cv::Vec3b>(y + ky)[x];
                int kernelValue = kernel[ky + 2];

                sumB += pixel[0] * kernelValue;
                sumG += pixel[1] * kernelValue;
                sumR += pixel[2] * kernelValue;
            }

            temp.ptr<cv::Vec3b>(y)[x][0] = sumB / kernelSum;
            temp.ptr<cv::Vec3b>(y)[x][1] = sumG / kernelSum;
            temp.ptr<cv::Vec3b>(y)[x][2] = sumR / kernelSum;
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
                cv::Vec3b pixel = temp.ptr<cv::Vec3b>(y)[x + kx];
                int kernelValue = kernel[kx + 2];

                sumB += pixel[0] * kernelValue;
                sumG += pixel[1] * kernelValue;
                sumR += pixel[2] * kernelValue;
            }

            dst.ptr<cv::Vec3b>(y)[x][0] = sumB / kernelSum;
            dst.ptr<cv::Vec3b>(y)[x][1] = sumG / kernelSum;
            dst.ptr<cv::Vec3b>(y)[x][2] = sumR / kernelSum;
        }
    }

    return 0;
}

// Apply a 3x3 Sobel filter in the X direction to an image
int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    // Ensure the source image is of type CV_8UC3
    if (src.type() != CV_8UC3)
    {
        return -1;
    }

    src.convertTo(dst, CV_16SC3);

    int kx[3] = {-1, 0, 1};
    int ky[3] = {1, 2, 1};

    cv::Mat temp = dst.clone();

    for (int y = 1; y < src.rows - 1; y++)
    {
        for (int x = 1; x < src.cols - 1; x++)
        {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int k = -1; k <= 1; k++)
            {
                cv::Vec3b pixel = src.ptr<cv::Vec3b>(y)[x + k];
                int kernelValue = kx[k + 1];

                sumB += pixel[0] * kernelValue;
                sumG += pixel[1] * kernelValue;
                sumR += pixel[2] * kernelValue;
            }

            temp.ptr<cv::Vec3s>(y)[x][0] = sumB;
            temp.ptr<cv::Vec3s>(y)[x][1] = sumG;
            temp.ptr<cv::Vec3s>(y)[x][2] = sumR;
        }
    }

    for (int y = 1; y < src.rows - 1; y++)
    {
        for (int x = 1; x < src.cols - 1; x++)
        {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int k = -1; k <= 1; k++)
            {
                cv::Vec3s pixel = temp.ptr<cv::Vec3s>(y + k)[x];
                int kernelValue = ky[k + 1];

                sumB += pixel[0] * kernelValue;
                sumG += pixel[1] * kernelValue;
                sumR += pixel[2] * kernelValue;
            }

            dst.ptr<cv::Vec3s>(y)[x][0] = sumB;
            dst.ptr<cv::Vec3s>(y)[x][1] = sumG;
            dst.ptr<cv::Vec3s>(y)[x][2] = sumR;
        }
    }

    return 0;
}

// Apply a 3x3 Sobel filter in the Y direction to an image
int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    // Ensure the source image is of type CV_8UC3
    if (src.type() != CV_8UC3)
    {
        return -1;
    }

    src.convertTo(dst, CV_16SC3);

    int kx[3] = {1, 2, 1};
    int ky[3] = {-1, 0, 1};

    cv::Mat temp = dst.clone();

    for (int y = 1; y < src.rows - 1; y++)
    {
        for (int x = 1; x < src.cols - 1; x++)
        {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int k = -1; k <= 1; k++)
            {
                cv::Vec3b pixel = src.ptr<cv::Vec3b>(y)[x + k];
                int kernelValue = kx[k + 1];

                sumB += pixel[0] * kernelValue;
                sumG += pixel[1] * kernelValue;
                sumR += pixel[2] * kernelValue;
            }

            temp.ptr<cv::Vec3s>(y)[x][0] = sumB;
            temp.ptr<cv::Vec3s>(y)[x][1] = sumG;
            temp.ptr<cv::Vec3s>(y)[x][2] = sumR;
        }
    }

    for (int y = 1; y < src.rows - 1; y++)
    {
        for (int x = 1; x < src.cols - 1; x++)
        {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int k = -1; k <= 1; k++)
            {
                cv::Vec3s pixel = temp.ptr<cv::Vec3s>(y + k)[x];
                int kernelValue = ky[k + 1];

                sumB += pixel[0] * kernelValue;
                sumG += pixel[1] * kernelValue;
                sumR += pixel[2] * kernelValue;
            }

            dst.ptr<cv::Vec3s>(y)[x][0] = sumB;
            dst.ptr<cv::Vec3s>(y)[x][1] = sumG;
            dst.ptr<cv::Vec3s>(y)[x][2] = sumR;
        }
    }

    return 0;
}

// Calculate the gradient magnitude of two images
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst)
{
    // Ensure the input images are of type CV_16SC3
    if (sx.type() != CV_16SC3 || sy.type() != CV_16SC3)
    {
        return -1;
    }

    dst.create(sx.size(), CV_8UC3);

    for (int y = 0; y < sx.rows; y++)
    {
        for (int x = 0; x < sx.cols; x++)
        {
            cv::Vec3s pixelX = sx.at<cv::Vec3s>(y, x);
            cv::Vec3s pixelY = sy.at<cv::Vec3s>(y, x);

            cv::Vec3b pixel;
            for (int c = 0; c < 3; c++)
            {
                int magnitude = std::sqrt(pixelX[c] * pixelX[c] + pixelY[c] * pixelY[c]);
                pixel[c] = cv::saturate_cast<uchar>(magnitude);
            }

            dst.at<cv::Vec3b>(y, x) = pixel;
        }
    }

    return 0;
}

// Apply a blur and quantize effect to an image
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels)
{
    // Blur the image
    cv::Mat blurred;
    cv::blur(src, blurred, cv::Size(5, 5));

    // Quantize the image
    int b = 255 / levels;
    dst = blurred.clone();
    for (int y = 0; y < dst.rows; y++)
    {
        for (int x = 0; x < dst.cols; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                int xt = dst.at<cv::Vec3b>(y, x)[c] / b;
                dst.at<cv::Vec3b>(y, x)[c] = xt * b;
            }
        }
    }
    return 0;
}

// Increase the brightness of an image
int bright(cv::Mat &src, cv::Mat &dst, int brightness = 50)
{
    if (src.empty())
    {
        return -1;
    }
    src.convertTo(dst, -1, 1, brightness);
    return 0;
}

// Apply an emboss effect to an image
int emboss(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
    {
        return -1;
    }

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // us cv::Sobel to compute the gradient in the x and y directions
    cv::Mat sobelX, sobelY;
    cv::Sobel(gray, sobelX, CV_32F, 1, 0);
    cv::Sobel(gray, sobelY, CV_32F, 0, 1);

    float directionX = 0.7071f;
    float directionY = 0.7071f;

    cv::Mat embossEffect = sobelX * directionX + sobelY * directionY;

    cv::normalize(embossEffect, embossEffect, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::cvtColor(embossEffect, dst, cv::COLOR_GRAY2BGR);

    return 0;
}

// Highlight the faces in an image with color
int colorfulFaces(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
    {
        return -1;
    }
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    detectFaces(grey, faces);

    dst = src.clone();

    cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
    cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);

    for (const auto &face : faces)
    {
        src(face).copyTo(dst(face));
    }

    return 0;
}

// Simulate the look of an old documentary film.
int oldDocumentary(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
    {
        return -1;
    }

    cv::Mat temp;

    if (dynamicFilmGrain(src, temp) != 0)
    {
        return -1;
    }

    if (vignette(temp, temp) != 0)
    {
        return -1;
    }

    if (filmFlicker(temp, dst) != 0)
    {
        return -1;
    }

    return 0;
}

// adding noise dynamically to create the impression of moving grain,
// similar to what is observed in old film footage.
int dynamicFilmGrain(cv::Mat &src, cv::Mat &dst, double intensity = 0.3)
{
    if (src.empty())
    {
        return -1;
    }

    cv::Mat noise = cv::Mat(src.size(), src.type());
    cv::randn(noise, 0, 50);

    dst = src.clone();
    cv::addWeighted(src, 1.0 - intensity, noise, intensity, 0, dst);

    return 0;
}

// A vignette darkens the corners of the frame,
// drawing attention to the center and mimicking lens artifacts in old cameras.
int vignette(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
    {
        return -1;
    }

    cv::Mat mask = cv::Mat::zeros(src.size(), CV_32F);

    cv::Point center = cv::Point(mask.cols / 2, mask.rows / 2);

    double maxDist = std::sqrt(center.x * center.x + center.y * center.y);

    for (int y = 0; y < mask.rows; y++)
    {
        for (int x = 0; x < mask.cols; x++)
        {
            double dist = std::sqrt((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y));
            mask.at<float>(y, x) = 1.0 - (dist / maxDist);
        }
    }

    cv::Mat mask3Channels;
    cv::merge(std::vector<cv::Mat>{mask, mask, mask}, mask3Channels);

    cv::Mat srcFloat;
    src.convertTo(srcFloat, CV_32F);
    cv::multiply(srcFloat, mask3Channels, srcFloat);
    srcFloat.convertTo(dst, src.type());

    return 0;
}

// Simulate the slight flickering of light intensity seen in old film footage.
int filmFlicker(cv::Mat &src, cv::Mat &dst, double intensity = 0.1)
{
    if (src.empty())
    {
        return -1;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0 - intensity, 1.0 + intensity);
    double brightnessFactor = dis(gen);

    src.convertTo(dst, -1, brightnessFactor, 0);

    return 0;
}
