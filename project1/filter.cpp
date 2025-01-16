#include "filter.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

int greyscale(cv::Mat &src, cv::Mat &dst)
{
    // Check if image is empty or has wrong format
    if (src.empty() || src.channels() != 3)
    {
        return -1;
    }

    // Create a vector of the weights (blue, green, red)
    std::vector<double> weights = {0.4, 0.45, 0.15};

    // Create array of matrices to hold each channel
    std::vector<cv::Mat> channels(3);
    cv::split(src, channels);

    // Convert to float for better precision during calculations
    cv::Mat result;
    channels[0].convertTo(channels[0], CV_32F);
    channels[1].convertTo(channels[1], CV_32F);
    channels[2].convertTo(channels[2], CV_32F);

    // Combine channels with weights
    result = channels[0] * weights[0] + channels[1] * weights[1] + channels[2] * weights[2];

    // Apply the darkness factor (0.9)
    result *= 0.9;

    // Convert back to 8-bit unsigned
    result.convertTo(result, CV_8U);

    // Create 3-channel output
    cv::merge(std::vector<cv::Mat>{result, result, result}, dst);

    return 0;
}

int blur5x5_1(cv::Mat &src, cv::Mat &dst)
{
    // Ensure the source image is of type CV_8UC3
    if (src.type() != CV_8UC3)
    {
        return -1;
    }

    // Copy the source image to the destination image
    dst = src.clone();

    // Define the Gaussian kernel
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
            // Initialize the sum for each color channel
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply the kernel to each color channel
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

            // Normalize the sum and assign it to the destination image
            dst.at<cv::Vec3b>(y, x)[0] = sumB / kernelSum;
            dst.at<cv::Vec3b>(y, x)[1] = sumG / kernelSum;
            dst.at<cv::Vec3b>(y, x)[2] = sumR / kernelSum;
        }
    }

    return 0;
}

cv::Mat processFrame(cv::Mat &input, const std::string &mode)
{
    cv::Mat output;

    if (mode == "opencv_gray")
    {
        cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
        cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
    }
    else if (mode == "custom_gray")
    {
        if (greyscale(input, output) != 0)
        {
            std::cout << "Error applying custom grayscale filter" << std::endl;
            output = input.clone();
        }
    }
    else if (mode == "sofia")
    {
        cv::Mat sofiaKernel = (cv::Mat_<float>(3, 3) << 0.272, 0.534, 0.131,
                               0.349, 0.686, 0.168,
                               0.393, 0.769, 0.189);
        cv::Mat temp;
        cv::transform(input, temp, sofiaKernel);
        temp.forEach<cv::Vec3f>([](cv::Vec3f &pixel, const int *position) -> void
                                {
            for (int i = 0; i < 3; i++) {
                if (pixel[i] > 255.0f) {
                    pixel[i] = 255.0f;
                } else if (pixel[i] < 0.0f) {
                    pixel[i] = 0.0f;
                }
            } });

        temp.convertTo(output, CV_8UC3);
    }
    else
    { // color mode
        output = input.clone();
    }

    return output;
}