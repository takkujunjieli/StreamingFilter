/*
  Junjie Li
  Spring 2025

  Main function and keyboard input control of how to process an video.

  Program takes a path to an video on the command line
*/

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "filter.h"
#include "faceDetect.h"
#include <ctime>

// Function to process keyboard commands
void handleKeyboardInput(char key, cv::Mat &frame, std::string &currentMode, cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &displayX, cv::Mat &displayY, cv::Mat &magnitudeImage, bool &paused)
{
    if (key == 'q')
    {
        // Quit the application
        std::cout << "Exiting..." << std::endl;
        exit(0);
    }
    else if (key == 's')
    {
        // Pause the video
        paused = true;

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);

        char buffer[80];
        strftime(buffer, sizeof(buffer), "captured_frame_%Y%m%d_%H%M%S.jpg", &tm);
        std::string filename(buffer);

        cv::Mat processedFrame = processFrame(frame, currentMode, sobelX, sobelY, magnitudeImage);

        // Save the processed image with the unique filename
        cv::imwrite(filename, processedFrame);
        std::cout << "Image saved to " << filename << std::endl;
    }
    else if (key == 'r')
    {
        // Resume the video
        paused = false;
        std::cout << "Resumed video playback" << std::endl;
    }
    else if (key == 'g' && currentMode != "opencv_gray")
    {
        currentMode = "opencv_gray";
        std::cout << "Switched to OpenCV grayscale" << std::endl;
    }
    else if (key == 'h' && currentMode != "custom_gray")
    {
        currentMode = "custom_gray";
        std::cout << "Switched to custom grayscale" << std::endl;
    }
    else if (key == '1' && currentMode != "sepia")
    {
        currentMode = "sepia";
        std::cout << "Switched to sepia" << std::endl;
    }
    else if (key == 'b' && currentMode != "blur")
    {
        currentMode = "blur";
        std::cout << "Switched to blur" << std::endl;
    }
    else if (key == 'x' && currentMode != "sobelX3x3")
    {
        currentMode = "sobelX3x3";
        std::cout << "Switched to Sobel X" << std::endl;
    }
    else if (key == 'y' && currentMode != "sobelY3x3")
    {
        currentMode = "sobelY3x3";
        std::cout << "Switched to Sobel Y" << std::endl;
    }
    else if (key == 'm' && currentMode != "magnitude")
    {

        currentMode = "magnitude";
        std::cout << "Switched to gradient magnitude" << std::endl;
    }
    else if (key == 'l' && currentMode != "blurQuantize")
    {
        currentMode = "blurQuantize";
        std::cout << "Switched to blur and quantize" << std::endl;
    }
    else if (key == 'f' && currentMode != "faceDetect")
    {
        currentMode = "faceDetect";
        std::cout << "Switched to face detection" << std::endl;
    }
    else if (key == '7' && currentMode != "brightness")
    {
        currentMode = "brightness";
        std::cout << "Switched to more brightness" << std::endl;
    }
    else if (key == '8' && currentMode != "emboss")
    {
        currentMode = "emboss";
        std::cout << "Switched to embossing effect" << std::endl;
    }
    else if (key == '9' && currentMode != "colorfulFaces")
    {
        currentMode = "colorfulFaces";
        std::cout << "Switched to colorful faces" << std::endl;
    }
    else if (key == '0' && currentMode != "oldDocumentary")
    {
        currentMode = "oldDocumentary";
        std::cout << "Switched to colorful faces" << std::endl;
    }

    else if (key == 'c' && currentMode != "color")
    {
        currentMode = "color";
        std::cout << "Switched to color" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened())
    {
        std::cout << "Error: Could not open video." << std::endl;
        return -1;
    }

    cv::Mat frame, sobelX, sobelY, displayX, displayY, magnitudeImage;
    std::string currentMode = "color";
    char key = 0;
    bool paused = false;

    const char *windowName = "Video";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // Main loop
    while (true)
    {
        // Check if window was closed
        if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1)
        {
            std::cout << "Window closed by user" << std::endl;
            break;
        }

        if (!paused)
        {
            cap >> frame;

            if (frame.empty())
            {
                // Restart the video if it reaches the end
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                cap >> frame;
            }
        }

        // Process frame based on current mode
        cv::Mat processedFrame = processFrame(frame, currentMode, sobelX, sobelY, magnitudeImage);
        cv::imshow(windowName, processedFrame);

        key = cv::waitKey(30);
        handleKeyboardInput(key, frame, currentMode, sobelX, sobelY, displayX, displayY, magnitudeImage, paused);
    }

    return 0;
}