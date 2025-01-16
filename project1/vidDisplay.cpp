#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "filter.h"

// Function to process keyboard commands
void handleKeyboardInput(char key, cv::Mat &frame, std::string &currentMode)
{
    if (key == 'q')
    {
        std::cout << "Quitting..." << std::endl;
        exit(0);
    }
    else if (key == 's')
    {
        cv::imwrite("captured_frame.jpg", frame);
        std::cout << "Image saved to captured_frame.jpg" << std::endl;
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
    else if ((key == 'g' || key == 'h') && currentMode != "color")
    {
        currentMode = "color";
        std::cout << "Switched to color" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    // Initialize camera
    cv::VideoCapture camera(0);
    if (!camera.open(0))
    {
        std::cout << "Error: Could not open camera." << std::endl;
        return -1;
    }

    const char *windowName = "Video";
    // Create display window
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // Track current display mode
    std::string currentMode = "color";

    // Main loop
    while (true)
    {
        // Check if window was closed
        if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1)
        {
            std::cout << "Window closed by user" << std::endl;
            break;
        }

        // Capture frame
        cv::Mat frame;
        camera >> frame;

        if (frame.empty())
        {
            std::cout << "Error: Blank frame grabbed" << std::endl;
            break;
        }

        // Process frame based on current mode
        cv::Mat processedFrame = processFrame(frame, currentMode);

        // Display the frame
        cv::imshow(windowName, processedFrame);

        // Handle keyboard input (wait 10ms for key press)
        char key = cv::waitKey(10);
        handleKeyboardInput(key, processedFrame, currentMode);
    }

    // Cleanup
    camera.release();
    cv::destroyAllWindows();
    return 0;
}