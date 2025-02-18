/*
  Junjie Li
  Spring 2025

  Main function and keyboard input control of how to process an image.

  Program takes a path to an image on the command line
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "or.h"
#include <ctime>

static const string dbFilePath = "C:/Users/alvin/Desktop/takkugit/projects/project3/build/Debug/object_db.csv";

// Function to process keyboard commands
void handleKeyboardInput(char key, cv::Mat &image, std::string &currentMode, std::string &label)
{
    if (key == 'q')
    {
        // Quit the application
        std::cout << "Exiting..." << std::endl;
        exit(0);
    }
    else if (key == 's')
    {
        // Save the processed image
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);

        char buffer[80];
        strftime(buffer, sizeof(buffer), "captured_image_%Y%m%d_%H%M%S.jpg", &tm);
        std::string filename(buffer);

        cv::Mat processedImage = processFrame(image, currentMode);

        // Save the processed image with the unique filename
        cv::imwrite(filename, processedImage);
        std::cout << "Image saved to " << filename << std::endl;
    }
    else if (key == 'd' && currentMode != "depthEstimation")
    {
        currentMode = "depthEstimation";
        cout << "Switched to depth estimation" << endl;
    }
    else if (key == '0' && currentMode != "original")
    {
        currentMode = "original";
        cout << "Switched to original image" << endl;
    }
    else if (key == '1' && currentMode != "threshold")
    {
        currentMode = "threshold";
        cout << "Switched to thresholded image" << endl;
    }
    else if (key == '2' && currentMode != "threshold_with_clean")
    {
        currentMode = "threshold_with_clean";
        cout << "Switched to cleaned and thresholded image" << endl;
    }
    else if (key == '3' && currentMode != "analyze_and_display_regions")
    {
        currentMode = "analyze_and_display_regions";
        cout << "Switched to segmented image" << endl;
    }
    else if (key == '4' && currentMode != "computeFeaturesForRegions")
    {
        currentMode = "computeFeaturesForRegions";
        cout << "Switched to computeFeaturesForRegions image" << endl;
    }
    else if (key == '5' && currentMode != "collectAndStoreFeatures")
    {
        currentMode = "collectAndStoreFeatures";
        cout << "Switched to collectAndStoreFeatures mode" << endl;

        // Prompt the user for a label
        cout << "Enter label for the current object: ";
        cin >> label;
    }
    else if (key == '6' && currentMode != "classifyImages")
    {
        currentMode = "classifyImages";
        cout << "Switched to classifyImages mode" << endl;
    }
    else
    {
        std::cout << "Invalid key pressed!" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    Mat image = imread(imagePath);

    if (image.empty())
    {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    std::string windowName = "Image Display";
    namedWindow(windowName, WINDOW_AUTOSIZE);

    std::string currentMode = "original";
    std::string label;
    Mat processedImage = processFrame(image, currentMode);
    imshow(windowName, processedImage);

    bool debugOutputPrinted = false;
    bool collectAndStoreCalled = false;

    while (true)
    {
        char key = (char)waitKey(0);
        if (key == 'q')
        {
            std::cout << "Exiting..." << std::endl;
            break;
        }

        handleKeyboardInput(key, image, currentMode, label);

        if (currentMode == "collectAndStoreFeatures" && !collectAndStoreCalled)
        {
            collectAndStoreFeatures(image, label, dbFilePath);
            collectAndStoreCalled = true;
            processedImage = image.clone();
        }
        else
        {
            processedImage = processFrame(image, currentMode);
            collectAndStoreCalled = false; // Reset the flag for other modes
        }

        if (getWindowProperty(windowName, WND_PROP_VISIBLE) < 1)
        {
            break;
        }

        imshow(windowName, processedImage);

        if (currentMode == "computeFeaturesForRegions" && !debugOutputPrinted)
        {
            // Print debug output only once
            debugOutputPrinted = true;
        }
    }

    destroyAllWindows();
    return 0;
}