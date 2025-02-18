/*
  Junjie Li
  Spring 2025

  Main function and keyboard input control of how to process a video.

  Program takes a path to a video on the command line
*/

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "or.h"
#include <ctime>

using namespace cv;
using namespace std;

// Define the database file path
static const string dbFilePath = "C:/Users/alvin/Desktop/takkugit/projects/project3/build/Debug/object_db.csv";

// Function to process keyboard commands
void handleKeyboardInput(char key, Mat &frame, string &currentMode, bool &paused, string &label, bool &recording, VideoWriter &videoWriter)
{
    if (key == 'q')
    {
        // Quit the application
        cout << "Exiting..." << endl;
        exit(0);
    }
    else if (key == 's')
    {
        // Pause the video
        paused = true;

        auto t = time(nullptr);
        auto tm = *localtime(&t);

        char buffer[80];
        strftime(buffer, sizeof(buffer), "captured_frame_%Y%m%d_%H%M%S.jpg", &tm);
        string filename(buffer);

        Mat processedFrame = processFrame(frame, currentMode);

        // Save the processed image with the unique filename
        imwrite(filename, processedFrame);
        cout << "Image saved to " << filename << endl;
    }
    else if (key == 'r')
    {
        // Resume the video
        paused = false;
        cout << "Resumed video playback" << endl;
    }
    else if (key == 'd' && currentMode != "depthEstimation")
    {
        currentMode = "depthEstimation";
        cout << "Switched to depth estimation" << endl;
    }
    else if (key == '0' && currentMode != "original")
    {
        currentMode = "original";
        cout << "Switched to original video" << endl;
    }
    else if (key == '1' && currentMode != "threshold")
    {
        currentMode = "threshold";
        cout << "Switched to thresholded video" << endl;
    }
    else if (key == '2' && currentMode != "threshold_with_clean")
    {
        currentMode = "threshold_with_clean";
        cout << "Switched to cleaned up thresholded video" << endl;
    }
    else if (key == '3' && currentMode != "analyze_and_display_regions")
    {
        currentMode = "analyze_and_display_regions";
        cout << "Switched to segmented video" << endl;
    }
    else if (key == '4' && currentMode != "computeFeaturesForRegions")
    {
        currentMode = "computeFeaturesForRegions";
        cout << "Switched to computeFeaturesForRegions video" << endl;
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
    else if (key == 'v')
    {
        // Toggle video recording
        if (!recording)
        {
            auto t = time(nullptr);
            auto tm = *localtime(&t);

            char buffer[80];
            strftime(buffer, sizeof(buffer), "recorded_video_%Y%m%d_%H%M%S.avi", &tm);
            string filename(buffer);

            // Define the codec and create VideoWriter object
            int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
            double fps = 30.0;
            videoWriter.open(filename, codec, fps, frame.size(), true);

            if (!videoWriter.isOpened())
            {
                cout << "Error: Could not open the video file for writing." << endl;
            }
            else
            {
                recording = true;
                cout << "Started recording video to " << filename << endl;
            }
        }
        else
        {
            recording = false;
            videoWriter.release();
            cout << "Stopped recording video" << endl;
        }
    }
}

int main(int argc, char *argv[])
{
    VideoCapture cap(0); // Open the default webcam
    if (!cap.isOpened())
    {
        cout << "Error: Could not open webcam." << endl;
        return -1;
    }

    Mat frame;
    string currentMode = "original";
    string label;
    char key = 0;
    bool paused = false;
    bool recording = false;
    VideoWriter videoWriter;

    const char *windowName = "Video";
    namedWindow(windowName, WINDOW_AUTOSIZE);

    // Main loop
    while (true)
    {
        // Check if window was closed
        if (getWindowProperty(windowName, WND_PROP_VISIBLE) < 1)
        {
            cout << "Window closed by user" << endl;
            break;
        }

        if (!paused)
        {
            cap >> frame;

            if (frame.empty())
            {
                cout << "Error: Could not capture frame." << endl;
                break;
            }
        }

        // Process frame based on current mode
        Mat processedFrame;
        if (currentMode == "classifyImages")
        {
            classifyImages(frame, processedFrame);
        }
        else
        {
            processedFrame = processFrame(frame, currentMode);
        }
        imshow(windowName, processedFrame);

        // Write the frame to the video file if recording
        if (recording)
        {
            videoWriter.write(processedFrame);
        }

        key = waitKey(30);
        handleKeyboardInput(key, frame, currentMode, paused, label, recording, videoWriter);
    }

    // Release the video writer if recording
    if (recording)
    {
        videoWriter.release();
    }

    return 0;
}