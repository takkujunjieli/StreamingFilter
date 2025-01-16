#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // Read the image file
    cv::Mat image = cv::imread("example.jpg");

    // Check if image was successfully loaded
    if (image.empty())
    {
        std::cout << "Error: Could not load image" << std::endl;
        return -1;
    }

    // Create a window to display the image
    cv::namedWindow("Image Display", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image Display", image);

    // Main loop for key detection
    while (true)
    {
        char key = (char)cv::waitKey(10);
        if (key == 'q' || key == 'Q')
        {
            std::cout << "Quitting program..." << std::endl;
            break;
        }
        else if (key == 'f' || key == 'F')
        {
            // Flip image horizontally
            cv::flip(image, image, 1);
            cv::imshow("Image Display", image);
        }
    }

    // Clean up
    cv::destroyAllWindows();
    return 0;
}