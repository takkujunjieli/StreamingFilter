#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>

// Constants
const cv::Size patternSize(9, 6); // Internal checkerboard corners
const cv::Size subPixWinSize(11, 11);
const cv::Size zeroZone(-1, -1);
const cv::TermCriteria termCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01);

// Storage for calibration data
std::vector<std::vector<cv::Vec3f>> point_list;
std::vector<std::vector<cv::Point2f>> corner_list;
int imageCount = 0;

/**
 * Detects checkerboard corners in the input frame.
 * @param frame Input image
 * @param corner_set Vector to store detected corners
 * @return true if corners were found, false otherwise
 */
bool detectCheckerboardCorners(const cv::Mat &frame, std::vector<cv::Point2f> &corner_set)
{
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    bool found = cv::findChessboardCorners(gray, patternSize, corner_set);
    if (found)
    {
        cv::cornerSubPix(gray, corner_set, subPixWinSize, zeroZone, termCriteria);
    }

    return found;
}

/**
 * Runs camera calibration using collected corner and point sets.
 * @param imageSize Size of the calibration images.
 * @return The final reprojection error.
 */
double calibrateCamera(const cv::Size &imageSize)
{
    if (corner_list.size() < 5)
    {
        std::cout << "Not enough calibration images (need at least 5).\n";
        return -1.0;
    }

    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1, 0, imageSize.width / 2,
                             0, 1, imageSize.height / 2,
                             0, 0, 1);
    cv::Mat distortion_coeffs = cv::Mat::zeros(5, 1, CV_64F); // No distortion initially
    std::vector<cv::Mat> rvecs, tvecs;

    double error = cv::calibrateCamera(point_list, corner_list, imageSize,
                                       camera_matrix, distortion_coeffs,
                                       rvecs, tvecs,
                                       cv::CALIB_FIX_ASPECT_RATIO);

    std::cout << "Final Reprojection Error: " << error << "\n";
    std::cout << "Camera Matrix:\n"
              << camera_matrix << "\n";
    std::cout << "Distortion Coefficients:\n"
              << distortion_coeffs << "\n";

    // Ensure the matrices are correctly sized
    if (camera_matrix.rows != 3 || camera_matrix.cols != 3)
    {
        std::cerr << "Error: Camera matrix has incorrect size.\n";
        return -1.0;
    }
    if (distortion_coeffs.rows != 5 || distortion_coeffs.cols != 1)
    {
        std::cerr << "Error: Distortion coefficients matrix has incorrect size.\n";
        return -1.0;
    }

    // Save intrinsic parameters and extrinsic parameters to a file
    cv::FileStorage fs("camera_calibration.yaml", cv::FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "distortion_coefficients" << distortion_coeffs;
    fs << "rotation_vectors" << rvecs;
    fs << "translation_vectors" << tvecs;
    fs.release();
    std::cout << "Calibration parameters saved to 'camera_calibration.yaml'.\n";

    // Visualize camera locations relative to the target
    for (size_t i = 0; i < rvecs.size(); ++i)
    {
        std::vector<cv::Point3f> axisPoints;
        axisPoints.push_back(cv::Point3f(0, 0, 0));
        axisPoints.push_back(cv::Point3f(3, 0, 0));
        axisPoints.push_back(cv::Point3f(0, 3, 0));
        axisPoints.push_back(cv::Point3f(0, 0, -3));

        std::vector<cv::Point2f> imagePoints;
        cv::projectPoints(axisPoints, rvecs[i], tvecs[i], camera_matrix, distortion_coeffs, imagePoints);

        cv::Mat frame = cv::imread("calib_image_" + std::to_string(i) + ".jpg");
        if (!frame.empty())
        {
            cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2);
            cv::line(frame, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 2);
            cv::line(frame, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 2);
            cv::imshow("Camera Pose", frame);
            cv::waitKey(500);
        }
    }

    return error;
}

/**
 * Generates 3D world coordinates corresponding to the checkerboard pattern.
 * @return Vector of 3D world points
 */
std::vector<cv::Vec3f> generate3DWorldPoints()
{
    std::vector<cv::Vec3f> point_set;
    for (int row = 0; row < patternSize.height; ++row)
    {
        for (int col = 0; col < patternSize.width; ++col)
        {
            point_set.push_back(cv::Vec3f(col, -row, 0)); // 1 unit per square
        }
    }
    return point_set;
}

/**
 * Saves detected checkerboard corners and corresponding 3D world points.
 * @param frame The current frame
 * @param corner_set The detected 2D corner points
 * @param imageSize Size of the calibration images.
 */
void saveCalibrationData(const cv::Mat &frame, const std::vector<cv::Point2f> &corner_set, const cv::Size &imageSize)
{
    corner_list.push_back(corner_set);
    point_list.push_back(generate3DWorldPoints());

    std::string filename = "calib_image_" + std::to_string(imageCount++) + ".jpg";
    cv::imwrite(filename, frame);
    std::cout << "Saved calibration image: " << filename << std::endl;

    if (corner_list.size() >= 5)
    {
        double error = calibrateCamera(imageSize);
        if (error >= 0)
        {
            std::cout << "Current Per Pixel Error: " << error << std::endl;
        }
    }
}

int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Couldn't open camera.\n";
        return -1;
    }
    cv::Size imageSize;
    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        std::vector<cv::Point2f> corner_set;
        imageSize = frame.size();
        bool found = detectCheckerboardCorners(frame, corner_set);

        if (found)
        {
            cv::drawChessboardCorners(frame, patternSize, corner_set, found);
        }

        cv::imshow("Checkerboard Detection", frame);

        char key = cv::waitKey(1);
        if (cv::waitKey(1) == 27 || cv::getWindowProperty("Checkerboard Detection", cv::WND_PROP_VISIBLE) < 1)
            break;
        else if (key == 's' && found)
        {
            saveCalibrationData(frame, corner_set, imageSize);
        }
        else if (key == 'c')
        {
            double error = calibrateCamera(imageSize);
            if (error >= 0)
            {
                std::cout << "Current Per Pixel Error: " << error << std::endl;
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
