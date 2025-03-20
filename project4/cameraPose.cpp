#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

// Constants
const cv::Size patternSize(9, 6); // Internal checkerboard corners
const cv::Size subPixWinSize(11, 11);
const cv::Size zeroZone(-1, -1);
const cv::TermCriteria termCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01);

/**
 * Reads camera calibration parameters from a file.
 * @param filename The name of the file containing the calibration parameters.
 * @param cameraMatrix The camera matrix to be filled.
 * @param distCoeffs The distortion coefficients to be filled.
 * @param rvecs Vector of rotation vectors to be filled.
 * @param tvecs Vector of translation vectors to be filled.
 * @return true if the parameters were successfully read, false otherwise.
 */
bool readCameraParameters(const std::string &filename, cv::Mat &cameraMatrix, cv::Mat &distCoeffs, std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Error: Could not open calibration file.\n";
        return false;
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs["rotation_vectors"] >> rvecs;
    fs["translation_vectors"] >> tvecs;
    fs.release();

    // Ensure the matrices are correctly sized
    if (cameraMatrix.rows != 3 || cameraMatrix.cols != 3)
    {
        std::cerr << "Error: Camera matrix has incorrect size.\n";
        return false;
    }
    if (distCoeffs.rows != 5 || distCoeffs.cols != 1)
    {
        std::cerr << "Error: Distortion coefficients matrix has incorrect size.\n";
        return false;
    }

    return true;
}

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

// Function to draw a shaded pyramid
void drawShadedPyramid(cv::Mat &frame, const std::vector<cv::Point2f> &pyramidImagePoints)
{
    // Define colors for the pyramid faces
    cv::Scalar baseColor(0, 255, 255); // Yellow
    cv::Scalar sideColor(0, 255, 0);   // Green

    // Draw base
    cv::line(frame, pyramidImagePoints[0], pyramidImagePoints[1], baseColor, 2);
    cv::line(frame, pyramidImagePoints[2], pyramidImagePoints[3], baseColor, 2);
    cv::line(frame, pyramidImagePoints[4], pyramidImagePoints[5], baseColor, 2);
    cv::line(frame, pyramidImagePoints[6], pyramidImagePoints[7], baseColor, 2);

    // Draw sides
    cv::line(frame, pyramidImagePoints[8], pyramidImagePoints[9], sideColor, 2);
    cv::line(frame, pyramidImagePoints[10], pyramidImagePoints[11], sideColor, 2);
    cv::line(frame, pyramidImagePoints[12], pyramidImagePoints[13], sideColor, 2);
    cv::line(frame, pyramidImagePoints[14], pyramidImagePoints[15], sideColor, 2);

    // Draw mid-section
    cv::line(frame, pyramidImagePoints[16], pyramidImagePoints[17], sideColor, 2);
    cv::line(frame, pyramidImagePoints[18], pyramidImagePoints[19], sideColor, 2);
    cv::line(frame, pyramidImagePoints[20], pyramidImagePoints[21], sideColor, 2);
    cv::line(frame, pyramidImagePoints[22], pyramidImagePoints[23], sideColor, 2);

    // Draw mid-section to top
    cv::line(frame, pyramidImagePoints[24], pyramidImagePoints[25], sideColor, 2);
    cv::line(frame, pyramidImagePoints[26], pyramidImagePoints[27], sideColor, 2);
    cv::line(frame, pyramidImagePoints[28], pyramidImagePoints[29], sideColor, 2);
    cv::line(frame, pyramidImagePoints[30], pyramidImagePoints[31], sideColor, 2);
}

int main(int argc, char **argv)
{
    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    if (!readCameraParameters("camera_calibration.yaml", cameraMatrix, distCoeffs, rvecs, tvecs))
    {
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Couldn't open camera.\n";
        return -1;
    }

    std::vector<cv::Vec3f> objectPoints = generate3DWorldPoints();

    // Define 3D points for the axes
    std::vector<cv::Vec3f> axesPoints = {
        {0, 0, 0}, {3, 0, 0}, {0, -3, 0}, {0, 0, 3}};

    // Define 3D points for the corners of the checkerboard
    std::vector<cv::Vec3f> cornerPoints = {
        {0.0f, 0.0f, 1.0f}, {static_cast<float>(patternSize.width - 1), 0.0f, 1.0f}, {0.0f, static_cast<float>(-patternSize.height + 1), 1.0f}, {static_cast<float>(patternSize.width - 1), static_cast<float>(-patternSize.height + 1), 1.0f}};

    cv::Vec3f boardCenter(4.0, -2.5, 0); // Center of chessboard

    // Define 3D points for the pyramid structure
    std::vector<cv::Vec3f> pyramidPoints = {
        // Base Square
        {boardCenter[0] - 2, boardCenter[1] - 2, 0},
        {boardCenter[0] + 2, boardCenter[1] - 2, 0},
        {boardCenter[0] + 2, boardCenter[1] - 2, 0},
        {boardCenter[0] + 2, boardCenter[1] + 2, 0},
        {boardCenter[0] + 2, boardCenter[1] + 2, 0},
        {boardCenter[0] - 2, boardCenter[1] + 2, 0},
        {boardCenter[0] - 2, boardCenter[1] + 2, 0},
        {boardCenter[0] - 2, boardCenter[1] - 2, 0},

        // Base to Apex
        {boardCenter[0] - 2, boardCenter[1] - 2, 0},
        {boardCenter[0], boardCenter[1], 4},
        {boardCenter[0] + 2, boardCenter[1] - 2, 0},
        {boardCenter[0], boardCenter[1], 4},
        {boardCenter[0] + 2, boardCenter[1] + 2, 0},
        {boardCenter[0], boardCenter[1], 4},
        {boardCenter[0] - 2, boardCenter[1] + 2, 0},
        {boardCenter[0] - 1, boardCenter[1] + 1, 2},

        // Mid Section Square
        {boardCenter[0] - 1, boardCenter[1] - 1, 2},
        {boardCenter[0] + 1, boardCenter[1] - 1, 2},
        {boardCenter[0] + 1, boardCenter[1] - 1, 2},
        {boardCenter[0] + 1, boardCenter[1] + 1, 2},
        {boardCenter[0] + 1, boardCenter[1] + 1, 2},
        {boardCenter[0] - 1, boardCenter[1] + 1, 2},
        {boardCenter[0] - 1, boardCenter[1] + 1, 2},
        {boardCenter[0] - 1, boardCenter[1] - 1, 2},

        // Mid Section to Top
        {boardCenter[0] - 1, boardCenter[1] - 1, 2},
        {boardCenter[0], boardCenter[1], 4},
        {boardCenter[0] + 1, boardCenter[1] - 1, 2},
        {boardCenter[0], boardCenter[1], 4},
        {boardCenter[0] + 1, boardCenter[1] + 1, 2},
        {boardCenter[0], boardCenter[1], 4},
        {boardCenter[0] - 1, boardCenter[1] + 1, 2},
        {boardCenter[0], boardCenter[1], 4},
    };

    // Load sand texture
    cv::Mat sandTexture = cv::imread("sand_texture.jfif");
    if (sandTexture.empty())
    {
        std::cerr << "Error: Could not load sand texture image.\n";
        return -1;
    }

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        std::vector<cv::Point2f> corner_set;
        bool found = detectCheckerboardCorners(frame, corner_set);

        if (found)
        {
            cv::Vec3d rvec, tvec;
            cv::solvePnP(objectPoints, corner_set, cameraMatrix, distCoeffs, rvec, tvec);

            // Project 3D points to the image plane
            std::vector<cv::Point2f> imagePoints;
            cv::projectPoints(axesPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

            // Draw the 3D axes
            cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2); // X-axis in red
            cv::line(frame, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 2); // Y-axis in green
            cv::line(frame, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 2); // Z-axis in blue

            // Project corner points to the image plane
            std::vector<cv::Point2f> cornerImagePoints;
            cv::projectPoints(cornerPoints, rvec, tvec, cameraMatrix, distCoeffs, cornerImagePoints);

            // Draw the projected corners
            for (const auto &pt : cornerImagePoints)
            {
                cv::circle(frame, pt, 5, cv::Scalar(0, 255, 255), -1); // Yellow circles
            }

            // Draw the chessboard grid with sand texture
            for (int i = 0; i < patternSize.height - 1; ++i)
            {
                for (int j = 0; j < patternSize.width - 1; ++j)
                {
                    cv::Point2f p1 = corner_set[i * patternSize.width + j];
                    cv::Point2f p2 = corner_set[i * patternSize.width + j + 1];
                    cv::Point2f p3 = corner_set[(i + 1) * patternSize.width + j + 1];
                    cv::Point2f p4 = corner_set[(i + 1) * patternSize.width + j];

                    std::vector<cv::Point2f> quad = {p1, p2, p3, p4};
                    std::vector<cv::Point2f> textureQuad = {cv::Point2f(0, 0), cv::Point2f(sandTexture.cols, 0), cv::Point2f(sandTexture.cols, sandTexture.rows), cv::Point2f(0, sandTexture.rows)};

                    cv::Mat H = cv::getPerspectiveTransform(textureQuad, quad);
                    cv::warpPerspective(sandTexture, frame, H, frame.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
                }
            }

            // Project 3D pyramid to 2D image space
            std::vector<cv::Point2f> pyramidImagePoints;
            cv::projectPoints(pyramidPoints, rvec, tvec, cameraMatrix, distCoeffs, pyramidImagePoints);

            // Draw the shaded pyramid
            drawShadedPyramid(frame, pyramidImagePoints);

            std::cout << "Rotation Vector: " << rvec << "\n";
            std::cout << "Translation Vector: " << tvec << "\n";
        }

        cv::imshow("Pose Estimation", frame);

        char key = cv::waitKey(1);
        if (key == 27 || cv::getWindowProperty("Pose Estimation", cv::WND_PROP_VISIBLE) < 1)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}