## Name: Junjie Li

## Links to a demo video for Task 8

- [Demo](https://drive.google.com/file/d/1d0qKONSDwuUSr8JUWpvIH29XqcLGSfow/view?usp=sharing)

## Env: Win11, Visual Studio 2022, Cmake

## Instructions for running executables.

- First you may modify dbFilePath var, which is a string of full-path-csv-file that store feature vectors

- Then You may run "vidDisplay.exe" directly for streaming, or

- run "imgDisplay.exe + full-path-of-image-file" for static images

## Instructions for testing any extensions you completed.

- Two tasks (1 and 3) are implemented from scratch, which is my extension.

- For task 1, check these functions in or.cpp:

  - int greyscale(Mat &src, Mat &dst);
  - int blur5x5_2(Mat &src, Mat &dst);
  - inline double euclideanDist(double a, double b);
  - void kmeansCustom(vector<float> &samples, int K, vector<float> &centers, vector<int> &labels, int maxIterations = 10);
  - void convertBGRtoHSV(const Mat &src, Mat &dst);
  - int threshold(Mat &src, Mat &dst);

- For task 3, check these functions in or.cpp:

  - int connectedComponentsWithStats(const Mat &binary, Mat &labels, Mat &stats, Mat &centroids)
  - int analyze_and_display_regions(Mat &src, Mat &dst)
