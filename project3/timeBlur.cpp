/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Example of how to time an image processing task.

  Program takes a path to an image on the command line
*/

#include <cstdio>  // a bunch of standard C/C++ functions like printf, scanf
#include <cstring> // C/C++ functions for working with strings
#include <cmath>
#ifdef _WIN32
#include <chrono>
#else
#include <sys/time.h>
#endif
#include "opencv2/opencv.hpp"

// prototypes for the functions to test
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

// returns a double which gives time in seconds
double getTime()
{
#ifdef _WIN32
  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  return elapsed.count();
#else
  struct timeval cur;
  gettimeofday(&cur, NULL);
  return (cur.tv_sec + cur.tv_usec / 1000000.0);
#endif
}

// argc is # of command line parameters (including program name), argv is the array of strings
// This executable is expecting the name of an image on the command line.

int main(int argc, char *argv[])
{                     // main function, execution starts here
  cv::Mat src;        // define a Mat data type (matrix/image), allocates a header, image data is null
  cv::Mat dst;        // cv::Mat to hold the output of the process
  char filename[256]; // a string for the filename

  // usage: checking if the user provided a filename
  if (argc < 2)
  {
    printf("Usage %s <image filename>\n", argv[0]);
    exit(-1);
  }
  strcpy(filename, argv[1]); // copying 2nd command line argument to filename variable

  // read the image
  src = cv::imread(filename); // allocating the image data
  // test if the read was successful
  if (src.data == NULL)
  { // src.data is the reference to the image data
    printf("Unable to read image %s\n", filename);
    exit(-1);
  }

  const int Ntimes = 10;

  //////////////////////////////
  // set up the timing for version 1
#ifdef _WIN32
  auto start = std::chrono::high_resolution_clock::now();
#else
  struct timeval start, end;
  gettimeofday(&start, NULL);
#endif

  // execute the file on the original image a couple of times
  for (int i = 0; i < Ntimes; i++)
  {
    blur5x5_1(src, dst);
  }

#ifdef _WIN32
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  double time_taken = elapsed.count();
#else
  gettimeofday(&end, NULL);
  double time_taken = (end.tv_sec - start.tv_sec) * 1e6;
  time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
#endif

  // compute the time per image
  double difference = time_taken / Ntimes;

  // print the results
  printf("Time per image (1): %.4lf seconds\n", difference);

  //////////////////////////////
  // set up the timing for version 2
#ifdef _WIN32
  start = std::chrono::high_resolution_clock::now();
#else
  gettimeofday(&start, NULL);
#endif

  // execute the file on the original image a couple of times
  for (int i = 0; i < Ntimes; i++)
  {
    blur5x5_2(src, dst);
  }

#ifdef _WIN32
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  time_taken = elapsed.count();
#else
  gettimeofday(&end, NULL);
  time_taken = (end.tv_sec - start.tv_sec) * 1e6;
  time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
#endif

  // compute the time per image
  difference = time_taken / Ntimes;

  // Save the blurred image
  cv::imwrite("blurred.jpg", dst);

  // print the results
  printf("Time per image (2): %.4lf seconds\n", difference);

  // terminate the program
  printf("Terminating\n");

  return (0);
}
