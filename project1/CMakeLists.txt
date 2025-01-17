cmake_minimum_required(VERSION 3.10)
project(Project1)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executables
add_executable(timeBlur timeBlur.cpp filter.cpp faceDetect.cpp)
add_executable(vidDisplay vidDisplay.cpp filter.cpp faceDetect.cpp) 

# Link libraries
target_link_libraries(timeBlur ${OpenCV_LIBS})
target_link_libraries(vidDisplay ${OpenCV_LIBS})