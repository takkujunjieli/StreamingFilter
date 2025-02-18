cmake_minimum_required(VERSION 3.10)
project(Project3)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Set ONNX Runtime paths
set(ONNXRUNTIME_DIR "C:/Program Files/onnxruntime-win-x64-1.20.1") 
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Add executables
add_executable(vidDisplay vidDisplay.cpp or.cpp)
add_executable(imgDisplay imgDisplay.cpp or.cpp) 
# Link libraries
target_link_libraries(vidDisplay ${OpenCV_LIBS} onnxruntime)
target_link_libraries(imgDisplay ${OpenCV_LIBS} onnxruntime)