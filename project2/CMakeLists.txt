cmake_minimum_required(VERSION 3.10)
project(Project2)

# Find OpenCV package
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Set ONNX Runtime paths
set(ONNXRUNTIME_DIR "C:/Program Files/onnxruntime-win-x64-1.20.1") 
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)
set(SOURCES
    findSimilarImages.cpp
    csv_util.cpp
    extractFeature.cpp
    processImages.cpp
    matchDistance.cpp
)
# Add executables
add_executable(similar ${SOURCES})
# Link libraries
target_link_libraries(similar ${OpenCV_LIBS} onnxruntime)