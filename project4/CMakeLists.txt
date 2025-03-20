cmake_minimum_required(VERSION 3.16)
project(Project5)

# Use vcpkg toolchain if available
if(EXISTS "C:/Users/alvin/Desktop/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/alvin/Desktop/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
endif()

# Find required packages
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

# Ensure OpenGL is found
if(NOT OPENGL_FOUND)
    message(FATAL_ERROR "OpenGL not found!")
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})

# Add executables
add_executable(vidDisplay vidDisplay.cpp)
add_executable(cameraPose cameraPose.cpp)
add_executable(featureDetection featureDetection.cpp)

# Link libraries
target_link_libraries(vidDisplay ${OpenCV_LIBS} ${OPENGL_LIBRARIES})
target_link_libraries(cameraPose ${OpenCV_LIBS} ${OPENGL_LIBRARIES})
target_link_libraries(featureDetection ${OpenCV_LIBS} ${OPENGL_LIBRARIES})
