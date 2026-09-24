cmake_minimum_required(VERSION 3.16)
project(sensor_fusion_lab CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# No dependencies: the lidar pipeline and the radar chain (including its FFT) are self-contained.
add_executable(lidar_pipeline lidar/lidar_pipeline.cpp)
add_executable(fmcw_radar radar/fmcw_radar.cpp)

# Kalman filters need Eigen (apt install libeigen3-dev).
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(Eigen3_FOUND)
  add_executable(fusion_filters kalman/fusion_filters.cpp)
  target_link_libraries(fusion_filters Eigen3::Eigen)
else()
  message(WARNING "Eigen3 not found: skipping fusion_filters (apt install libeigen3-dev)")
endif()

# Camera TTC needs OpenCV 4 (apt install libopencv-dev). BRIEF and FREAK additionally need
# opencv_contrib; the code compiles them only if HAVE_OPENCV_XFEATURES2D is defined.
find_package(OpenCV 4 QUIET)
if(OpenCV_FOUND)
  add_executable(camera_ttc camera/camera_ttc.cpp)
  target_include_directories(camera_ttc PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(camera_ttc ${OpenCV_LIBS})
else()
  message(WARNING "OpenCV 4 not found: skipping camera_ttc (apt install libopencv-dev)")
endif()
