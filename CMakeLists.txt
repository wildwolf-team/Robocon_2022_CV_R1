cmake_minimum_required(VERSION 3.0.0)
project(Robocon_2022_CV VERSION 0.1.0)

set(CMAKE_CXX_FLAGS "-pthread") 

add_executable(Robocon_2022_CV main.cpp)

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Robocon_2022_CV ${OpenCV_LIBS})

#fmt
find_package(fmt REQUIRED)
target_link_libraries(Robocon_2022_CV fmt::fmt)

#Camera
add_subdirectory(Camera)
include_directories(Camera)
target_link_libraries(Robocon_2022_CV Camera)