cmake_minimum_required(VERSION 3.5)

project(robocon_2022_cv_r1)

add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -pthread")

# add_definitions(-DRELEASE)
set(SRCS
    main.cpp
    robo.cpp
    devices/camera/mv_video_capture.cpp
    devices/serial/uart_serial.cpp)
add_executable(${PROJECT_NAME} ${SRCS})

find_package(InferenceEngine REQUIRED)
if(InferenceEngine_FOUND)
add_definitions(-DUSE_OV_DETECTOR)
include_directories(${InferenceEngine_INCLUDE_DIRS})
add_library(OpenVinoSRC OpenVINO/ovyolov5.cpp)
target_link_libraries(${PROJECT_NAME} OpenVinoSRC ${InferenceEngine_LIBRARIES})
endif(InferenceEngine_FOUND)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} fmt::fmt ${EIGEN3_LIBS})

#Camera
add_library(MVSDK SHARED IMPORTED)
set(MIMDVISION_SDK_LIBRARY /lib/libMVSDK.so)
set_property(TARGET MVSDK PROPERTY IMPORTED_LOCATION ${MIMDVISION_SDK_LIBRARY})
target_link_libraries(${PROJECT_NAME} MVSDK)

# new serial
set(serial_SRCS
    devices/new_serial/src/serial.cc
    devices/new_serial/src/impl/unix.cc
    devices/new_serial/serial/serial.h
    devices/new_serial/serial/v8stdint.h
    devices/new_serial/serial.cpp
)
include_directories(devices/new_serial)
add_library(new_serial ${serial_SRCS})
target_link_libraries(${PROJECT_NAME} new_serial)

# OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#TensorRTx
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
add_definitions(-DUSE_TRT_DETECTOR)
add_library(tensorrt TensorRTx/common.cpp)
# cuda
include_directories(/usr/local/cuda11.6/include)
link_directories(/usr/local/cuda11.6/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
cuda_add_library(myplugins SHARED TensorRTx/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)
cuda_add_library(TensorRTx SHARED TensorRTx/calibrator.cpp TensorRTx/yolov5.hpp TensorRTx/preprocess.cu)
target_link_libraries(TensorRTx nvinfer cudart myplugins ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} TensorRTx tensorrt)
endif(CUDA_FOUND)

set(CONFIG_FILE_PATH ${PROJECT_SOURCE_DIR}/configs)
set(SOURCE_PATH ${PROJECT_SOURCE_DIR})
target_compile_definitions(${PROJECT_NAME} PRIVATE "CONFIG_FILE_PATH=\"${CONFIG_FILE_PATH}\"")
target_compile_definitions(${PROJECT_NAME} PRIVATE "SOURCE_PATH=\"${PROJECT_SOURCE_DIR}\"")