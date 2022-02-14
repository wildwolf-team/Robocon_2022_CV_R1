cmake_minimum_required(VERSION 3.0.0)

project(Robocon_2022_CV_R1)

add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -pthread")

# add_definitions(-DRELEASE)

add_executable(Robocon_2022_CV_R1 main.cpp)
include_directories(${PROJECT_SOURCE_DIR})

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Robocon_2022_CV_R1 ${OpenCV_LIBS})

#fmt
find_package(fmt REQUIRED)
target_link_libraries(Robocon_2022_CV_R1 fmt::fmt)

#Egien
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
target_link_libraries(Robocon_2022_CV_R1 ${EIGEN3_LIBS})

#Camera
add_library(MVSDK SHARED IMPORTED)
add_library(Camera devices/camera/mv_video_capture.cpp)
set(MIMDVISION_SDK_LIBRARY /lib/libMVSDK.so)
set_property(TARGET MVSDK PROPERTY IMPORTED_LOCATION ${MIMDVISION_SDK_LIBRARY})
target_link_libraries(Robocon_2022_CV_R1 Camera MVSDK)

#Uart
add_library(serial devices/serial/uart_serial.cpp)
target_link_libraries(Robocon_2022_CV_R1 serial)

# new serial
set(serial_SRCS
    devices/new_serial/src/serial.cc
    devices/new_serial/src/impl/unix.cc
    devices/new_serial/src/impl/list_ports/list_ports_linux.cc
    devices/new_serial/serial/serial.h
    devices/new_serial/serial/v8stdint.h
)
include_directories(devices/new_serial)
add_library(new_serial ${serial_SRCS})
target_link_libraries(Robocon_2022_CV_R1 new_serial)

#TensorRTx
find_package(CUDA REQUIRED)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

cuda_add_library(myplugins SHARED TensorRTx/yololayer.cu) # 生成动态库 myplugins
# target_link_libraries(myplugins nvinfer cudart) # nvinfer cudart 是 TensorRT的库?
target_link_libraries(myplugins nvinfer) # nvinfer cudart 是 TensorRT的库?

#OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# 生成/×可×执×行×文×件×/ 动态库√ TensorRTx, 基于calibrator.cpp preprocess.cu
cuda_add_library(TensorRTx SHARED TensorRTx/calibrator.cpp TensorRTx/yolov5.hpp TensorRTx/preprocess.cu) 

# 把nvinfer、cudart、myplugins 和 opencv库 -->(链接到) TensorRTx
target_link_libraries(TensorRTx nvinfer)
# target_link_libraries(TensorRTx cudart)
target_link_libraries(TensorRTx myplugins)
target_link_libraries(TensorRTx ${OpenCV_LIBS})

target_link_libraries(Robocon_2022_CV_R1 TensorRTx)

set(CONFIG_FILE_PATH ${PROJECT_SOURCE_DIR}/configs)
set(SOURCE_PATH ${PROJECT_SOURCE_DIR})
target_compile_definitions(Robocon_2022_CV_R1 PRIVATE "CONFIG_FILE_PATH=\"${CONFIG_FILE_PATH}\"")
target_compile_definitions(Robocon_2022_CV_R1 PRIVATE "SOURCE_PATH=\"${PROJECT_SOURCE_DIR}\"")