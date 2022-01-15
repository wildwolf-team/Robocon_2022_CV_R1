cmake_minimum_required(VERSION 3.0.0)

project(Robocon_2022_CV)

set(CMAKE_CXX_FLAGS "-pthread") 
add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

add_executable(Robocon_2022_CV main.cpp)
include_directories(${PROJECT_SOURCE_DIR})

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Robocon_2022_CV ${OpenCV_LIBS})

#fmt
find_package(fmt REQUIRED)
target_link_libraries(Robocon_2022_CV fmt::fmt)

#Camera
add_subdirectory(Camera)
target_link_libraries(Robocon_2022_CV Camera)

#solvePnP
# add_subdirectory(angle/solvePnP)
# target_link_libraries(Robocon_2022_CV solvePnP)

#Uart
add_subdirectory(serial)
target_link_libraries(Robocon_2022_CV serial)

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

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)

target_link_libraries(Robocon_2022_CV TensorRTx)

set(CONFIG_FILE_PATH ${PROJECT_SOURCE_DIR}/configs)
set(SOURCE_PATH ${PROJECT_SOURCE_DIR})
target_compile_definitions(Robocon_2022_CV PRIVATE "CONFIG_FILE_PATH=\"${CONFIG_FILE_PATH}\"")
target_compile_definitions(Robocon_2022_CV PRIVATE "SOURCE_PATH=\"${PROJECT_SOURCE_DIR}\"")