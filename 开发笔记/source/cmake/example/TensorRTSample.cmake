cmake_minimum_required(VERSION 3.13)
project(sample)

set(CMAKE_CXX_STANDARD 11)

# TensorRT
set(TENSORRT_INSTALL_DIR "$ENV{HOME}/application/TensorRT-8.0.0.3")
# CUDA
find_package(CUDA REQUIRED)
message(“CUDA_LIBRARIES: ${CUDA_LIBRARIES}”)
message(“CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}”)

link_directories(/usr/local/cuda/lib64)
link_directories(${TENSORRT_INSTALL_DIR}/lib/)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TENSORRT_INSTALL_DIR}/samples/common/)
include_directories(${TENSORRT_INSTALL_DIR}/include/)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "${TENSORRT_INSTALL_DIR}/lib")
add_executable(${PROJECT_NAME} sampleOnnxMNIST.cpp)

# try to find the tensorRT modules
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVCAFFE_PARSER NAMES nvcaffe_parser)
find_library(NVONNX_PARSER NAMES nvonnxparser)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
if(NVINFER
   AND NVPARSERS
   AND NVCAFFE_PARSER
   AND NVINFER_PLUGIN)
  message("TensorRT is available!")
  message("NVINFER: ${NVINFER}")
  message("NVPARSERS: ${NVPARSERS}")
  message("NVCAFFE_PARSER: ${NVCAFFE_PARSER}")
else()
  message("TensorRT is NOT Available")
endif()

file(GLOB source_files ${TENSORRT_INSTALL_DIR}/samples/common/*.cpp)
message(STATUS “source_files: ${source_files}”)
add_library(common SHARED ${source_files})
target_link_libraries(common ${CUDA_LIBRARIES} ${NVINFER} ${NVCAFFE_PARSER}
                      ${NVINFER_PLUGIN})

target_link_libraries(${PROJECT_NAME} common ${NVINFER} ${NVCAFFE_PARSER}
                      ${NVINFER_PLUGIN} ${NVONNX_PARSER})
