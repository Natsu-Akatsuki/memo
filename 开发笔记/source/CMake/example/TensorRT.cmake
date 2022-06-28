cmake_minimum_required(VERSION 3.0.2)
project(TensorRT)

# CUDA配置
find_package(CUDA)
find_library(CUBLAS_LIBRARIES cublas HINTS
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib
)
include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA Libs: ${CUDA_LIBRARIES}")
message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")

# TensorRT配置
include_directories($ENV{HOME}/application/TensorRT-8.0.0.3/include)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "$ENV{HOME}/application/TensorRT-8.0.0.3/lib")

find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
find_library(NVONNX_PARSER NAMES nvonnxparser)
set(TENSORRT_LIBRARIES ${NVINFER} ${NVPARSERS} ${NVINFER_PLUGIN} ${NVONNX_PARSER})

message("TensorRT is available!")
message("NVINFER: ${NVINFER}")
message("NVPARSERS: ${NVPARSERS}")
message("NVONNX_PARSER: ${NVONNX_PARSER}")

# CUDNN配置
set(CUDNN_LIBRARY /usr/local/cuda/lib64/libcudnn.so)
# try to find the CUDNN module
find_library(CUDNN_LIBRARY
NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
PATH_SUFFIXES lib lib64 bin
DOC "CUDNN library." )
message("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")

target_link_libraries(tensorrt_apollo_cnn_lib
  ${TENSORRT_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDNN_LIBRARY}
)

