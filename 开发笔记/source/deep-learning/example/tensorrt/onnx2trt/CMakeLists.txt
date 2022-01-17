cmake_minimum_required(VERSION 3.0.2)
project(onnx2trt)

# set cuda
find_package(CUDA)
find_library(CUBLAS_LIBRARIES cublas HINTS
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib
  )
message("CUDA Libs: ${CUDA_LIBRARIES}")
message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/../include)

# set tensorrt
include_directories($ENV{HOME}/application/TensorRT-8.0.0.3/include)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "$ENV{HOME}/application/TensorRT-8.0.0.3/lib")
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNX_PARSER NAMES nvonnxparser)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
message("NVINFER: ${NVINFER}")
message("NVPARSERS: ${NVPARSERS}")
message("NVONNX_PARSER: ${NVONNX_PARSER}")

# build project
add_executable(onnx2trt onnx2trt.cpp)
target_link_libraries(onnx2trt ${NVINFER} ${NVPARSERS} ${NVONNX_PARSER} ${CUBLAS_LIBRARIES})
set_target_properties(onnx2trt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(onnx2trt_record_time onnx2trt_record_time.cpp)
target_link_libraries(onnx2trt_record_time ${NVINFER} ${NVPARSERS} ${NVONNX_PARSER} ${CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
set_target_properties(onnx2trt_record_time PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(onnx2trt_v1 onnx2trt_v1.cpp)
target_link_libraries(onnx2trt_v1 ${NVINFER} ${NVPARSERS} ${NVONNX_PARSER} ${CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
set_target_properties(onnx2trt_v1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(onnx2trt_v2 onnx2trt_v2.cpp)
target_link_libraries(onnx2trt_v2 ${NVINFER} ${NVPARSERS} ${NVONNX_PARSER} ${CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
set_target_properties(onnx2trt_v2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})