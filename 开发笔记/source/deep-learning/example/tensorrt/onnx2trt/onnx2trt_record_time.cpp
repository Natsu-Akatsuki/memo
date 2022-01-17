#include "NvInferRuntime.h"
#include "common.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cassert>
#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

int main() {
  std::string input_file = "pfe_baseline32000.onnx";
  float time_elapsed = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start); // create a event
  cudaEventCreate(&stop);

  cudaEventRecord(start); // record the current time
  // 步骤二：实例化builder对象
  nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);

  // 步骤三：创建config对象（用于指导TensorRT优化模型）
  auto config = builder->createBuilderConfig();
  config->setFlag(nvinfer1::BuilderFlag::kFP16);

  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  // 步骤四：实例化NetworkDefinition对象（i.e. tensorrt对象）
  auto network = builder->createNetworkV2(explicitBatch);

  // 步骤五：实例化一个parser来将onnx模型的权值populate到tensorrt模型中
  auto parser = nvonnxparser::createParser(*network, logger);
  parser->parseFromFile(input_file.c_str(),
                        static_cast<int>(Logger::Severity::kWARNING));

  // This function allows building and serialization of a network without
  // creating an engine note: this api from tensorrt 8.0.1
  nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
  std::ofstream planFile("pfe_baseline32000.trt", std::ios::binary);
  planFile.write(static_cast<char *>(plan->data()), plan->size());

  cudaEventRecord(stop); // record the current time

  cudaEventSynchronize(start); // wait for an event to complete
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_elapsed, start, stop);

  cudaEventDestroy(start); // destroy the event
  cudaEventDestroy(stop);

  std::cout << std::string("Time to calculate results: ") << time_elapsed
            << "ms" << std::endl;
  return 0;
}