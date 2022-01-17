#include "NvInferRuntime.h"
#include "common.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

int main() {
  std::string input_file = "pfe_baseline32000.onnx";

  // 创建builder对象
  nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);

  // 创建config对象（用于指导TensorRT优化模型）
  auto config = builder->createBuilderConfig();
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  config->setMaxWorkspaceSize(5_GiB);

  // INT8不能直接使用
  // Internal Error (Calibration failure occurred with no scaling factors detected.
  // This could be due to no int8 calibrator or insufficient custom scales for network layers.
  // Please see int8 sample to setup calibration correctly.)
  // config->setFlag(nvinfer1::BuilderFlag::kINT8);


  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  // 创建Tensorrt模型对象
  auto network = builder->createNetworkV2(explicitBatch);

  // 创建paser对象，将onnx模型的权值populate到TensorRT模型中
  auto parser = nvonnxparser::createParser(*network, logger);
  parser->parseFromFile(input_file.c_str(),
                        static_cast<int>(Logger::Severity::kWARNING));

  // This function allows building and serialization of a network without
  // creating an engine note: this api from tensorrt 8.0.1
  nvinfer1::IHostMemory *plan =
      builder->buildSerializedNetwork(*network, *config);
  std::ofstream planFile("pfe_baseline32000.trt", std::ios::binary);
  planFile.write(static_cast<char *>(plan->data()), plan->size());

  return 0;
}