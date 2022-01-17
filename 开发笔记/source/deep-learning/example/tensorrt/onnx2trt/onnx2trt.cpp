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

void create_engine(std::string input_file) {
  // 创建builder对象
  nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);

  // 创建config对象（用于指导TensorRT优化模型）
  auto config = builder->createBuilderConfig();
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  config->setMaxWorkspaceSize(5_GiB);

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
}

int main() {
  std::string input_file = "pfe_baseline32000.onnx";
  std::string plan_file = "pfe_baseline32000.trt";

  std::ifstream fs(plan_file);
  if (!fs.is_open()) {
    std::cout << "Could not find " << plan_file.c_str()
              << " try making TensorRT engine from onnx model";
    create_engine(input_file);
  }

  std::fstream file(plan_file, std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    std::cout << "read engine file " << plan_file << " failed" << std::endl;
    return -1;
  }

  file.seekg(0, std::ios::end);
  int length = file.tellg();
  file.seekg(0, std::ios::beg);
  std::unique_ptr<char[]> data(new char[length]);
  file.read(data.get(), length);
  file.close();

  auto runtime = createInferRuntime(logger);
  auto engine = runtime->deserializeCudaEngine(data.get(), length, nullptr);
  std::cout << "read engine file " << plan_file << " successfully" << std::endl;

  return 0;
}