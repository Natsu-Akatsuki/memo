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

inline int64_t volume(const nvinfer1::Dims &d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
  case nvinfer1::DataType::kINT32:
    return 4;
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kINT8:
    return 1;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

inline size_t getOutputSize(const std::vector<int64_t> &mTrtBindBufferSize,
                            int mTrtInputCount) {
  return std::accumulate(mTrtBindBufferSize.begin() + mTrtInputCount,
                         mTrtBindBufferSize.end(), 0);
};

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


  cudaEvent_t start, stop; // using cuda events to measure time
  float elapsed_time_ms;   // which is applicable for asynchronous code also


  cudaEventCreate(&start); // instrument code to measure start time
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  nvinfer1::IExecutionContext *context = engine->createExecutionContext();
  // enqueue() does not currently support profiling.
  context->setProfiler(&profiler);
  int nbBindings = engine->getNbBindings();

  int mTrtInputCount = 0;
  const int maxBatchSize = 1;
  std::vector<void *> mTrtCudaBuffer;
  std::vector<int64_t> mTrtBindBufferSize;
  mTrtCudaBuffer.resize(nbBindings);
  mTrtBindBufferSize.resize(nbBindings);
  for (int i = 0; i < nbBindings; ++i) {
    Dims dims = engine->getBindingDimensions(i);
    DataType dtype = engine->getBindingDataType(i);
    int64_t totalSize = volume(dims) * maxBatchSize * getElementSize(dtype);
    mTrtBindBufferSize[i] = totalSize;
    cudaMalloc(&mTrtCudaBuffer[i], mTrtBindBufferSize[i]);
    if (engine->bindingIsInput(i))
      mTrtInputCount++;
  }

  float *host_data_input = new float[32000 * 20 * 10];
  float *host_data_output = new float[32000 * 64];
  int inputIndex = 0;
  cudaMemcpyAsync(mTrtCudaBuffer[inputIndex], host_data_input,
                  mTrtBindBufferSize[inputIndex], cudaMemcpyHostToDevice);

  /*
   * Parameters
   * batchSize: The batch size. This is at most the value supplied when the
   * engine was built.
   * bindings（指针数组）: An array of pointers to input and output
   * buffers for the network
   */
  const int batchSize = 1;
  inputIndex = 0;
  context->execute(batchSize, &mTrtCudaBuffer[inputIndex]);

  // data GPU->CPU
  for (size_t bindingIdx = mTrtInputCount;
       bindingIdx < mTrtBindBufferSize.size(); ++bindingIdx) {
    auto size = mTrtBindBufferSize[bindingIdx];
    cudaMemcpyAsync(host_data_output, mTrtCudaBuffer[bindingIdx], size,
                    cudaMemcpyDeviceToHost);
  }

  cudaEventRecord(stop, 0); // instrument code to measue end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

  printf("Time to calculate results: %f ms.\n",
         elapsed_time_ms); // print out execution time
  // TODO:
  delete[] host_data_input;
  delete[] host_data_output;
  return 0;
}