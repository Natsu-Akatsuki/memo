
int main(){
  std::vector<char> engineData(fsize);
  engineFile.read(engineData.data(), fsize);

  util::UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};

  util::UniquePtr<nvinfer1::ICudaEngine> mEngine(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
}