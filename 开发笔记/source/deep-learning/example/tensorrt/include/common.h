/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H

// For loadLibrary
#ifdef _MSC_VER
// Needed so that the max/min definitions in windows.h do not conflict with
// std::max/min.
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <dlfcn.h>
#endif

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;

constexpr long double operator"" _GiB(long double val) {
  return val * (1 << 30);
}
constexpr long double operator"" _MiB(long double val) {
  return val * (1 << 20);
}
constexpr long double operator"" _KiB(long double val) {
  return val * (1 << 10);
}

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(unsigned long long val) {
  return val * (1 << 30);
}
constexpr long long int operator"" _MiB(unsigned long long val) {
  return val * (1 << 20);
}
constexpr long long int operator"" _KiB(unsigned long long val) {
  return val * (1 << 10);
}

/**
 * 步骤一：实例化一个ILogger接口类来捕获TensorRT的日志信息
 */
class Logger : public nvinfer1::ILogger {
public:
  // void log(Severity severity, const char *msg)   + noexcept 才行的原因？
  void log(Severity severity, const char *msg) noexcept {
    // 设置日志等级
    if (severity <= Severity::kWARNING) {
      timePrefix();
      std::cout << severityPrefix(severity) << std::string(msg) << std::endl;
    }
  }

private:
  static const char *severityPrefix(Severity severity) {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
      return "[F] ";
    case Severity::kERROR:
      return "[E] ";
    case Severity::kWARNING:
      return "[W] ";
    case Severity::kINFO:
      return "[I] ";
    case Severity::kVERBOSE:
      return "[V] ";
    default:
      // #include <cassert>
      assert(0);
      return "";
    }
  }
  void timePrefix() {
    std::time_t timestamp = std::time(nullptr);
    tm *tm_local = std::localtime(&timestamp);
    std::cout << "[";
    std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon
              << "/";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
    std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year
              << "-";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
  }

} logger;

bool saveEngine(const nvinfer1::ICudaEngine &engine,
                const std::string &fileName) {
  std::ofstream engineFile(fileName, std::ios::binary | std::ios::out);
  if (!engineFile) {
    logger.log(Logger::Severity::kERROR,
               (std::string("Cannot open engine file") + "fileName").c_str());
    return false;
  }

  auto serializedEngine = engine.serialize();
  if (serializedEngine == nullptr) {
    logger.log(Logger::Severity::kERROR, "Engine serialization failed");
    return false;
  }

  engineFile.write(static_cast<char *>(serializedEngine->data()),
                   serializedEngine->size());
  return !engineFile.fail();
}

/**
 * Collect per-layer profile information, assuming times are reported in the
 * same order abstract from autoware
 * lidar_apollo_instance_segmentation/lib/include/Utils.h
 */
class Profiler : public nvinfer1::IProfiler {
public:
  void printLayerTimes(int itrationsTimes) {
    float totalTime = 0;
    for (size_t i = 0; i < mProfile.size(); i++) {
      printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(),
             mProfile[i].second / itrationsTimes);
      totalTime += mProfile[i].second;
    }
    printf("Time over all layers: %4.3f\n", totalTime / itrationsTimes);
  }

private:
  typedef std::pair<std::string, float> Record;
  std::vector<Record> mProfile;

  virtual void reportLayerTime(const char *layerName, float ms) noexcept {
    auto record =
        std::find_if(mProfile.begin(), mProfile.end(),
                     [&](const Record &r) { return r.first == layerName; });
    if (record == mProfile.end())
      mProfile.push_back(std::make_pair(layerName, ms));
    else
      record->second += ms;
  }
} profiler;

struct InferDeleter {
  template <typename T> void operator()(T *obj) const {
    if (obj)
      obj->destroy();
  }
};

template <typename T> using UniquePtr = std::unique_ptr<T, InferDeleter>;

// class BufferManager {
// public:
//   static const size_t kINVALID_SIZE_VALUE = ~size_t(0);
//
//   //!
//   //! \brief Create a BufferManager for handling buffer interactions with
//   //! engine.
//   //!
//   BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine,
//                 const int batchSize = 0,
//                 const nvinfer1::IExecutionContext *context = nullptr)
//       : mEngine(engine), mBatchSize(batchSize) {
//     // Full Dims implies no batch size.
//     assert(engine->hasImplicitBatchDimension() || mBatchSize == 0);
//     // Create host and device buffers
//     for (int i = 0; i < mEngine->getNbBindings(); i++) {
//       auto dims = context ? context->getBindingDimensions(i)
//                           : mEngine->getBindingDimensions(i);
//       size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
//       nvinfer1::DataType type = mEngine->getBindingDataType(i);
//       int vecDim = mEngine->getBindingVectorizedDim(i);
//       if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
//       {
//         int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
//         dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
//         vol *= scalarsPerVec;
//       }
//       vol *= samplesCommon::volume(dims);
//       std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
//       manBuf->deviceBuffer = DeviceBuffer(vol, type);
//       manBuf->hostBuffer = HostBuffer(vol, type);
//       mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
//       mManagedBuffers.emplace_back(std::move(manBuf));
//     }
//   }
//
//   //!
//   //! \brief Returns a vector of device buffers that you can use directly as
//   //!        bindings for the execute and enqueue methods of IExecutionContext.
//   //!
//   std::vector<void *> &getDeviceBindings() { return mDeviceBindings; }
//
//   //!
//   //! \brief Returns a vector of device buffers.
//   //!
//   const std::vector<void *> &getDeviceBindings() const {
//     return mDeviceBindings;
//   }
//
//   //!
//   //! \brief Returns the device buffer corresponding to tensorName.
//   //!        Returns nullptr if no such tensor can be found.
//   //!
//   void *getDeviceBuffer(const std::string &tensorName) const {
//     return getBuffer(false, tensorName);
//   }
//
//   //!
//   //! \brief Returns the host buffer corresponding to tensorName.
//   //!        Returns nullptr if no such tensor can be found.
//   //!
//   void *getHostBuffer(const std::string &tensorName) const {
//     return getBuffer(true, tensorName);
//   }
//
//   //!
//   //! \brief Returns the size of the host and device buffers that correspond to
//   //! tensorName.
//   //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
//   //!
//   size_t size(const std::string &tensorName) const {
//     int index = mEngine->getBindingIndex(tensorName.c_str());
//     if (index == -1)
//       return kINVALID_SIZE_VALUE;
//     return mManagedBuffers[index]->hostBuffer.nbBytes();
//   }
//
//   //!
//   //! \brief Dump host buffer with specified tensorName to ostream.
//   //!        Prints error message to std::ostream if no such tensor can be
//   //!        found.
//   //!
//   void dumpBuffer(std::ostream &os, const std::string &tensorName) {
//     int index = mEngine->getBindingIndex(tensorName.c_str());
//     if (index == -1) {
//       os << "Invalid tensor name" << std::endl;
//       return;
//     }
//     void *buf = mManagedBuffers[index]->hostBuffer.data();
//     size_t bufSize = mManagedBuffers[index]->hostBuffer.nbBytes();
//     nvinfer1::Dims bufDims = mEngine->getBindingDimensions(index);
//     size_t rowCount = static_cast<size_t>(
//         bufDims.nbDims > 0 ? bufDims.d[bufDims.nbDims - 1] : mBatchSize);
//     int leadDim = mBatchSize;
//     int *trailDims = bufDims.d;
//     int nbDims = bufDims.nbDims;
//
//     // Fix explicit Dimension networks
//     if (!leadDim && nbDims > 0) {
//       leadDim = bufDims.d[0];
//       ++trailDims;
//       --nbDims;
//     }
//
//     os << "[" << leadDim;
//     for (int i = 0; i < nbDims; i++)
//       os << ", " << trailDims[i];
//     os << "]" << std::endl;
//     switch (mEngine->getBindingDataType(index)) {
//     case nvinfer1::DataType::kINT32:
//       print<int32_t>(os, buf, bufSize, rowCount);
//       break;
//     case nvinfer1::DataType::kFLOAT:
//       print<float>(os, buf, bufSize, rowCount);
//       break;
//     case nvinfer1::DataType::kHALF:
//       print<half_float::half>(os, buf, bufSize, rowCount);
//       break;
//     case nvinfer1::DataType::kINT8:
//       assert(0 && "Int8 network-level input and output is not supported");
//       break;
//     case nvinfer1::DataType::kBOOL:
//       assert(0 && "Bool network-level input and output are not supported");
//       break;
//     }
//   }
//
//   //!
//   //! \brief Templated print function that dumps buffers of arbitrary type to
//   //! std::ostream.
//   //!        rowCount parameter controls how many elements are on each line.
//   //!        A rowCount of 1 means that there is only 1 element on each line.
//   //!
//   template <typename T>
//   void print(std::ostream &os, void *buf, size_t bufSize, size_t rowCount) {
//     assert(rowCount != 0);
//     assert(bufSize % sizeof(T) == 0);
//     T *typedBuf = static_cast<T *>(buf);
//     size_t numItems = bufSize / sizeof(T);
//     for (int i = 0; i < static_cast<int>(numItems); i++) {
//       // Handle rowCount == 1 case
//       if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
//         os << typedBuf[i] << std::endl;
//       else if (rowCount == 1)
//         os << typedBuf[i];
//       // Handle rowCount > 1 case
//       else if (i % rowCount == 0)
//         os << typedBuf[i];
//       else if (i % rowCount == rowCount - 1)
//         os << " " << typedBuf[i] << std::endl;
//       else
//         os << " " << typedBuf[i];
//     }
//   }
//
//   //!
//   //! \brief Copy the contents of input host buffers to input device buffers
//   //! synchronously.
//   //!
//   void copyInputToDevice() { memcpyBuffers(true, false, false); }
//
//   //!
//   //! \brief Copy the contents of output device buffers to output host buffers
//   //! synchronously.
//   //!
//   void copyOutputToHost() { memcpyBuffers(false, true, false); }
//
//   //!
//   //! \brief Copy the contents of input host buffers to input device buffers
//   //! asynchronously.
//   //!
//   void copyInputToDeviceAsync(const cudaStream_t &stream = 0) {
//     memcpyBuffers(true, false, true, stream);
//   }
//
//   //!
//   //! \brief Copy the contents of output device buffers to output host buffers
//   //! asynchronously.
//   //!
//   void copyOutputToHostAsync(const cudaStream_t &stream = 0) {
//     memcpyBuffers(false, true, true, stream);
//   }
//
//   ~BufferManager() = default;
//
// private:
//   void *getBuffer(const bool isHost, const std::string &tensorName) const {
//     int index = mEngine->getBindingIndex(tensorName.c_str());
//     if (index == -1)
//       return nullptr;
//     return (isHost ? mManagedBuffers[index]->hostBuffer.data()
//                    : mManagedBuffers[index]->deviceBuffer.data());
//   }
//
//   void memcpyBuffers(const bool copyInput, const bool deviceToHost,
//                      const bool async, const cudaStream_t &stream = 0) {
//     for (int i = 0; i < mEngine->getNbBindings(); i++) {
//       void *dstPtr = deviceToHost ? mManagedBuffers[i]->hostBuffer.data()
//                                   : mManagedBuffers[i]->deviceBuffer.data();
//       const void *srcPtr = deviceToHost
//                                ? mManagedBuffers[i]->deviceBuffer.data()
//                                : mManagedBuffers[i]->hostBuffer.data();
//       const size_t byteSize = mManagedBuffers[i]->hostBuffer.nbBytes();
//       const cudaMemcpyKind memcpyType =
//           deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
//       if ((copyInput && mEngine->bindingIsInput(i)) ||
//           (!copyInput && !mEngine->bindingIsInput(i))) {
//         if (async)
//           CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
//         else
//           CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
//       }
//     }
//   }
//
//   std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The pointer to the engine
//   int mBatchSize; //!< The batch size for legacy networks, 0 otherwise.
//   std::vector<std::unique_ptr<ManagedBuffer>>
//       mManagedBuffers; //!< The vector of pointers to managed buffers
//   std::vector<void *> mDeviceBindings; //!< The vector of device buffers needed
//                                        //!< for engine execution
// };
//
// } // namespace samplesCommon

#endif
