# TensorRT实战

## TensorRT工作流

### Main Code

主要步骤：构建builder，构建network，导出engine，执行engine

```c++
// 创建logger类型对象（基本操作）
Tn::Logger logger;

// 实例化builder
nvinfer1::IBuilder * builder = nvinfer1::createInferBuilder(logger);
builder->setMaxBatchSize(batch_size);

// 实例化config（用于指导TensorRT优化模型）
nvinfer1::IBuilderConfig * config = builder->createBuilderConfig();
const int batch_size = 1;    
config->setMaxWorkspaceSize(1 << 30);

// 实例化network
nvinfer1::INetworkDefinition * network = builder->createNetworkV2(0U);

// 实例化parser
nvcaffeparser1::ICaffeParser * parser = nvcaffeparser1::createCaffeParser(); 

// 使用parser构建network（加入各种参数）
const nvcaffeparser1::IBlobNameToTensor * blob_name2tensor = parser->parse(
    prototxt_file.c_str(), caffemodel_file.c_str(), *network, nvinfer1::DataType::kFLOAT);

// 序列化engine以保存至硬盘
nvinfer1::IHostMemory * trt_model_stream = engine->serialize();
assert(trt_model_stream != nullptr);    
std::ofstream outfile(engine_file, std::ofstream::binary);assert(!outfile.fail());
outfile.write(reinterpret_cast<char *>(trt_model_stream->data()), trt_model_stream->size());
outfile.close();
```

### ILogger class

大部分的TensorRT API都需要传入ILogger类型的对象作为实参，ILogger可以进行override

```c++
// e.g. offical
class Logger : public ILogger           
 {
     void log(Severity severity, const char* msg) override
     {
         // suppress info-level messages
         if (severity != Severity::kINFO)
             std::cout << msg << std::endl;
     }
 } gLogger;

// e.g. autoware
static Tn::Logger gLogger;
/*
Tn::Logger logger;
nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
*/
```

### [构建TensorRT模型](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#conversion)

- 方法一：调用API构建TensorRT模型（network definition）

- 方法二：调用model parser：pytorch parser([torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt))、第三方库([Tencent Forward](https://github.com/Tencent/Forward))
- 方法三：...

### 构建引擎

```c++
nvinfer1::IRuntime * mTrtRunTime = nullptr;
mTrtRunTime = createInferRuntime(gLogger);
assert(mTrtRunTime != nullptr);

//1st arg: The memory that holds the serialized engine.
//2nd arg: The size of the memory in bytes.
mTrtEngine = mTrtRunTime->deserializeCudaEngine(data.get(), length, nullptr);
assert(mTrtEngine != nullptr);
```

### 导出引擎

- 保存序列化engine

```c++
void saveEngine(std::string fileName)
{
  if (mTrtEngine) {
    // 先序列化再写入以二进制的方式写入文件
    nvinfer1::IHostMemory * data = mTrtEngine->serialize();
    std::ofstream file;
    file.open(fileName, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
      std::cout << "read create engine file" << fileName << " failed" << std::endl;
      return;
    }

    file.write((const char *)data->data(), data->size());
    file.close();
  }
};
```

## [术语](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#glossary)

- [序列化](https://en.wikipedia.org/wiki/Serialization)：序列化模型能够更好的存储模型

- network definition：TensorRT中model的别称

- plan：序列化后的优化模型(inference model)/TensorRT导出的模型 - An optimized inference engine in a serialized format.

  ![image-20211227151748279](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211227151748279.png)

- engine：被TensorRT builder优化好的模型(model)

- In **CUDA**, the **host** refers to the CPU and its memory, while the **device** refers to the GPU and its memory. Code run on the **host** can manage memory on both the **host** and **device**, and also launches **kernels** which are functions executed on the **device**.

## DEBUG

### 未成功导入头文件NvInfer.h: No such file or directory

```bash
.../.../inference_helper_tensorrt.cpp:30:10: fatal error: NvInfer.h: No such file or directory 
   30 | #include <NvInfer.h> 
      |          ^~~~~~~~~~~
```

### error: ‘class nvinfer1::IBuilder’ has no member named ‘buildSerializedNetwork’

TensorRT版本号不对应：原使用了TensorRT 7.2.3的库，而以下的成员函数是从8.0.1开始才有的

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/f7wFWD3eJdtgncoK.png!thumbnail" alt="img" style="zoom:67%;" />

## Q&A

### 为什么plan（TensorRT模型文件）不能够在不同架构下运行？

架构：e.g. Turing架构（RTX 2060）、Pascal架构(GTX 1080)

但可在相同架构的不同显卡下运行

### TensorRT的输入为什么要固定？

为什么要调用setmaxbatchsize？对输入定死后才能够进行模型的调优？

### TensorRT的调优策略？

该部分耗时是最长的。涉及：模型转换、kernel自动调优、算子融合和低精度

### 常用的设置参数

- setMaxWorkspaceSize()：执行时的显存用量

```c++
config->setMaxWorkspaceSize(16_MiB)
```

![image-20211227140227316](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211227140227316.png)

some tatics do not have suffiient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.

### TensorRT版本的选择

1. 选择LTS版本的，例如能选7.2就不要选7.0和7.1
2. 根据显卡来选TensorRT的版本。并不是版本更好越新越好，版本越新仅是对新的显卡优化效果更好，旧的效果反而效果会差一些。（软件TensoRT每次的优化和迭代都是与推出的N卡息息相关）
