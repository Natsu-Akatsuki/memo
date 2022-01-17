# Tensor-practice

## workflow

步骤一：创建`logger`对象，用于捕获TensorRT运行时的日志
步骤二：创建`builder`对象，构建TensorRT模型
步骤三：创建`config`对象，用于指导TensorRT的优化方式
步骤四：创建`paser`对象，将onnx模型的权值populate到TensorRT模型中
步骤五：创建`context`对象，进行预测

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211228132613702.png" alt="image-20211228132613702" style="zoom:67%;" />

```c++
#include "NvInferRuntime.h"
#include <NvInfer.h>
// create a config object
nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

// network->plan->engine
nvinfer1::IHostMemory *plan = builder->buildSerializedNetwork(*network, *config);
nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(plan->data(), plan->size(), nullptr);

// network->engine
nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config

// engine->context
nvinfer1::IExecutionContext *context = engine->createExecutionContext();                   
```

- logger：logger是共用的

![image-20220115160310062](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220115160310062.png)

## command

### trtexec

- 各种option的含义可参考[here](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#runtime)和`trtexec -h`的输出

```bash
$ trtexec --onnx=fcn-resnet101.onnx --fp16 --workspace=64 \
--minShapes=input:1x3x256x256 \
--optShapes=input:1x3x1026x1282 \
--maxShapes=input:1x3x1440x2560 \
--buildOnly \
--saveEngine=fcn-resnet101.engine
--explicitBatch
# --buildOnly：不需要inference performance measurements
# --saveEngine: 模型导出

# e.g.
$ trtexec --onnx=pfe_baseline32000.onnx --fp16 --workspace=16384 --saveEngine=pfe_baseline_fp16.engine 
$ trtexec --onnx=rpn_baseline.onnx --fp16 --workspace=16384 --saveEngine=rpn_baseline_fp16.engine
```

## TensorRT plugin

plugin为TensorRT的精髓，提供了一个接口进行自定义算子的导入



## DEBUG

### 未成功导入头文件NvInfer.h: No such file or directory

```bash
.../.../inference_helper_tensorrt.cpp:30:10: fatal error: NvInfer.h: No such file or directory 
   30 | #include <NvInfer.h> 
      |          ^~~~~~~~~~~
```

#### error: ‘class nvinfer1::IBuilder’ has no member named ‘buildSerializedNetwork’

TensorRT版本号不对应：原使用了TensorRT 7.2.3的库，而以下的成员函数是从8.0.1开始才有的

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/f7wFWD3eJdtgncoK.png!thumbnail" alt="img" style="zoom:67%;" />

## Q&A

### 多流为什么有效？

- CPU->GPU数据是经过PCIe总线进行传输的。在传输过程中，CPU和GPU处于空闲的等待状态。多流则可以实现数据传输与核函数计算的并行。
- 多流可以让多个核函数同时计算，充分利用GPU算理

.. note:: 流并非越多越好，GPU内可同时执行的流数量是有限的

.. note:: GOU流指的是GPU操作(operation)序列(sequence)

### [为什么plan（TensorRT模型文件）不能够在不同架构下运行？](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#compatibility-serialized-engines)

架构：e.g. Turing架构（RTX 2060）、Pascal架构(GTX 1080)

但可在相同架构的不同显卡下运行

.. note:: Serialized engines are not portable across platforms or TensorRT versions. Engines are specific to the exact GPU model they were built on (in addition to the platform and the TensorRT version).

### TensorRT的输入为什么要固定？

为什么要调用setmaxbatchsize？对输入定死后才能够进行模型的调优？

### TensorRT的调优策略？

该部分耗时是最长的。涉及：模型转换、kernel自动调优、算子融合和低精度

kernel自动调优：不需要考虑分支（能解释不同plan）

### 常用的设置参数

- setMaxWorkspaceSize()：执行时的显存用量

```c++
// IBuilderConfig::setMaxWorkspaceSize
auto builder = nvinfer1::createInferBuilder(gLogger);
auto config = builder->createBuilderConfig();
// config->setMaxWorkspaceSize(128*(1 << 20)); // 128 MiB

config->setMaxWorkspaceSize(16_MiB);
config->setMaxWorkspaceSize(5_GiB);
```

![image-20211227140227316](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211227140227316.png)

.. note:: One important property is the maximum workspace size. Layer implementations often require a temporary workspace, and this parameter limits the maximum size that any layer in the network can use. If insufficient workspace is provided, it is possible that TensorRT will not be able to find an implementation for a layer.

.. note:: some tatics do not have suffiient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.

![image-20211228160528545](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211228160528545.png)

- [Change the workspace size](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt/)：太低将得到次优的模型

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211229090819788.png" alt="image-20211229090819788" style="zoom: 50%;" />

#### [精度配置](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#network-level-control)

```c++
config->setFlag(BuilderFlag::kFP16);
config->setFlag(BuilderFlag::kINT8);
```

---

**NOTE**

- [查看硬件所支持的精度](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix)



### 程序中binding的意思？

存储输入输出内存地址的数组(An array of pointers to input and output buffers for the network)，所以单输入单输出的一般的nbBindinds=2

```c++
int nbBindings = engine->getNbBindings();
std::vector<void *> mTrtCudaBuffer;
std::vector<int64_t> mTrtBindBufferSize;
mTrtCudaBuffer.resize(nbBindings);
mTrtBindBufferSize.resize(nbBindings);
```

### TensorRT版本的选择

1. 选择LTS版本的，例如能选7.2就不要选7.0和7.1
2. 根据显卡来选TensorRT的版本。并不是版本更好越新越好，版本越新仅是对新的显卡优化效果更好，旧的效果反而效果会差一些。（软件TensoRT每次的优化和迭代都是与推出的N卡息息相关）

### [官方Q&A](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#troubleshooting)

### 为什么TensorRT的很多对象都有智能指针管理？

- TensorRT的对象需要调用destroy()进行析构

![image-20220116222110245](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220116222110245.png)

### 查看onnx模型的输入和输出大小

方法一：使用onnx脚本查看

```bash
$ pip install onnx
```

相关代码：

```python
import onnx

def print_shape_info(channel):
    for input in eval(f"model.graph.{channel}"):
        print(input.name, end=": ")
        # get type of input tensor
        tensor_type = input.type.tensor_type
        # check if it has a shape:
        if tensor_type.HasField("shape"):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if d.HasField("dim_value"):
                    print(d.dim_value, end=", ")  # known dimension
                elif d.HasField("dim_param"):
                    print(d.dim_param, end=", ")  # unknown dimension with symbolic name
                else:
                    print("?", end=", ")  # unknown dimension with no name
        else:
            print("unknown rank", end="")

model_path = "....onnx"
model = onnx.load(model_path)

print_shape_info("input")
print()
print_shape_info("output")
```

方法二：[netron online](https://netron.app/)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/Zz7SjGciDpzbgA3F.png)



### 验证TensorRT engine

- 命令行测试

```bash
$ trtexec --shapes=input:32000x64 --loadEngine=pfe_baseline32000.trt
# input大小可参考上一节：查看onnx模型的输入和输出大小
```

## [术语](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#glossary)

- [序列化](https://en.wikipedia.org/wiki/Serialization)：序列化模型能够更好的存储模型

- network definition：TensorRT中model的别称

- plan：序列化后的**优化**模型(inference model)/TensorRT导出的模型 - An optimized inference engine in a serialized format.

  ![image-20211227151748279](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211227151748279.png)

- engine：被TensorRT builder**优化好**的模型(model)

- In **CUDA**, the **host** refers to the CPU and its memory, while the **device** refers to the GPU and its memory. Code run on the **host** can manage memory on both the **host** and **device**, and also launches **kernels** which are functions executed on the **device**.

- 

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211228112641903.png" alt="image-20211228112641903" style="zoom:67%;" />
