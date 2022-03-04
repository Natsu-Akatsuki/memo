# ONNXPractice

- 导出的onnx model是经过protobuf序列化后的二值文件（需要rb打开）

## 安装

- pip install onnx
- [conda install -c conda-forge onnx](https://anaconda.org/conda-forge/onnx)

## [ONNX IR](https://github.com/onnx/onnx/blob/main/docs/IR.md)

- 不同的onnx可能采用不同的protobuf类型定义文件，可在[here](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)查看对应的版本

  <img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220211015008402.png" alt="image-20220211015008402" style="zoom: 50%;" />

- [使用protobuf定义文件，解读onnx文件](https://github.com/onnx/onnx/blob/main/docs/IR.md#models)

```bash
$ protoc --decode=onnx.ModelProto onnx.proto < yourfile.onnx
```

.. note:: Where onnx.proto is the file that is part of the repository.

- 查看protobuf版本

```bash
$ protoc --version
```

## 查看ONNX模型结构

- 方法一：[netron online](https://netron.app/)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/Zz7SjGciDpzbgA3F.png)

.. note:: 实测，比较低的版本ONNX IR v4也可导入

- 方法二：使用onnx脚本查看

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

## 优化

```python
# 以往的优化器是继承到onnx模块的
# import onnx
# new_model = onnx.optimizer.optimize(model)

# 现在是单独的模块，需pip另外安装
# pip install onnxoptimizer
import onnxoptimizer
new_model = onnxoptimizer.optimize(model)
```

## Q&A

- [libprotobuf ERROR google/protobuf/text_format.cc:298] Error parsing text-format onnx2trt_onnx.ModelProto: 1:1: Invalid control characters encountered in text.... Error parsing text-format onnx2trt_onnx.ModelProto: 1:17: Message type "onnx2trt_onnx.ModelProto" has no field named "pytorch".

> 一种情况是模型在解压缩后broken了（无关onnx version和protobuf version）

## 实战

### h5模型转onnx

```python
# $pip install keras2onnx
import keras
import keras2onnx
import onnx
from keras.models import load_model
model = load_model('model.h5')  
onnx_model = keras2onnx.convert_keras(model, model.name)
temp_model_file = 'model.onnx'
onnx.save_model(onnx_model, temp_model_file)
```
