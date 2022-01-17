# README

本部分于TensorRT 8.2.2.1上进行测试，将使用最新的API

PS：掌握 TensoRT API，有种重回学习单片机看例程，看说明文档的感觉:)

## module

`logger`：使用TensorRT logger进行日志输出

`onnx2rt`：使用c++ binary将onnx模型文件转换为TensorRT plan

.. note:: 自TensorRT 8.0.0.1后有一个新的导出模型的 API `builder->buildSerializedNetwork`，可以不用生成engine再导出序列化模型，而可以直接从TensorRT模型生成优化好的序列化模型。

`onnx2rt_record_time`：在onnx2rt的基础上，加入计时功能

`onnx2rt_v1`：在onnx2rt的基础上，加入engine的导入

`onnx2rt_v2`：在onnx2rt的基础上，加入预测的部分
