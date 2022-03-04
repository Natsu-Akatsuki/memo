# 炼丹清单

## 数据处理

1. 数据是否符合模型输入格式(e.g. 强度信息是否进行归一化)

2. 

dataset里面最重要就是重写__getitem__和__len__getitem就是每次读取数据值时应该返回什么样的数据（比如label, images）

数据集的格式要求，是否有 `split txt` 



### 预处理

- 零均值、1方差的必要性：在BN出现后已经不需要了

![image-20220228131830115](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220228131830115.png)

## 图像预处理

### 通道位置的变换

由opencv读取的图片其shape默认是channel在后，即(H, W, C)；而pytorch网络一般需要的shape为(C, H, W)，因此可以使用：

```python
<i>.permute(2, 0, 1)
```

## 损失函数

### 图像分类

- BCE（二分类）



# 实例

## [lidar_super_resolution](https://github.com/RobustFieldAutonomyLab/lidar_super_resolution) 

### 数据集处理

1. 距离信息归一化（e.g.16线最远探测测距为100m，则除以100，距离大于100的话则设置为0）
2. 

数据集的格式要求，是否有 `split txt` 



top-down flipping,

horizontal flipping and shifting, and range scaling to account for

different environment structures and sensor mounting heights

## 模型(for pytorch)

输入：(batch_size, channel, low_res_img_row, img_col )



| Conv2DTransposeBlock | parameter                                                    |
| -------------------- | ------------------------------------------------------------ |
| Conv2DTranspose      | in_channels<br />out_channels<br />stride<br />kernel_size<br />padding=same<br />padding=(0,1,1,1)<br /> |
| BN+RELU              |                                                              |



| upBlockModule                                   | parameter                                 |
| ----------------------------------------------- | ----------------------------------------- |
| nn.AvgPool2d((2, 2)), <br />nn.Dropout2d(0.25), | stride=(2,2)<br />kernel_size=(2,2)<br /> |
| DoubleConv                                      | (in_channels, out_channels)               |



| downBlockModule                                        | parameter                                               |
| ------------------------------------------------------ | ------------------------------------------------------- |
| nn.AvgPool2d((2, 2)), <br />nn.Dropout2d(0.25), <br /> | stride=(2,2)<br />kernel_size=(2,2)<br />               |
| nn.Conv2d(), <br />nn.BatchNorm2d(),<br /> nn.ReLU()   | stride=(1,1)<br />kernel_size=(3,3)<br />padding='same' |
| nn.Conv2d(), <br />nn.BatchNorm2d(),<br />nn.ReLU()    | stride=(1,1)<br />kernel_size=(3,3)<br />padding='same' |



| convBlockModule                                      | parameter                                                    |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| nn.Conv2d(), <br />nn.BatchNorm2d(),<br /> nn.ReLU() | in_channels1 <br />out_channels1=in_channels1<br />stride=(1,1)<br />kernel_size=(3,3)<br />padding='same' |
| nn.Conv2d(), <br />nn.BatchNorm2d(),<br />nn.ReLU()  | in_channels2=out_channels1 <br />out_channels2=in_channels1<br />stride=(1,1)<br />kernel_size=(3,3)<br />padding='same' |





| layer                | parameter                                                    |
| -------------------- | ------------------------------------------------------------ |
| Conv2DTransposeBlock | (1,1,16,1024)->(1,64,16,1024)<br />in_channels: 1<br />out_channels: 64<br />stride=(2,1)<br />kernel_size=(3,3) |
| Conv2DTransposeBlock | (1,64,16,1024)->(1,64,16,1024)<br />input_channel: 64<br />output_channel: 64 |
| convBlockModule      | (1,64,16,1024)->(1,64,16,1024)<br />input_channel: 64<br />output_channel: 64 |
| downBlockModule      | (1,64,64,1024)->(1,128,64,1024)                              |
| downBlockModule      | (1,128,64,1024)->(1,256,64,1024)                             |
| downBlockModule      | (1,256,64,1024)->(1,512,64,1024)                             |
| downBlockModule      | (1,512,64,1024)->(1,1024,64,1024)                            |
| upBlockModule        | (1,64,64,1024)->(1,128,64,1024)                              |
| upBlockModule        | (1,128,64,1024)->(1,256,64,1024)                             |
| upBlockModule        | (1,256,64,1024)->(1,512,64,1024)                             |
| upBlockModule        | (1,512,64,1024)->(1,1024,64,1024                             |
| Conv2D+relu          | (1,1,16,1024)->(1,64,16,1024)<br />input_channel: 1<br />output_channel: 64<br />stride=(2,1)<br />kernel_size=(3,3) |

- 两次转置矩阵（让图像放大）
- 5次下采样卷积操作
- 4次上采样操作



### 明确损失函数

### 明确优化策略



# 实例

### 医疗图像分类

|      |                               |      |
| ---- | ----------------------------- | ---- |
| 输入 | n张exam后的图片               |      |
| 输出 | 良性肿瘤的概率/恶性肿瘤的肿瘤 |      |
|      |                               |      |





# 防止模型过拟合

过拟合：在当前数据集上有较好的效果，但在其他数据集上效果较差。就好像一个死记硬背的学生，不会灵活地做题。

## 正则化

基于奥卡剃须刀原理（如无必要，勿增实体，在效果同样的基础上，只需参数最少的模型），对模型的参数进行约束
