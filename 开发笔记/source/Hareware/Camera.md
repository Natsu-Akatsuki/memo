# Camera

## Q&A

### 深度相机的几种类型？

**结构光相机**：基于光信息变化量（如相位信息）得到深度信息

**立体视觉相机**：基于三角法得到深度信息

**ToF相机**：基于可见光的飞行时间得到深度信息

.. note:: realsensed435i是采用立体视觉的方法，它的红外发射器只是为了发射红外散斑（提供红外光噪声，以丰富特征点的个数）

- 相机

|                型号                |          描述          |
| :--------------------------------: | :--------------------: |
|          realsense d435i           |       已不再生产       |
| [zed](https://www.stereolabs.com/) | 需要配合含nvidia的设备 |

## Realsense

### Driver

#### [源码安装Realsense](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)

当内核版本非官方直接支持的版本时或者相机输出不太稳定时则建议使用源码编译。或者使用RSUSB backend(但实测在5.13，没有希望的效果)

- 为什么强烈建议使用patch：

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/FeCwfRwDKLBg4kON.png!thumbnail)

- 现支持的版本：

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/AC4sXLKg96fADiPY.png!thumbnail" alt="img" style="zoom:67%;" />

- [有关支持内核5.11的PR说明](https://github.com/IntelRealSense/librealsense/pull/9727)（目前最稳定和有效的是5.11.34，再高就不行了）
- 有关[RSUSB backend](https://github.com/IntelRealSense/librealsense/issues/10306)：在5.13测试，依然会有数据丢失的问题

依然会有

- 大概安装的过程：

```bash
$ git clone https://github.com/IntelRealSense/librealsense.git --depth=1

# 构建特定内核下的内核模块
$ cd librealsense 
$ bash ./scripts/patch-realsense-ubuntu-lts.sh

$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release  ..
$ make -j4
$ sudo make install

# 添加udev
$ ./scripts/setup_udev_rules.sh

# test:
$ realsense-viewer
```

#### 自定义补丁

需要修正相关的文件，尝试修正5.13版本的驱动模块（以为可以无损patch），但相关的API有一些差异，感觉改起来比较耗时，坐等官方提供支持了（告辞.jpg）

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220324104535573.png" alt="image-20220324104535573" style="zoom:67%;" />

#### Q&A

- ubuntu18.04/20.04等版本容易出现数据丢失或读取不到元数据的问题，Sometimes "rs2_pipeline_wait_for_frames - pipe:0x2326020 :Frame didn't arrive within 5000" occurred after running normally several hours. then can't get any device by calling ctx.query_devices（）even though I reboot the computer。或者：

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/Ubik2ySGaJfFRChA.png!thumbnail)

或者：(ds5-timestamp.cpp:69) UVC metadata payloads not available. Please refer to the installation chapter for details；或者：HID timestamp not found, switching to Host timestamps；官方说明：

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/OMexwVLPme42VIqT.png!thumbnail" alt="img" style="zoom:67%;" />

- "failed to open usb interface: 0, error: RS2_USB_STATUS_ACCESS"

[没有成功添加udev文件](https://github.com/IntelRealSense/realsense-ros/issues/1408)

```bash
$ git clone https://github.com/IntelRealSense/librealsense.git --depth=1
$ cd librealsense
$ ./scripts/setup_udev_rules.sh
```

- ros包没有IMU数据输出

 没有/camera/imu数据进行发布，在使能陀螺仪和重力加速度计后默认是分别发布这两个主题；将它们合成为一个topic的话则需要设置：

```xml
<!-- 或者copy -->
unite_imu_method:="linear_interpolation"
```

- d435i有多少摄像头？

一对红外摄像头、一个RGB相机、一个**红外发射器**

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/QSLj0qWun6t5Rwnu.png!thumbnail" alt="img" style="zoom: 50%;" />

- 相机的类型？

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220401193712141.png" alt="image-20220401193712141" style="zoom: 50%;" />

- 相机的位置？

![image-20220401193845152](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220401193845152.png)

- 红外发射器的作用？

新版本使用枚举变量来表示开/关：0(false)

```xml
<rosparam>
 /camera/stereo_module/emitter_enabled: 0
</rosparam>
```

提供红外散斑，以更好地恢复深度信息，不开IR：

![image-20220401193108781](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220401193108781.png)

开IR：

![image-20220401193122997](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220401193122997.png)

- 获取相机内外参TF信息：

```bash
$ rs-sensor-control

16 : Infrared #1 (Video Stream: Y8 640x480@ 30Hz)
52 : Infrared #2 (Video Stream: Y8 640x480@ 30Hz)
0  : Accel #0


0->16
Translation Vector : [0.00552,-0.0051,-0.01174]
Rotation Matrix    : [1,0,0]
                   : [0,1,0]
                   : [0,0,1]

0->52
Translation Vector : [-0.0444489,-0.0051,-0.01174]
Rotation Matrix    : [1,0,0]
                   : [0,1,0]
                   : [0,0,1]

# 或者（读IMU->红外的外参）
$ rs-enumerate-devices -c | grep -A 6 'Extrinsic from "Accel"' | grep -A 6 "Infrared"
# 读红外的内参
$ rs-enumerate-devices -c | grep -A 8 "Intrinsic" | grep -A 8 "Infrared" | grep -A 8 "640x480"
```

![image-20220405153747888](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220405153747888.png)

.. note:: 可用于填写vins的配置文档

```yaml
body_T_cam0: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [1, 0, 0, 0.0052, 0, 1 , 0, -0.0051, 0, 0, 1, -0.01174, 0, 0, 0, 1]

body_T_cam1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [ 1, 0, 0, -0.0444489, 0, 1, 0, -0.0051, 0, 0, 1, -0.01174, 0, 0, 0, 1]
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220405160236745.png" alt="image-20220405160236745" style="zoom:50%;" />

#### 拓展资料

- [3种深度相机 realsense 官方](https://www.intelrealsense.com/beginners-guide-to-depth/)
