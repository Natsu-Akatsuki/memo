.. role:: raw-html-m2r(raw)
   :format: html


Driver
======

Realsense
---------

Realsense数据丢失问题
^^^^^^^^^^^^^^^^^^^^^


* ubuntu18.04/20.04等版本容易出现数据丢失或读取不到元数据的问题，Sometimes "rs2_pipeline_wait_for_frames - pipe:0x2326020 :Frame didn't arrive within 5000" occurred after running normally several hours. then can't get any device by calling ctx.query_devices（）even though I reboot the computer。或者：


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/Ubik2ySGaJfFRChA.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/Ubik2ySGaJfFRChA.png!thumbnail
   :alt: img


或者：(ds5-timestamp.cpp:69) UVC metadata payloads not available. Please refer to the installation chapter for details.


* 官方说明

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/OMexwVLPme42VIqT.png!thumbnail" alt="img" style="zoom:67%;" />`

`源码安装Realsense <https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当内核版本非官方直接支持的版本时或者相机输出不太稳定时则建议使用源码编译。或者使用RSUSB backend(但实测在5.13，没有希望的效果)


* 为什么强烈建议使用patch：


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/FeCwfRwDKLBg4kON.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/FeCwfRwDKLBg4kON.png!thumbnail
   :alt: img



* 现支持的版本：

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/AC4sXLKg96fADiPY.png!thumbnail" alt="img" style="zoom:67%;" />`


* `有关支持内核5.11的PR说明 <https://github.com/IntelRealSense/librealsense/pull/9727>`_\ （目前最稳定和有效的是5.11.34，再高就不行了）
* 有关\ `RSUSB backend <https://github.com/IntelRealSense/librealsense/issues/10306>`_\ ：在5.13测试，依然会有数据丢失的问题

依然会有


* 大概安装的过程：

.. prompt:: bash $,# auto

   $ git clone https://github.com/IntelRealSense/librealsense.git --depth=1

   # 构建特定内核下的内核模块
   $ cd librealsense ./scripts/patch-realsense-ubuntu-lts.sh

   $ mkdir build && cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release  ..
   $ make -j4
   $ sudo make install

   # 添加udev
   $ ./scripts/setup_udev_rules.sh

   # test:
   $ realsense-viewer

自定义补丁
^^^^^^^^^^

需要修正相关的文件，尝试修正5.13版本的驱动模块（以为可以无损patch），但相关的API有一些差异，感觉改起来比较耗时，坐等官方提供支持了（告辞.jpg）

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220324104535573.png" alt="image-20220324104535573" style="zoom:67%;" />`

ROS realsense
^^^^^^^^^^^^^

没有IMU数据输出
~~~~~~~~~~~~~~~

 没有/camera/imu数据进行发布，在使能陀螺仪和重力加速度计后默认是分别发布这两个主题；将它们合成为一个topic的话则需要设置：

.. code-block:: xml

   <!-- 或者copy -->
   unite_imu_method:="linear_interpolation"
