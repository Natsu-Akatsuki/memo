# EnvSetup

cudnn和TensorRT的tar包下载页面需要使用NVIDIA账号登入

## nvidia-driver

- 为了节省功耗和兼顾性能，nvidia设计了`optimus`混合独显和集显的技术方案，使用独显进行计算，独显进行显示
- optimus有如下几种模式：（1）仅使用集显（2）仅使用独显（3）同时使用集显和独显（只有在用到N卡时才使用N卡）
- 使用显卡不一定能够实现加速，因为将数据从内存搬到显存是需要时间的

### 安装

- 方案一：基于GUI

> An alternate method of installing the NVIDIA driver was detected. (This is usually a package provided by your distributor.) A driver installed via that method may integrate better with your system than a driver installed by nvidia-installer. Please review the message provided by the maintainer of this alternate installation method and decide how to proceed: The NVIDIA driver provided by Ubuntu can be installed by launching the "Software & Updates" application, and by selecting the NVIDIA driver from the "Additional Drivers" tab.

即使用ubuntu开发者提供的驱动包，会有更好的兼容性

- 方案二：用apt安装显卡驱动

本部分等价于在gui来安装显卡驱动

步骤一：查看可安装的驱动

.. note:: 在ubuntu16中执行时或返回空值，则可以使用 `官网安装包 <https://www.nvidia.cn/Download/index.aspx?lang=cn>`_\  进行安装

```bash
$ sudo apt update
# 查看能用的驱动版本
$ sudo ubuntu-drivers devices  
# 如果返回空值，则这种方法无效，则需要到官网上进行
```

步骤二：选择相关版本并安装，例如：

```bash
$ sudo apt-get install nvidia-driver-450
```

步骤三：验证（有时需要重启后才能生效）

```bash
$ nvidia-smi
```

- 方案三：官网安装包下载

步骤一：在[官网](https://www.nvidia.cn/Download/index.aspx?lang=cn)选择适合的driver安装包进行下载

步骤二：安装安装包和执行

```bash
# 安装一些相关依赖，否则会有warning
$ sudo apt install pkg-config libglvnd-dev
# 切换至非图形化界面
$ sudo systemctl isolate multi-user.target
# 如显示nvidia-drm正在使用，则关闭该内核模块
$ sudo modprobe -r nvidia-drm
```

步骤三：验证

```bash
$ nvidia-smi
```

### 卸载

```bash
# --- 方法一（适用于用安装包安装的）
$ nvidia-uninstall
# --- 方法二（适用于用apt安装）
$ sudo apt purge nvidia-driver-*
$ sudo apt autoremove
```

### 解决方案

#### 显卡库版本和驱动版本不同步的问题(preview)

```bash
# 显卡库版本更新但显卡驱动版本没有同步更新时会显示：
$ nvidia-smi
# Failed to initialize NVML: Driver/library version mismatch

# 显示显卡驱动内核版本的指令
$ cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  455.38  Thu Oct 22 06:06:59 UTC 2020
GCC version:  gcc version 7.5.0 (Ubuntu 7.5.0-3ubuntu1~18.04)

# 一般可以选择重装，若有DKMS时可尝试重启
```

- DKMS

> Would you like to register the kernel module souces with DKMS? This will allow DKMS to automatically build a new module, if you install a different kernel later?

此处选择yes，当内核更新时，显卡驱动也会进行更新，而不用自己再手动去升级了

#### 解决nouveau冲突问题

- `nouveau` （开源，但功能非常少）和 `nvidia driver` 都是nvidia的显卡驱动。部分计算机默认使用 `nouveau` 作为驱动，那么在这些机子上装N卡官网驱动时，就有冲突的问题，需要[先关闭nouveau模块](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-nouveau-ubuntu)（记得update）。

- 部分版本（如460）可以在安装时，提供一个选项，替我们完成这一步（相关文件为`/etc/modprobe.d/nvidia-installer-disable-nouveau.conf` 文件）

```bash
# 可用该指定判断当前系统有无nouveau模块
$ lsmod | grep nou
```

#### 重新使用nouveau

- 取消掉`/usr/lib/modprobe.d`或者`/etc/modprobe.d`中屏蔽nouveau的配置即可

.. attention:: 注意这两个位置都可能有

- 更新内核配置

```bash
$ sudo update-initramfs -u
```

#### [5.14内核下用安装包安装驱动有问题](https://bbs.archlinux.org/viewtopic.php?id=268421)

使用NVIDIA-Linux-x86_64-470.57.02安装时会出现如链接上的报错，`error: ‘struct task_struct’ has no member named ‘state’; did you mean ‘__state’?`；从470.74开始该BUG已修复，安装更新的显卡驱动即可。

#### 无法调节亮度

- 内核版本为5.14，已安装显卡驱动，原先只启动了独显

步骤一：查看有无使用集显

```bash
$ sudo lshw -c display
# 发现只使用了独显，从bios中设置混合模式
```

[启动后发现电脑黑屏](https://forums.developer.nvidia.com/t/rtx3070-laptop-gpu-on-ubuntu-20-04-doesnt-work-properly-with-amd-ryzen-7-5800h/168148/3)，让电脑自动生成X配置文档

![image-20211101225228174](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211101225228174.png)

.. note:: 实测中，ubuntu16.04 4.15内核，如上设置无效

#### 30系gpu频闪与卡顿

- 垂直同步不生效（驱动460）

```bash
# 无反应，无提示语Running synchronized to the vertical refresh. The framerate should be
# approximately the same as the monitor refresh rate.
$ __GL_SYNC_TO_VBLANK=1 glxgears
```

- 安装最新版的470驱动即解决问题

#### 显卡模式切换

- 命令行

```bash
$ prime-select --help
# Usage: /usr/bin/prime-select nvidia|intel|on-demand|query
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/pSNh11oF66LqVSLi.png!thumbnail)

- GUI

```bash
$ nvidia-settings
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/Zmnd1WPPg7uTVZPF.png!thumbnail)

#### 查看显卡信息(for KDE)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/if2ZYzZUgGLlLsH3.png!thumbnail)

#### 混合模式(optimus)下，指定某个程序用独显

```bash
$ __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia <command>
```

.. attention:: 需要混合模式下，才能生效

#### opengl

- 查看opengl相关信息（命令行）

```bash
$ glxinfo
```

- 查看opengl相关信息（for KDE），直接查询即可找到

![image-20211129014323532](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211129014323532.png)

### [KMS](https://wiki.archlinux.org/title/Kernel_mode_setting)

KMS使能内核区(kernel space)设置分辨率和颜色深度，而不是在用户区，其能够使framebuffer有更好的可视化效果和实现tty的快速切换

#### 禁用KMS

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/gyJEMxYgqOVPxrc4.png!thumbnail)

## 集显

- 查看inter gpu使用情况

```bash
$ sudo intel_gpu_top
```

![image-20211129013232309](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211129013232309.png)

- [查看amd gpu使用情况](https://linuxhint.com/apps-monitor-amd-gpu-linux/)

```bash
$ sudo apt install radeontop
# c means color
$ radeontop -c
```

## [cudnn](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

### 安装

步骤一：[tar包下载](https://developer.nvidia.com/cudnn)

步骤二：解压与赋值

```bash
$ tar -xzvf cudnn-x.x-linux-x64-v8.x.x.x.tgz
$ sudo cp cuda/include/cudnn*.h /usr/local/cuda/include \
&& sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 \
&& sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
# -P 表示保留权限属性地复制
```

## cuda

.. attention:: 20.04ubuntu对应cuda11+的版本

### 安装

步骤一：[run包下载与安装](https://developer.nvidia.com/cuda-toolkit-archive)，[e.g. cuda11.2.2](https://developer.nvidia.com/cuda-11.2.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=runfilelocal)

步骤二：创建软链接和导入环境变量

```bash
# 用runfile装cuda11+，会自动创建软链接
$ export PATH=$PATH:/usr/local/cuda/bin
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

### 卸载

```bash
# 执行安装目录的bin文件夹下的
$ ./cuda-uninstaller
```

## [TensorRT](https://developer.nvidia.com/tensorrt)

### [安装](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

步骤一：查看相关依赖是否满足（已测试）

| TensorRT 版本    | cuda版本        | cudnn版本   |
| ---------------- | --------------- | ----------- |
| tensorRT 7.0.0   | cuda_10.02      | cudnn 7.6.5 |
| tensorRT 7.2.3   | cuda_11.1       | cudnn 8.1.0 |
| tensorRT 8.0.0.3 | cuda_11.2.r11.2 | cudnn 8.1.1 |
| tensorRT 8.2.2.1 |                 | cudnn 8.2.1 |
| tensorRT 8.2.3.0 | cuda_11.4.r11.4 | cudnn 8.2.4 |

步骤二：

[tar包下载与安装](https://developer.nvidia.com/tensorrt)：更灵活的安装方式，可灵活地切换版本，不需要很严格的版本对应(e.g. cuda/cudnn)

Debian下载：这种下载方式需要解决的依赖问题挺多的，较麻烦的，e.g.：

![image-20220121015313916](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220121015313916.png)

步骤三：导入动态库位置

```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"install_path/lib"
```

---

**NOTE**

- [使用wget来下载tensorrt tar包或deb包](https://forums.developer.nvidia.com/t/download-cudnn-via-wget-or-curl/48952/5)：找到带auth token的重定向链接

![image-20220121020150604](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220121020150604.png)

---
