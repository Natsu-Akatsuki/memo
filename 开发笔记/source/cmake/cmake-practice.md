# cmake-practice

<p align="right">Author: kuzen, Natsu-Akatsuki</p>

## cmake

### 安装

方法一： `apt` 下载

```bash
# linux 18.04对应3.10版本
$ sudo apt-get install cmake
```

方法二：[源码下载](https://cmake.org/download/)

```bash
# e.g.
$ wget https://github.com/Kitware/CMake/releases/download/v3.18.3/cmake-3.18.3.tar.gz 
# 要安装cmake-qt-gui时需要添加如下option
$ ./bootstrap --qt-gui
```

### 使用conda环境的cmake文件

在conda环境安装了相关包之后，需要conda activate才能使用其cmake文件，如果不activate的话，需要类似如下类型的参数配置

```bash
# 以pybind11为例 
-Dpybind11_DIR=${env_path}/share/cmake/pybind11`
```

### 逻辑判断

判断一个路径对应的是否是一个文件夹

```bash
if(IS_DIRECTORY "...")
```

### 处理可执行文件

* [find_program](https://cmake.org/cmake/help/latest/command/find_program.html)：类似which，找到某个可执行文件的路径
* execute_process：执行某个可执行文件

```bash
# 判断某个可执行文件是否存在
find_program(GDOWN_AVAIL "gdown")
if (NOT GDOWN_AVAIL)
  message("...")
endif()

# 执行某个可执行文件
execute_process(COMMAND mkdir [args...])
execute_process(COMMAND gdown [args...])
```

### 引入外部项目

### [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)

该command为3.11的特性，会在configure time时导入(pollute)文件

```cmake
cmake_minimum_required(VERSION 3.14)

# 导入FetchContent module
include(FetchContent)
# 配置等下Fetch时的配置参数
FetchContent_Declare(
  mycom_toolchains
  URL  https://intranet.mycompany.com//toolchains_1.3.2.tar.gz
)
# 触发下载(Fetch)
FetchContent_MakeAvailable(mycom_toolchains)
```

.. hint:: 对于cmake，configure time是指生成cache文件的时间段；有三种time，分别是配置期(configure time)，编译期(build time)和安装期(install time)；配置期的命令包括add_subdirectory, include, file

.. todo:: 暂未清楚不同期导入文件所带来的结果

## catkin_make

### 单独编译某些package

```bash
$ catkin_make -DCATKIN_WHITELIST_PACKAGES="package1;package2"
# 等价于：
$ catkin_make --only-pkg-with-deps
# 撤销白名单设置
$ catkin_make -DCATKIN_WHITELIST_PACKAGES=""
```

.. note:: 要屏蔽某些包被编译，可以创建一个名为 `CATKIN_IGNORE <https://github.com/tier4/velodyne_vls/tree/tier4/master/velodyne_msgs>` _ 的文件到这些包所在的目录下

### 使用ninja编译

```bash
$ catkin_make --use-ninja
```

.. note:: catkin_make用ninja编译速度会快些，但对报错信息没有语法高亮，很影响调试

## [catkin build](https://catkin-tools.readthedocs.io/en/latest/index.html)

### [安装catkin build](https://catkin-tools.readthedocs.io/en/latest/installing.html)

### 编译

* 跳过对某些已编译包的编译（实际上只是检查）

```bash
$ catkin build --start-with <pkg>
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/dIW8tcn1J6m2KYLp.png!thumbnail" alt="img" style="zoom:67%; " />

* 配置编译参数

```bash
$ catkin config -DPYTHON_EXECUTABLE=/opt/conda/bin/python3 \
-DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.8 \
-DPYTHON_LIBRARY=/opt/conda/lib/libpython3.8.so
# 使用catkin_make参数
$ catkin config --catkin-make-args [args]
```

* 配置黑白名单

```bash
# 配置白名单（或黑名单）
$ catkin config --whitelist/blacklist <pkg>
# 取消白名单配置
$ catkin config --no-whitelist  
```

* 追加和移除而非覆盖配置参数

```bash
# 追加配置参数
$ catkin config -a <配置参数>
# 移除配置参数
$ catkin config -r <配置参数>
```

* 编译当前所处的`package`

```bash
$ catkin build --this
```

* [缓存Environment来提高编译速度](https://catkin-tools.readthedocs.io/en/latest/verbs/catkin_config.html?highlight=cache#accelerated-building-with-environment-caching)

```bash
$ catkin config/build --env-cache
$ catkin config/build --no_env_cache
```

.. todo:: 暂未比较过编译时间的差别

### 清理编译产物

```bash
# 指定删除某个package
$ catkin clean <package_name>
# 删除所有 product 
$ catkin clean --deinit
# 移除非src文件夹下的包的编译产物 
$ catkin clean --orphans
```

.. note:: catkin clean 默认删除 devel, log等目录，但隐藏目录 .catkin_tools , .catkin_workspace不会清除

### [配置文档](https://catkin-tools.readthedocs.io/en/latest/verbs/catkin_profile.html)

catkin build可以设置配置文档profile

.. todo:: 尚未明晰可用的场景

### [deploy a catkin package](https://answers.ros.org/question/226581/deploying-a-catkin-package/)

## DEBUG

### 使用catkin builld编译时显示could not find a package configuration file

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210912141918386.png" alt="image-20210912141918386" style="zoom: 80%; " />

一般来说catkin build不用像catkin_make一样，需要在cmakelists中指明依赖关系，其能够合理地安排编译顺序，会出现上述问题可检查一波 `package.xml` 是否写好了build tag

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/AYu9WKlHPlES5yu7.png!thumbnail" alt="img" style="zoom:67%; " />

### /usr/bin/ld: cannot find -l

* 在使用TensorRT部署时出现如下的一些报错

```bash
/usr/bin/ld: cannot find -lnvonnxparser
/usr/bin/ld: cannot find -lnvinfer_plugin 
/usr/bin/ld: cannot find -lcudnn
```

一种解决方案为使用环境变量 `LIBRARY_PATH` 。此前认为时需要修改环境变量 `LD_LIBRARY_PATH` ，添加动态库链接搜索路径，但实际上该环境变量，只影响运行期(runtime)链接器 `ld.so` 的搜索路径。而不影响编译期(complie time)链接器 `/usr/bin/ld` 的搜索路径。要影响编译期链接的话，需要修改环境变量 `LIBRARY_PATH`

```bash
env LIBRARY_PATH=/usr/local/cuda/lib64:${HOME}/application/TensorRT-8.0.0.3/lib make
```

另一种解决方案为在CMakeLists上增设：

```cmake
# e.g.
link_directories(/usr/local/cuda/lib64/ $ENV{HOME}/application/TensorRT-8.0.0.3/lib)
```

* 拓展资料

  * [ld和ld.so命令的区别](https://blog.csdn.net/jslove1997/article/details/108033399)
  * [stackoverflow answer](https://stackoverflow.com/questions/61016108/collect2-error-ld-returned-1-exit-status-lcudnn)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/U9PWBBMXKy4vBo31.png!thumbnail)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/FvUyBNAT1nHvGPiG.png!thumbnail)

* [查找动态链接库的顺序 for runtime](https://man7.org/linux/man-pages/man8/ld.so.8.html)

### No CMAKE_CXX_COMPILER could be find

```bash
sudo apt install build-essential
```

### 未定义的引用（undefined reference）

这错错误发生在链接时期。一般来说有以下几种情况。一种是没下载相关的链接库（可locate看一下）；一种是库的冲突，比如ros的opencv库与从源码编译安装到系统的opencv库发生冲突，至依赖被覆盖而使目标文件无法成功链接到库。可卸载安装到系统的opencv库（如用sudo make uninstall来卸载）；一种是已下载但没找到，添加相关搜素路径即可

### imported target \"\...\" references the file \"\...\" but this file does not exist

[locate 定位相关位置后，使用软链接](https://blog.csdn.net/weixin_45617478/article/details/104513572)

### no such file or directory：没有找到头文件的路径，导入头文件失败

在已有头文件的情况下，可直接添加绝对路径进行搜索；[或者头文件名不对，进行修改即可](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM/issues/219)

```cmake
# e.g. include/utility.h:13:10: fatal error: opencv2/cv.h: No such file or directory #include <opencv2/cv.h>
include_directories(
   include
   绝对路径   # e.g. /home/helios/include
)
```

### 目标文件命名冲突(for catkin)

rslidar和velodyne package的目标文件重名

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/M5KhRzVvmtcWapDQ.png!thumbnail)

### 找不到cuda库和tensorrt库相关文件

在autoware中，使用有关深度学习的cmake时，不能直接通过find_package找到cuda库和tensorRT；autoware配置环境时是使用deb包来安装的，会随带着将cmake等文件也安装到系统路径中；而如果使用的是local的安装方式，则find_package失效时，可参考如下方法进行添加：

```bash
include_directories($ENV{HOME}/application/TensorRT-7.2.3.4/include/) link_directories($ENV{HOME}/application/TensorRT-7.2.3.4/lib)`
````

### [ROS中编译通过但是遇到可执行文件找不到的问题](https://blog.csdn.net/u014157968/article/details/86516797)：指令顺序的重要性

* catkin_package要放在add_executable前，[案例（松灵底盘）](https://github.com/agilexrobotics/agx_sdk/issues/1)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/BdZu0UoMbhAAPawe.png!thumbnail" alt="img" style="zoom:50%; " />

* [为什么有些情况即使顺序不对，catkin_make也能编译成功？](https://jbohren-ct.readthedocs.io/en/pre-0.4.0-docs/migration.html)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/0EA9e6jBjsZnVsIF.png!thumbnail" alt="img" style="zoom:67%; " />

### opencv库兼容性问题

* 不同版本的opencv库或有功能相同但名字不同的问题，在编译时可能会出现未声明等报错，这时候就需要查文档就行修改。

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/Sz3d8VYj2wt2TNqb.png!thumbnail" alt="img" style="zoom:50%; " />

实例：[kalibr 16.04/14.04](https://github.com/ethz-asl/kalibr) -> [kalibr 20.04](https://github.com/ori-drs/kalibr)

* CheckLists

| 16.04(apt version)                | 20.04(apt version 4.2) |
| --------------------------------- | ---------------------- |
| CV_LOAD_IMAGE_COLOR (icv::imread) | cv:: IMREAD_COLOR       |

* 一般来说可以尝试先将`CV_`转化为`cv::`来进行替换

### boost库的升级换代

* 有关模块

![image-20210918004819514](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210918004819514.png)

![image-20210918005720515](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210918005720515.png)

* 有关函数

```c++
// for 16.04
boost::this_thread::sleep(boost::chrono::microseconds(SmallIterval)); 
// for 20.04
std::this_thread::sleep_for(std::chrono::microseconds(SmallIterval)); 

```

.. note:: 在编译时有些函数不存在，可能是因为更新换代而被取代了，这时候查一下google和相关文档即可

### ambigious candidate

> Reference to 'shared_ptr' is ambiguous candidate found by name lookup is 'boost::shared_ptr' candidate found by name lookup is 'pcl::shared_ptr'

pcl库和boost都有自己的share_ptr实现，而[源程序](https://github.com/fverdoja/Fast-3D-Pointcloud-Segmentation)使用了using这种方法，使得编译器不知道该调用哪个share_ptr

```c++
using namespace boost;
using namespace pcl;

void removeText(shared_ptr<visualization::PCLVisualizer> viewer); // ERROR
void removeText(pcl::shared_ptr<visualization::PCLVisualizer> viewer); // TRUE
```

## 拓展工具

### [catkin-lint](https://fkie.github.io/catkin_lint/)

静态查看catkin工程错误

```bash
# 安装
$ sudo apt install catkin-lint
# example
$ catkin_lint -W0 .
```

![image-20210912200754563](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210912200754563.png)

.. note:: catkin_lint相关提示信息仅供参考，不一定准确

### [ccmake](https://cmake.org/cmake/help/latest/manual/ccmake.1.html)

cmake TUI程序，在**终端**交互式地配置选项

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210925215521631.png" alt="image-20210925215521631" style="zoom:67%; " />

### cmake-gui

cmake GUI程序，在**图形化界面**交互式地配置选项
