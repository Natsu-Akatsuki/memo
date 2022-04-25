# RosPractice

## [基于ros环境导入某个package下的python包](https://roboticsbackend.com/ros-import-python-module-from-another-package/)

### 简例

步骤一：创建python模板，相应的目录树如下

```plain
└── directory_name        # 一般可以设置为package_name
    ├── CMakeLists.txt
    ├── package.xml
    ├── setup.py
    └── src
        └── module_name      # 一般设置为module_name
            ├── import_me_if_you_can.py
            └── __init__.py
```

步骤二：编写 `setup.py` 文件

```python

## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['module_name'],
    package_dir={'': 'src'},
    requires=['rospy']
)

setup(**setup_args)
```

步骤三：编写 `CMakeLists.txt` 文件

```cmake
cmake_minimum_required(VERSION 3.13)
project(project_name)
find_package(catkin REQUIRED COMPONENTS rospy)
# 调用当前CMakeLists文件所在目录下的setup.py
catkin_python_setup()   
catkin_package()      
```

## 构建元包

有的时候可能需要catkin build多个包，相应的命令行可能为`catkin build pkg1 pkg2 pkg3`。使用元包则可以简化命令行，其将多个包进行统一管理在一起，最后只需要使用命令行`catkin build meat_pkg`。

### 参考教程

* 方案一：[answer.ros.org](https://answers.ros.org/question/322340/creating-metapackage/)

```bash
catkin_create_pkg <MY_META_PACKAGE> --meta
```

* 方案二：[ros-wiki教程](http://wiki.ros.org/catkin/package.xml#Metapackages)

直接添加一些细节到 `CMakeLists` 和 `package.xml` 上

## 使用nodelet

ros节点的通信是进程的通信，采用ros tcp的方法。当节点间传输的数据体量较大，通信（比如要反序列和序列化）的开销将比较大。因此若**希望减少节点间通讯的开销来提高实时性**，这就需要用到nodelet技术。具体例子，比如跑一些点云的预处理模块，涉及到采集的点云转换成ros数据，点云滤波去离群点，点云裁剪，点云去地面等节点，这些节点允许放在一个进程作为单独的线程去跑（ `ros nodelet` 程序能将一个 `node` 进程节点转换为 `nodelet` 线程节点），然后想提高实时性就可以用到。总体给人一种将进程通信转换为线程通信的感觉。

### [显示当前系统的nodelet和其描述文件](http://wiki.ros.org/nodelet#Helper_tools)

```bash
rospack plugins --attrib=plugin nodelet   # 显示.xml文件
rosrun nodelet declared_nodelets          # 显示ros工作空间中已有的nodelet
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210810223516109.png" alt="image-20210810223516109" style="zoom:67%; " />

### plugin相关文档解析

nodelet与plugin密切相关，其中ros中的插件([plugin]([pluginlib](http://wiki.ros.org/pluginlib/Tutorials/Writing%20and%20Using%20a%20Simple%20Plugin)))即动态库中的可动态加载的类

![image-20210811003457276](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210811003457276.png)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210515175724200.png" alt="image-20210515175724200" style="zoom:67%; " />

### nodelet的启动

步骤一：一般而言，每个nodelet需要一个NodeletManager来管理，启动NodeletManager的方法

```bash
# 同时重命名NodeletManager
$ rosrun nodelet nodelet manager __name:=nodelet_manager 
```

等价于在launch文档中的：

```xml
<node pkg="nodelet" type="nodelet" name="nodelet_manager" args="manager"/>
```

步骤二：加载nodelet到NodeletManager

```bash
# Launch a nodelet of type pkg/type(包名/xml文件中的class name) on manager manager
$ rosrun nodelet nodelet load nodelet_tutorial_math/Plus nodelet_manager
```

等价于在launch文档中的：

```xml
<node pkg="nodelet" type="nodelet" name="Plus" args="load nodelet_tutorial_math/Plus nodelet_manager"/>
```

### nodelet launch文档解读

```xml
<!--都需要启动nodelet包的nodelet可执行文件，不过相应的启动参数不一样-->
<node pkg="nodelet" type="nodelet" name="euclidean_cluster_manager" args="manager" output="screen" />

<node pkg="nodelet" type="nodelet" name="$(anon voxel_grid_filter)" args="load pcl/VoxelGrid euclidean_cluster_manager" output="screen"> 
</node>

<node pkg="nodelet" type="nodelet" name="$(anon euclidean_cluster)" args="load euclidean_cluster/voxel_grid_based_euclidean_cluster_nodelet euclidean_cluster_manager" output="screen">
</node>

<!--standalone nodelet，不需要加载到nodelet manager，相关于启动一个普通node-->
<node pkg="nodelet" type="nodelet" name="Plus3" args="standalone nodelet_tutorial_math/Plus">
</node>
```

### 参考教程

* [ros.wiki官方教程](http://wiki.ros.org/nodelet)

需要对虚函数进行重载，所有有关ros的api需要在该部分进行初始化

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210810224100470.png" alt="image-20210810224100470" style="zoom:67%; " />

形如：

```c++
void EuclideanClusterNodelet::onInit()
{
  // Get the private node handle (provides this nodelets custom remappings in its private namespace)
  private_nh_ = getPrivateNodeHandle(); 

  private_nh_.param<std::string>("target_frame", target_frame_, "base_link"); 
  private_nh_.param<bool>("use_height", use_height_, false); 
  private_nh_.param<int>("min_cluster_size", min_cluster_size_, 3); 
  private_nh_.param<int>("max_cluster_size", max_cluster_size_, 200); 
  private_nh_.param<float>("tolerance", tolerance_, 1.0); 

  nh_ = getNodeHandle(); 
  pointcloud_sub_ =

    private_nh_.subscribe("input", 1, &EuclideanClusterNodelet::pointcloudCallback, this);

  cluster_pub_ =

    private_nh_.advertise<autoware_perception_msgs::DynamicObjectWithFeatureArray>("output", 10);

  debug_pub_ = private_nh_.advertise<sensor_msgs:: PointCloud2>("debug/clusters", 1); 
}

```

* [nodelet code template](https://www.clearpathrobotics.com/assets/guides/kinetic/ros/Nodelet%20Everything.html)

## 函数解读

```bash
// 专属于nodelet的日志输出
NODELET_INFO("Initializing nodelet TemplatePackageNodelet...");
```

### ros自带的nodelet

```plain
...
pcl/PassThrough
pcl/VoxelGrid
pcl/ProjectInliers
pcl/ExtractIndices
pcl/StatisticalOutlierRemoval
pcl/RadiusOutlierRemoval
pcl/CropBox
pcl/NodeletMUX
pcl/NodeletDEMUX
pcl/PCDReader
pcl/BAGReader
...
```

## 回调函数

### 同时接收多个数据进行处理

```c++
// 导入相关头文件
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"

// 声明回调函数
void objectsCallback(
    const autoware_perception_msgs::DynamicObjectWithFeatureArray::ConstPtr & input_object0_msg,
 const autoware_perception_msgs::DynamicObjectWithFeatureArray::ConstPtr & input_object1_msg);

// 定义订阅器
message_filters::Subscriber<autoware_perception_msgs::DynamicObjectWithFeatureArray> object0_sub_(pnh_, "input/object0", 1);
message_filters::Subscriber<autoware_perception_msgs::DynamicObjectWithFeatureArray> object1_sub_(pnh_, "input/object1", 1);

// 定义同步机制
typedef message_filters::sync_policies::ApproximateTime<   
    autoware_perception_msgs::DynamicObjectWithFeatureArray, autoware_perception_msgs::DynamicObjectWithFeatureArray>
    SyncPolicy;

typedef message_filters::Synchronizer<SyncPolicy> Sync;
Sync sync_;
```

### Ptr类型

> [sensor_msgs::ImageConstPtr](https://docs.ros.org/en/diamondback/api/sensor_msgs/html/namespaces.html)是什么类型数据？

* 共享指针

![image-20220206215812571](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220206215812571.png)

实际可以如此使用：

```c++
void TensorrtYolo::callback(const sensor_msgs::Image::Ptr& in_image_msg)
void TensorrtYolo::callback(const sensor_msgs::Image::Ptr in_image_msg)
```

## [使用gdb调试launch中的节点](http://wiki.ros.org/roslaunch/Tutorials/Roslaunch%20Nodes%20in%20Valgrind%20or%20GDB)

核心为使用gdb -p

步骤一：修改 `CmakeLists` 的build type

```cmake
SET(CMAKE_BUILD_TYPE "Debug")
```

步骤二：gdb对应的进程

```bash
sudo gdb -p <pid_id>
```

## ros命名空间

* cpp程序

```c++
// c++中描述的节点名不包含命名空间，无'/'(e.g即没有/.../...，只有...)
ros::init(argc, argv, "节点名"); 
```

* launch文件

```xml
<!-- launch中的节点名也不包含命名空间 -->
<node pkg="talker" type="talker" name="talker"/>
<!-- 可以通过group tag或ns attribute来添加命名空间 -->
<!-- 前者可同时给多个节点附上一个命名空间，后者则针对具体的一个，前者可被后者覆盖-->
<group ns="namespaceA">  
 <node ns='namespaceB' pkg="talker" type="talker" name="talker"/>
</group>
```

* topic
  * 当创建的节点有命名空间时，base类型(e.g. `node_name`，而非`/.../node_name`)的topic会附上节点的命名空间
  * 当创建的句柄带有(~)时，base类型的topic除了附上节点的命名空间，还会附上节点名作为命名空间

## rosDebug

* 工具A：rqt（可查看相关的主题、节点等各种操作）

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210902082605313.png" alt="image-20210902082605313" style="zoom:67%; " />

## 部署ros package

> There is likely **a lot** more **learning curve** than catkin_make install + copy binaries. @[answers.ros](https://answers.ros.org/question/226581/deploying-a-catkin-package/)

[实战教程 github](https://github.com/GDUT-IIDCC/Sleipnir.PreCompile)

.. attention:: source的先后顺序非常重要

## 常用ros python api解读

### rospy.init_node()为什么在主线程才能调用？

* 一般来说只能在**主线程**中进行调用，因为**init_node**时会构建信号回调函数(signal handlers)，而python中构建信号回调函数需要在主线程中进行构建（python特性）。
* 设置不构建信号回调函数即可以在非主线程调用`rospy.init_node`

![image-20210909214309037](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210909214309037.png)

```python
# 测试案例
import rospy
import threading

class myThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        rospy.init_node('my_node_name')
        rospy.spin()

if __name__ == '__main__':
    thread = myThread()
    thread.start()
    thread.join()
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210909214844411.png" alt="image-20210909214844411" style="zoom:50%; " />

* 其中rospy_init构筑的signal handlers如下：

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210909215248055.png" alt="image-20210909215248055" style="zoom:50%; " />

### rospy回调函数的多线程处理机制

![rospy-cb-multithread.png](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/rospy-cb-multithread.png)

* `rospy`中处理回调函数时会派生出一个新的线程去执行（线程名与主题名相同）

> These threads are different from the main thread of your program.

* 拓展：如果有n个回调函数（处理的是不同的topic）则会派生出n个线程；如果有回调函数处理相同的topic则共用一个线程

.. attention:: 此处阐述的是 `rospy` 的回调函数的机制，在 `roscpp` 中会有所不同

#### 拓展资料

* [Threads in ROS and Python](https://nu-msr.github.io/me495_site/lecture08_threads.html#what-this-threading-model-means-for-you)
* [ROS Spinning, Threading, Queuing](https://levelup.gitconnected.com/ros-spinning-threading-queuing-aac9c0a793f)

## 自定义rviz插件

### 参考资料

* [应用案例官方教程](https://github.com/autolaborcenter/rviz_navi_multi_goals_pub_plugin.git)（含display, panel, tool的自定义设置）

相关用法

1. 继承rviz:: Panel类

```c++
class TeleopPanel: public rviz:: Panel{

    ...

}

```

#### [给自定义插件添加icon](https://answers.ros.org/question/213971/how-to-add-an-icon-to-a-custom-display-type-in-rviz/)

只需要在icon**s**/class**es**目录下添加icon.png文件即可，icon文件名同插件名

## [rviz Qt (for python)](https://github.com/Natsu-Akatsuki/memo/tree/master/%E5%BC%80%E5%8F%91%E7%AC%94%E8%AE%B0/source/ros%E7%AC%94%E8%AE%B0/example/rviz_qt.py)

**用完一圈之后，不推荐使用这个rviz的python api，一是文档太少，难以进行开发，二是坑很多**。比如退出Qt应用程序后，rviz节点将成为僵尸节点（即不能被rosnode kill掉，只能使用rosnode cleanup清理），而在实测中c++中不存在这个问题，进程可以退出得很干净；不能够在Qt中的rviz中添加图像面板，否则会有段错误提示

* 官方[简例](http://docs.ros.org/en/lunar/api/rviz_python_tutorial/html/ind)含：frame（rviz界面）、thickness_slider（滑动条）、按键；只显示 3D render

## 生成和调用自定义消息类型

### 生成

* 在`package.xml中`：增加`message_generation` 这种`build_depend` tag ；增加`message_runtime` 这种`exec_depend` tag

* 在`CMakeLists.txt`中：增加依赖 `message_generation` 到 `find_package(catkin REQUIRED COMPONENTS ...)`；add\_*\_files这部分内容选择性取消注释，添加`.msg`文件；`generate_messages`这部分内容选择性取消注释 `generate_messages(DEPENDENCIES ...）`中添加依赖的包名

### 调用

* 在`package.xml`：增加包的`build_depend` tag和`exec_depend` tag

* 在`CMakeLists.txt`中：将相关信息类型包添加到`find_package`即可

.. note:: depend = exec_depend + build_depend，可以用depend tag来替代其他两个tag

### 实例

#### 生成自定义的消息类型

步骤一：创建一个简易package

```bash
catkin_create_pkg msg_test01 rospy generate_messages
```

步骤二：创建 `CMakeLists.txt` （追加）

```cmake
cmake_minimum_required(VERSION 3.0.2)
project(msg_test01)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
)

## 生成自定义的消息类型（build阶段，生成在build/下）

add_message_files(
  # DIRECTORY (arg) 指定文件夹
  FILES  # 指定文件
  test01.msg
)

#  根据依赖，生成msg源文件（run阶段，生成在devel/下）
generate_messages(
#   DEPENDENCIES   有依赖时需取消#
#   std_msgs  # Or other packages containing msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy
)
```

步骤三：创建 `package.xml`

```xml
<?xml version="1.0"?>
<package format="2">
  <name>msg_test01</name>
  <version>0.0.0</version>
  <description>The msg_test01 package</description>

  <maintainer email="helios@todo.todo">helios</maintainer>

  <license>TODO</license>

  <buildtool_depend>catkin</buildtool_depend>
  
  <build_depend>rospy</build_depend>
  <exec_depend>rospy</exec_depend>

  <build_depend>message_generation</build_depend>
  <exec_depend>message_runtime</exec_depend>
  
</package>
```

.. note:: 生成的 `python msg module` 在 `devel/lib/python*/dist-packages/` 中

#### 调用自定义的消息类型

步骤一：创建一个简易package

```bash
catkin_create_pkg msg_test02 rospy
```

步骤二：创建 `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.0.2)
project(msg_test02)

find_package(catkin REQUIRED COMPONENTS
  msg_test01
  rospy
)
```

步骤三：创建 `package.xml`

```xml
<?xml version="1.0"?>
<package format="2">
  <name>msg_test02</name>
  <version>0.0.0</version>
  <description>The msg_test02 package</description>
  <maintainer email="helios@todo.todo">helios</maintainer>
  <license>TODO</license>

  <buildtool_depend>catkin</buildtool_depend>

  <build_depend>msg_test01</build_depend>
  <exec_depend>msg_test01</exec_depend>

  <build_depend>rospy</build_depend>
  <exec_depend>rospy</exec_depend>

</package>
```

步骤四：创建 `msg_test02.py` 和使用自定义的消息类型

```python
import rospy
# 注意此处的import含.msg
from msg_test01.msg import test01

rospy.init_node('msg_test01', anonymous=False)
test01 = test01()
```

## TF2

### 使用ROS2发布静态TF

#### roslaunch

```xml
<!-- static_transform_publisher x y z yaw pitch roll 父 子坐标系 -->
<node pkg="tf2_ros" type="static_transform_publisher" name="camera_to_lidar" args="0, 0, 0, 0, 0, 0 lidar camera" />
```

### 图形化查看TF树

```bash
# noetic使用tf2
$ rosrun rqt_tf_tree rqt_tf_tree
# sudo apt install ros-noetic-tf2-tools
$ rosrun tf2_tools view_frames.py
```

## 队列长度设置

* 在ros中，可能会遇到一些很耗时的操作，比如点云配准，图像特征提取。这样的话，回调函数的处理时间就会变得很长。如果发布端发布数据的频率高于订阅端处理的速度，同时订阅端没有限制地处理所有的数据的话，就会使订阅端一直处理较旧的数据。最终的数据和数据的处理之间的时延将会很高。希望处理最新的数据的话，就需要将发布器和订阅器的队列长度设置为1。

* 如下为图像处理效果图（左为输出效果，右为输入图像，可看出有较大的时延）（实测：inference时间和ros image数据传输耗时为ms级别）

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/latency.gif)

* 参考资料：[csdn](https://blog.csdn.net/qq_32618327/article/details/121650164)

## 传感器

### 相机

* 将数据从opencv->ros时，一般采用bgr编码方式（opencv原本的数据默认即bgr通道）

```c++
cv_bridge::CvImagePtr cv_ptr;
try                 
{ // 不提供第二个参数时将等效于"passthrough"，不对图片进行变换
 cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
}
catch (cv_bridge::Exception& e)
{
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
}

// Update GUI Window
 image_pub_.publish(cv_ptr->toImageMsg());
```

* 参考资料：

1. [cv_bridge](http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages)

---

**NOTE**

* opencv默认使用**bgr通道**，im.read读图片的时候是使用bgr通道；im.show正常显示图片需要使用bgr通道；im.write写图片也是需要使用bgr通道（[reference](https://stackoverflow.com/questions/50963283/python-opencv-imshow-doesnt-need-convert-from-bgr-to-rgb)）

*

---

## TroubleShooting

* [ros wiki trouble shooting](http://roswiki.autolabor.com.cn/rospy(2f)Troubleshooting.html)(含ctrl+c和import问题)
