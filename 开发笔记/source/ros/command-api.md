# ros command and api

<p align="right">Author: kuzen, Natsu-Akatsuki</p>

## rosrun

```bash
$ rosrun <package_name> <executable>
```

.. attention:: 有可执行权限的文件，其文件名才能被命令行补全；python脚本记得在首行添加解释器路径

## rosbag

### 常用命令行

```bash
# ---回放---
$ rosbag play <包名>  
# --clock     # publish the clock time
# --topic     # 只发布自己想要的topic；另外可在后面加:=进行重映射
# -r <n>      # 以n倍速播放
  
# ---录制---
$ rosbag record <主题名> 

# ---裁剪---
# 这种时刻指的是ros时间戳，类似 1576119471.511449
$ rosbag filter <输入包名> <输出包名> "t.to_sec() < 某个时刻 and t.to_sec() > 某个时刻"

# ---压缩和解压---
$ rosbag compress/decompress <待压缩的包名>
```

### 拓展插件

* [命令行rqt-bag](http://wiki.ros.org/rqt_bag)：一个rosbag录制和回放，查看topic的图形化程序

## rosnode

* [官方文档 for python](http://docs.ros.org/en/hydro/api/rosnode/html/)

### rosnode list

```python
# python
import rosnode
node_list = rosnode.get_node_names()
```

### rosnode kill

```python
# python
node_list = rosnode.get_node_names()
_, _ = rosnode.kill_nodes(node_list)
```

## rostopic

### 常用命令行

```bash
$ rostopic list       # 查看当前发布和订阅的主题
$ rostopic type <topic_name> # 查看主题的类型
$ rostopic echo <topic_name> # 查看主题中的数据
```

* [发布数据时带时间戳](http://wiki.ros.org/ROS/YAMLCommandLine#Headers.2Ftimestamps)

```bash
$ rostopic pub /topic_name topic_type [args...]  # 发布数据
# options
# -r: 指定发布的频率
# -f: 从yaml文件中读取args
# -s: 需配合-r模式使用，可使用auto和now这两个词的substitution

# example
$ rostopic pub -s -r 4 /clicked_point geometry_msgs/PointStamped "header: auto  
point:
  x: 0.0
  y: 0.0
  z: 0.0"
$ rostopic pub -s--use-rostime -r 4 /clicked_point geometry_msgs/PointStamped "header:
  seq: 0
  stamp: now
  frame_id: ''
point:
  x: 0.0
  y: 0.0
  z: 0.0"
```

.. attention::  `-s`好像只能替换命令行中的keyword；使用上`-f`时，只能替换第一次的数据

### rostopic list

```python
# python
import rospy
topic_list = rospy.get_published_topics()
```

## roslaunch

* [官方文档 for python](http://docs.ros.org/en/kinetic/api/roslaunch/html/index.html)，[官方文档 wiki with example](http://wiki.ros.org/roslaunch/API%20Usage)

* 该API支持中文路径

## rospack

* [官方文档 for python](http://docs.ros.org/en/independent/api/rospkg/html/python_api.html)

### 常用命令行

```bash
# 返回某个包的绝对路径
$ rospack find <pkg>
```

### rospack find

```python
# python
# 获取某个package的绝对路径
from rospkg import RosPack
rp = RosPack()
path = rp.get_path('package_name')   # 返回某个包的绝对路径
```

---

```c++
// cpp
#include <ros/package.h>
std::string path = ros::package::getPath("package_name"); 
```

## [rviz(cpp)](http://docs.ros.org/en/jade/api/rviz/html/c++/classrviz_1_1VisualizationFrame.html#a76773514f60d7abbc5db8bd590acd79c)

## rosdep

rosdep相关于ros的apt，用于下载依赖包

```bash
$ rosdep install --from-paths src --ignore-src -r -y
# -i, --ignore-packages-from-source, --ignore-src：若ROS_PACKAGE_PATH有这个包，则不rosdep安装
# --from-paths：搜索路径
# -r：Continue installing despite errors.
# -y：Tell the package manager to default to y or fail when
```
