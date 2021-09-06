 

 

 

# 基于统计学的滤波

- 作用：通过统计分析，实现去噪/离群点的效果

https://pcl.readthedocs.io/projects/tutorials/en/latest/statistical_outlier.html#statistical-outlier-removal



## 背景

- 激光器的测量误差会生成一些稀疏的离群点，从而影响算法的结果（如测量平面法向量、曲率变化时会得到错误的结果，进而影响点云的配准）
- 这种问题，可以对每个激光点的领域进行统计分析，然后去掉那些不符合阈值的领域点
- 本算法的统计量是：领域点到激光点均值距离的分布（假设该分布服从正态分布，有均值和方差，其中均值为领域均值距离的期望）；若某个激光点的均值距离大于某个间隔（由总体均值和总体标准差决定），则认为该激光点为离群点，要去除它。

- 对于每个激光点，都要算它的领域点到它的距离的均值



下图显示了去噪的结果：左边是没去噪的，右边是去噪的。

![https://pcl.readthedocs.io/projects/tutorials/en/latest/_images/statistical_removal_2.jpg](/home/helios/pcl%20%E7%82%B9%E4%BA%91%E5%BA%93%E7%BF%BB%E8%AF%91/Day4%20%E5%8E%BB%E5%99%AA%20%EF%BC%88%E7%BB%9F%E8%AE%A1%E5%AD%A6%E6%BB%A4%E6%B3%A2%EF%BC%89.assets/statistical_removal_2.jpg)

## 代码

1. [下载点云文件](https://raw.github.com/PointCloudLibrary/data/master/tutorials/table_scene_lms400.pcd)；创建文件 `statistical_removal.cpp` 

```c++
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

  // Fill in the cloud data
  pcl::PCDReader reader;
  // Replace the path below with the path where you saved your file
  reader.read<pcl::PointXYZ> ("table_scene_lms400.pcd", *cloud);

  std::cerr << "Cloud before filtering: " << std::endl;
  std::cerr << *cloud << std::endl;

  // Create the filtering object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (50);
  sor.setStddevMulThresh (1.0);
  sor.filter (*cloud_filtered);

  std::cerr << "Cloud after filtering: " << std::endl;
  std::cerr << *cloud_filtered << std::endl;

  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ> ("table_scene_lms400_inliers.pcd", *cloud_filtered, false);

  sor.setNegative (true);
  sor.filter (*cloud_filtered);
  writer.write<pcl::PointXYZ> ("table_scene_lms400_outliers.pcd", *cloud_filtered, false);

  return (0);
}

```



## 代码解构

1、读取pcd文件加载点云

```c++
  // Fill in the cloud data
  pcl::PCDReader reader;
  // Replace the path below with the path where you saved your file
  reader.read<pcl::PointXYZ> ("table_scene_lms400.pcd", *cloud);
```

2、创建一个 `pcl::StatisticalOutlierRemoval` 滤波器。每个激光点用来分析的领域点为50，标准差乘子为1（意味着离query点的距离大于均值的1标准差的领域点则被去除），输出结果存放在`cloud_filtered` 中

```c++
  // Create the filtering object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (50);
  sor.setStddevMulThresh (1.0);
  sor.filter (*cloud_filtered);
```

3、滤波后的激光点存储到磁盘

```c++
 pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ> ("table_scene_lms400_inliers.pcd", *cloud_filtered, false);
```

4、获取离群点，并存储到磁盘中

```c++
  sor.setNegative (true);
  sor.filter (*cloud_filtered);
  writer.write<pcl::PointXYZ> ("table_scene_lms400_outliers.pcd", *cloud_filtered, false);
```



## 编译和运行代码

1、创建CMakeLists.txt文件

```cmake
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(statistical_removal)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (statistical_removal statistical_removal.cpp)
target_link_libraries (statistical_removal ${PCL_LIBRARIES})
```

2、执行二进制`target`文件

```
$ ./statistical_removal
```

3、可看到相应的输出

```
Cloud before filtering:
header:
seq: 0
stamp: 0.000000000
frame_id:
points[]: 460400
width: 460400
height: 1
is_dense: 0

Cloud after filtering:
header:
seq: 0
stamp: 0.000000000
frame_id:
points[]: 429398
width: 429398
height: 1
is_dense: 0
```



# 小结

- 可以发现pcl库配置属性时一般都是调用方法的 e.g. 对象名.set<某个属性>()