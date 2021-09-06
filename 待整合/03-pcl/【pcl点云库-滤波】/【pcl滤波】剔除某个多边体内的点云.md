# 剔除某个多边体内的点云

## [代码](https://blog.csdn.net/qq_42318305/article/details/81985663)

1. 创建文件 `crop_box.cpp` 

```c++
//PCL CropBox   过滤掉用户给定立方体内的点云数据
#include <pcl/filters/crop_box.h>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>


typedef pcl::PointXYZ PointT;
using namespace std;

int main()
{
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::console::TicToc tt;

    std::cerr<<"ReadImage...\n",tt.tic();
    pcl::PCDReader reader;
    reader.read("2.pcd",*cloud);
    std::cerr<<"Done  "<<tt.toc()<<"  ms\n"<<std::endl;
    std::cerr<<"The points data:  "<<cloud->points.size()<<std::endl;

    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    pcl::CropBox<PointT> crop;

    crop.setMin(Eigen::Vector4f(-0.2,-0.2,0.0,1.0)); //给定立体空间
    crop.setMax(Eigen::Vector4f(-2.0,1.0,2.0,1.0));  //数据随意给的，具体情况分析
    crop.setInputCloud(cloud);
    crop.setKeepOrganized(true);
    crop.setUserFilterValue(0.1f);
    crop.filter(*cloud_filtered);
    std::cerr<<"The points data:  "<<cloud_filtered->points.size()<<std::endl;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    int v1(0);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->setBackgroundColor(0, 0, 0, v1);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud1", v1);

    int v2(0);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, "sample cloud2", v2);
    viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);
    viewer->addCoordinateSystem(1.0);

    viewer->initCameraParameters();
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    };

    return 0;
}
```



## 代码解构

1、读取pcd文件加载点云

```c++
 	// 创建crop对象
	pcl::CropBox<PointT> crop;
    crop.setMin(Eigen::Vector4f(-0.2,-0.2,0.0,1.0)); //给定立体空间
    crop.setMax(Eigen::Vector4f(-2.0,1.0,2.0,1.0));  //数据随意给的，具体情况分析
    crop.setInputCloud(cloud);
    crop.setKeepOrganized(true);
    crop.setUserFilterValue(0.1f);
    crop.filter(*cloud_filtered);
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



# 拓展案例

- [案例1](https://blog.csdn.net/ethan_guo/article/details/80359313])

 ```c++
<crop_object>.setNegative(false);  // 可设置为true，代表滤除立方体外的激光点
 ```



# [直通和统计学滤波](https://blog.csdn.net/peach_blossom/article/details/78271683)

- 直通应用：剔除某个方向上的激光点

