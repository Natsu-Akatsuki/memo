# 平面拟合

## 代码

1. 创建文件 `planar_segmentation.cpp` 

```c++
#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

int
main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Fill in the cloud data
    cloud->width  = 15;
    cloud->height = 1;
    cloud->points.resize (cloud->width * cloud->height);

    // Generate the data
    for (auto& point: *cloud)
    {
        point.x = 1024 * rand () / (RAND_MAX + 1.0f);
        point.y = 1024 * rand () / (RAND_MAX + 1.0f);
        point.z = 1.0;
    }

    // Set a few outliers
    (*cloud)[0].z = 2.0;
    (*cloud)[3].z = -2.0;
    (*cloud)[6].z = 4.0;

    std::cerr << "Point cloud data: " << cloud->size () << " points" << std::endl;
    for (const auto& point: *cloud)
        std::cerr << "    " << point.x << " "
                  << point.y << " "
                  << point.z << std::endl;

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);  	 // 拟合模型的类型 e.g. 平面、圆柱
    seg.setMethodType (pcl::SAC_RANSAC);			   // 拟合模型的方法
    seg.setDistanceThreshold (0.01); 							  // 距离阈值

    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);

    if (inliers->indices.size () == 0)
    {
        PCL_ERROR ("Could not estimate a planar model for the given dataset.");
        return (-1);
    }

    std::cerr << "Model coefficients: " << coefficients->values[0] << " "
              << coefficients->values[1] << " "
              << coefficients->values[2] << " "
              << coefficients->values[3] << std::endl;

    std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
    // for (std::size_t i = 0; i < inliers->indices.size (); ++i) 
    for (const auto& idx: inliers->indices)
        std::cerr << idx << "    " << cloud->points[idx].x << " "
        << cloud->points[idx].y << " "
        << cloud->points[idx].z << std::endl;

    return (0);
}
```



## 代码解构

1、创建点云；人为设置离群点

```c++
  // Fill in the cloud data
  cloud->width  = 15;
  cloud->height = 1;
  cloud->points.resize (cloud->width * cloud->height);

  // Generate the data
  for (auto& point: *cloud)
  {
    point.x = 1024 * rand () / (RAND_MAX + 1.0f);
    point.y = 1024 * rand () / (RAND_MAX + 1.0f);
    point.z = 1.0;
  }

  // Set a few outliers
  (*cloud)[0].z = 2.0;
  (*cloud)[3].z = -2.0;
  (*cloud)[6].z = 4.0;

  std::cerr << "Point cloud data: " << cloud->size () << " points" << std::endl;
  for (const auto& point: *cloud)
    std::cerr << "    " << point.x << " "
                        << point.y << " "
                        << point.z << std::endl;
```

2、创建 [pcl::SACSegmentation ](https://pcl.readthedocs.io/projects/tutorials/en/latest/planar_segmentation.html?highlight=SACSegmentation#id1) 对象，并设置模型和方法的类型。设置 and set the model and method type. 距离阈值`distance threshold`来判别哪个点是inlier。此处使用的方法是RANSAC，更多细节可参考 [Wikipedia page](https://en.wikipedia.org/wiki/RANSAC)

```c++
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);

  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);
```

3、输出拟合的平面模型参数

```c++
 std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3] << std::endl;
```



## 注意事项

- 本例中得到的默认是inlier点的索引值





## 编译和运行代码

1、创建CMakeLists.txt文件

```cmake
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(planar_segmentation)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (planar_segmentation planar_segmentation.cpp)
target_link_libraries (planar_segmentation ${PCL_LIBRARIES})
```

2、执行二进制`target`文件

```
$ ./region_growing_segmentation
```

3、得到运行结果

```c++
Point cloud data: 15 points
    0.352222 -0.151883 2
    -0.106395 -0.397406 1
    -0.473106 0.292602 1
    -0.731898 0.667105 -2
    0.441304 -0.734766 1
    0.854581 -0.0361733 1
    -0.4607 -0.277468 4
    -0.916762 0.183749 1
    0.968809 0.512055 1
    -0.998983 -0.463871 1
    0.691785 0.716053 1
    0.525135 -0.523004 1
    0.439387 0.56706 1
    0.905417 -0.579787 1
    0.898706 -0.504929 1
Model coefficients: 0 0 1 -1
Model inliers: 12
1    -0.106395 -0.397406 1
2    -0.473106 0.292602 1
4    0.441304 -0.734766 1
5    0.854581 -0.0361733 1
7    -0.916762 0.183749 1
8    0.968809 0.512055 1
9    -0.998983 -0.463871 1
10    0.691785 0.716053 1
11    0.525135 -0.523004 1
12    0.439387 0.56706 1
13    0.905417 -0.579787 1
14    0.898706 -0.504929 1
```

