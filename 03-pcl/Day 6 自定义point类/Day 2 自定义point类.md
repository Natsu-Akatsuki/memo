# [自定义PointT类](https://pcl.readthedocs.io/projects/tutorials/en/latest/adding_custom_ptype.html#id5)

1. 定义一个结构体

```
struct MyPointType {  float test; }; 
```

Then, you need to make sure your code includes the template header implementation of the specific class/algorithm in PCL that you want your new point type MyPointType to work with. For example, say you want to use pcl::PassThrough. All you would have to do is:

```
#define PCL_NO_PRECOMPILE
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>

// the rest of the code goes here
```

If your code is part of the library, which gets used by others, it might also make sense to try to use explicit instantiations for your MyPointType types, for any classes that you expose (from PCL our outside PCL).



PS：

从`PCL-1.7`开始在导入任何pcl头文件？前，都需要`define`（定义）`PCL_NO_PRECOMPILE` before you include any PCL headers to include the 模板类算法 templated algorithms as well.



# [Example](https://pcl.readthedocs.io/projects/tutorials/en/latest/adding_custom_ptype.html#id6)

创建一个新的激光点类，包含x,y,z和test字段

```
#define PCL_NO_PRECOMPILE
#include <pcl/memory.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

struct MyPointType
{
  PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
  float test;
  PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (MyPointType,           // here we assume a XYZ + "test" (as fields)
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, test, test)
)


int
main (int argc, char** argv)
{
  pcl::PointCloud<MyPointType> cloud;
  cloud.points.resize (2);
  cloud.width = 2;
  cloud.height = 1;

  cloud[0].test = 1;
  cloud[1].test = 2;
  cloud[0].x = cloud[0].y = cloud[0].z = 0;
  cloud[1].x = cloud[1].y = cloud[1].z = 3;

  pcl::io::savePCDFile ("test.pcd", cloud);
}
```



# 代码

1、创建源文件 `kdtree_search.cpp` 

```c++
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <iostream>
#include <vector>
#include <ctime>

int
main (int argc, char** argv)
{
  srand (time (NULL));

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  // Generate pointcloud data
  cloud->width = 1000;
  cloud->height = 1;
  cloud->points.resize (cloud->width * cloud->height);

  for (std::size_t i = 0; i < cloud->size (); ++i)
  {
    (*cloud)[i].x = 1024.0f * rand () / (RAND_MAX + 1.0f);
    (*cloud)[i].y = 1024.0f * rand () / (RAND_MAX + 1.0f);
    (*cloud)[i].z = 1024.0f * rand () / (RAND_MAX + 1.0f);
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

  kdtree.setInputCloud (cloud);

  pcl::PointXYZ searchPoint;

  searchPoint.x = 1024.0f * rand () / (RAND_MAX + 1.0f);
  searchPoint.y = 1024.0f * rand () / (RAND_MAX + 1.0f);
  searchPoint.z = 1024.0f * rand () / (RAND_MAX + 1.0f);

  // K nearest neighbor search

  int K = 10;

  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);

  std::cout << "K nearest neighbor search at (" << searchPoint.x 
            << " " << searchPoint.y 
            << " " << searchPoint.z
            << ") with K=" << K << std::endl;

  if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
  {
    for (std::size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
      std::cout << "    "  <<   (*cloud)[ pointIdxNKNSearch[i] ].x 
                << " " << (*cloud)[ pointIdxNKNSearch[i] ].y 
                << " " << (*cloud)[ pointIdxNKNSearch[i] ].z 
                << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
  }

  // Neighbors within radius search
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;

  float radius = 256.0f * rand () / (RAND_MAX + 1.0f);

  std::cout << "Neighbors within radius search at (" << searchPoint.x 
            << " " << searchPoint.y 
            << " " << searchPoint.z
            << ") with radius=" << radius << std::endl;


  if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
  {
    for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
      std::cout << "    "  <<   (*cloud)[ pointIdxRadiusSearch[i] ].x 
                << " " << (*cloud)[ pointIdxRadiusSearch[i] ].y 
                << " " << (*cloud)[ pointIdxRadiusSearch[i] ].z 
                << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
  }


  return 0;
}
```



# 代码解构

1、使用系统时间作为随机数的种子；用随机数给点云对象赋值

```c++
  srand (time (NULL));

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  // Generate pointcloud data
  cloud->width = 1000;
  cloud->height = 1;
  cloud->points.resize (cloud->width * cloud->height);

  for (std::size_t i = 0; i < cloud->size (); ++i)
  {
    (*cloud)[i].x = 1024.0f * rand () / (RAND_MAX + 1.0f);
    (*cloud)[i].y = 1024.0f * rand () / (RAND_MAX + 1.0f);
    (*cloud)[i].z = 1024.0f * rand () / (RAND_MAX + 1.0f);
  }
```

1、创建一个Kdtree对象，以刚刚的点云对象为输入；构建一个搜索点

```c++
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

  kdtree.setInputCloud (cloud);

  pcl::PointXYZ searchPoint;

  searchPoint.x = 1024.0f * rand () / (RAND_MAX + 1.0f);
  searchPoint.y = 1024.0f * rand () / (RAND_MAX + 1.0f);
  searchPoint.z = 1024.0f * rand () / (RAND_MAX + 1.0f);
```

2、创建一个整型和浮点型的向量来存储近邻结果

```c++
  // K nearest neighbor search

  int K = 10;

  std::vector<int> pointIdxNKNSearch(K);  // 该向量有大小
  std::vector<float> pointNKNSquaredDistance(K);

  std::cout << "K nearest neighbor search at (" << searchPoint.x 
            << " " << searchPoint.y 
            << " " << searchPoint.z
            << ") with K=" << K << std::endl;
```

3、输出 “searchPoint” 的10个最近邻点的位置，并存放到我们创建的向量中

```c++
  if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
  {
    for (std::size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
      std::cout << "    "  <<   (*cloud)[ pointIdxNKNSearch[i] ].x 
                << " " << (*cloud)[ pointIdxNKNSearch[i] ].y 
                << " " << (*cloud)[ pointIdxNKNSearch[i] ].z 
                << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
  }
```

4、接下来我们将增加得到的最近邻点的约束，它的值还需要在一定半径范围内；创建两个向量来存储结果

```c++
  // Neighbors within radius search

  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;

  float radius = 256.0f * rand () / (RAND_MAX + 1.0f);
```

5、输出和存储结果

```c++
  if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
  {
    for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
      std::cout << "    "  <<   (*cloud)[ pointIdxRadiusSearch[i] ].x 
                << " " << (*cloud)[ pointIdxRadiusSearch[i] ].y 
                << " " << (*cloud)[ pointIdxRadiusSearch[i] ].z 
                << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
  }
```



# 编译和执行结果

1、创建CMakeLists.txt文件

```cmake
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(kdtree_search)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (kdtree_search kdtree_search.cpp)
target_link_libraries (kdtree_search ${PCL_LIBRARIES})
```

2、执行二进制`target`文件

```
$ ./kdtree_search
```

3、可看到输出结果：

```
K nearest neighbor search at (455.807 417.256 406.502) with K=10
  494.728 371.875 351.687 (squared distance: 6578.99)
  506.066 420.079 478.278 (squared distance: 7685.67)
  368.546 427.623 416.388 (squared distance: 7819.75)
  474.832 383.041 323.293 (squared distance: 8456.34)
  470.992 334.084 468.459 (squared distance: 10986.9)
  560.884 417.637 364.518 (squared distance: 12803.8)
  466.703 475.716 306.269 (squared distance: 13582.9)
  456.907 336.035 304.529 (squared distance: 16996.7)
  452.288 387.943 279.481 (squared distance: 17005.9)
  476.642 410.422 268.057 (squared distance: 19647.9)
Neighbors within radius search at (455.807 417.256 406.502) with radius=225.932
  494.728 371.875 351.687 (squared distance: 6578.99)
  506.066 420.079 478.278 (squared distance: 7685.67)
  368.546 427.623 416.388 (squared distance: 7819.75)
  474.832 383.041 323.293 (squared distance: 8456.34)
  470.992 334.084 468.459 (squared distance: 10986.9)
  560.884 417.637 364.518 (squared distance: 12803.8)
  466.703 475.716 306.269 (squared distance: 13582.9)
  456.907 336.035 304.529 (squared distance: 16996.7)
  452.288 387.943 279.481 (squared distance: 17005.9)
  476.642 410.422 268.057 (squared distance: 19647.9)
  499.429 541.532 351.35 (squared distance: 20389)
  574.418 452.961 334.7 (squared distance: 20498.9)
  336.785 391.057 488.71 (squared distance: 21611)
  319.765 406.187 350.955 (squared distance: 21715.6)
  528.89 289.583 378.979 (squared distance: 22399.1)
  504.509 459.609 541.732 (squared distance: 22452.8)
  539.854 349.333 300.395 (squared distance: 22936.3)
  548.51 458.035 292.812 (squared distance: 23182.1)
  546.284 426.67 535.989 (squared distance: 25041.6)
  577.058 390.276 508.597 (squared distance: 25853.1)
  543.16 458.727 276.859 (squared distance: 26157.5)
  613.997 387.397 443.207 (squared distance: 27262.7)
  608.235 467.363 327.264 (squared distance: 32023.6)
  506.842 591.736 391.923 (squared distance: 33260.3)
  529.842 475.715 241.532 (squared distance: 36113.7)
  485.822 322.623 244.347 (squared distance: 36150.5)
  362.036 318.014 269.201 (squared distance: 37493.6)
  493.806 600.083 462.742 (squared distance: 38032.3)
  392.315 368.085 585.37 (squared distance: 38442.9)
  303.826 428.659 533.642 (squared distance: 39392.8)
  616.492 424.551 289.524 (squared distance: 39556.8)
  320.563 333.216 278.242 (squared distance: 41804.5)
  646.599 502.256 424.46 (squared distance: 43948.8)
  556.202 325.013 568.252 (squared distance: 44751)
  291.27 497.352 515.938 (squared distance: 45463.9)
  286.483 322.401 495.377 (squared distance: 45567.2)
  367.288 550.421 550.551 (squared distance: 46318.6)
  595.122 582.77 394.894 (squared distance: 46938.1)
  256.784 499.401 379.931 (squared distance: 47064.1)
  430.782 230.854 293.829 (squared distance: 48067.2)
  261.051 486.593 329.854 (squared distance: 48612.7)
  602.061 327.892 545.269 (squared distance: 48632.4)
  347.074 610.994 395.622 (squared distance: 49475.6)
  482.876 284.894 583.888 (squared distance: 49718.6)
  356.962 247.285 514.959 (squared distance: 50423.7)
  282.065 509.488 516.216 (squared distance: 50730.4)
```

 
