# How to use a KdTree to search

- Kdtree来寻找某个特定点的最近邻点的位置
- 寻找一定半径范围内的最近领点



# Theoretical primer

A k-d tree, or k-dimensional tree, is a data structure used in  computer science for organizing some number of points in a space with k  dimensions.  It is a binary search tree with other constraints imposed  on it. K-d trees are very useful for range and nearest neighbor  searches.  For our purposes we will generally only be dealing with point clouds in three dimensions, so all of our k-d trees will be  three-dimensional.  Each level of a k-d tree splits all children along a specific dimension, using a hyperplane that is perpendicular to the  corresponding axis.  At the root of the tree all children will be split  based on the first dimension (i.e. if the first dimension coordinate is  less than the root it will be in the left-sub tree and if it is greater  than the root it will obviously be in the right sub-tree).  Each level  down in the tree divides on the next dimension, returning to the first  dimension once all others have been exhausted.  The most efficient way  to build a k-d tree is to use a partition method like the one Quick Sort uses to place the median point at the root and everything with a  smaller one-dimensional value to the left and larger to the right.  You  then repeat this procedure on both the left and right sub-trees until  the last trees that you are to partition are only composed of one  element.

From [[Wikipedia\]](https://pcl.readthedocs.io/projects/tutorials/en/latest/random_sample_consensus.html#wikipedia):

![Example of a 2-d k-d tree](/home/helios/pcl%20%E7%82%B9%E4%BA%91%E5%BA%93%E7%BF%BB%E8%AF%91/Kdtree.assets/2d_kdtree.png)

This is an example of a 2-dimensional k-d tree

![img](/home/helios/pcl%20%E7%82%B9%E4%BA%91%E5%BA%93%E7%BF%BB%E8%AF%91/Kdtree.assets/nn_kdtree.gif)

This is a demonstration of how the Nearest-Neighbor search works.



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

 
