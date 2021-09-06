 

 

# [基于区域生长的分割](https://pcl.readthedocs.io/projects/tutorials/en/latest/region_growing_segmentation.html#region-growing-segmentation)

### 代码

1. [下载点云文件](https://raw.githubusercontent.com/PointCloudLibrary/data/master/tutorials/region_growing_tutorial.pcd)；创建文件 `region_growing_segmentation.cpp` 

```c++
#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>

int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if ( pcl::io::loadPCDFile <pcl::PointXYZ> ("region_growing_tutorial.pcd", *cloud) == -1)
  {
    std::cout << "Cloud reading failed." << std::endl;
    return (-1);
  }

  pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (50);
  normal_estimator.compute (*normals);

  pcl::IndicesPtr indices (new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 1.0);
  pass.filter (*indices);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (50);
  reg.setMaxClusterSize (1000000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (cloud);
  //reg.setIndices (indices);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (1.0);

  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);

  std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
  std::cout << "First cluster has " << clusters[0].indices.size () << " points." << std::endl;
  std::cout << "These are the indices of the points of the initial" <<
  std::endl << "cloud that belong to the first cluster:" << std::endl;
  int counter = 0;
  while (counter < clusters[0].indices.size ())
  {
    std::cout << clusters[0].indices[counter] << ", ";
    counter++;
    if (counter % 10 == 0)
      std::cout << std::endl;
  }
  std::cout << std::endl;

  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  pcl::visualization::CloudViewer viewer ("Cluster viewer");
  viewer.showCloud(colored_cloud);
  while (!viewer.wasStopped ())
  {
  }

  return (0);
}
```



### 代码解构

1. 读取pcd文件加载点云

```c++
 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if ( pcl::io::loadPCDFile <pcl::PointXYZ> ("region_growing_tutorial.pcd", *cloud) == -1)
  {
    std::cout << "Cloud reading failed." << std::endl;
    return (-1);
  }
```

2. 该算法需要计算法向量。此处调用 `pcl::NormalEstimation` 类来计算。更多细节可参考 [Estimating Surface Normals in a PointCloud](https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation.html#normal-estimation)

```c++
// 要有kd+pointcloud+领域点个数  

  pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (50);
  normal_estimator.compute (*normals);
```

3. 本部分属于拓展，可注释掉。 `pcl::RegionGrowing` 继承于 `pcl::PCLBase`，可使用索引来选择性地进行点云分割

```c++
  pcl::IndicesPtr indices (new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 1.0);
  pass.filter (*indices);
```

4. `pcl::RegionGrowing` 是一个模板类，有两个参数`parameters`

- PointT - 点的类型 (本例中为 `pcl::PointXYZ`)
- NormalT - 法线的类型 (本例中为`pcl::Normal`)

设置聚类点云团的尺寸，用于去除激光点数小于或大于某个阈值的点云团。

```c++
  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (50);  // 默认值为1
  reg.setMaxClusterSize (1000000);  // 默认值为尽可能大
```

5. 提供kdtree树对象和领域点个数K

```c++
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (cloud);
  //reg.setIndices (indices);   todo：待分割点云在原点云的索引
  reg.setInputNormals (normals);
```

6. 这两部分是初始化算法的核心部分，因为涉及到平滑度（smoothness）的约束；第一个方法，设置允许的法线偏差范围角度（弧度），如果两点的法线偏差小于平滑度阈值则认为是同一个聚类点云团。第二个是曲率阈值，如果两点的发现偏差较小，则衡量他们的曲率偏差。如果小于曲率阈值，则两个激光点属于同一类。

```
  reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (1.0);
```

7. 执行(launch) 分割算法。

```c++
  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);
```

8.  描述了如何使用`pcl::PointIndices` 来访问元素

```c++
  // 聚类点云团的个数
  std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
  // 第一个聚类点云团的激光点的索引
  std::cout << "First cluster has " << clusters[0].indices.size () << " points." << std::endl;
  std::cout << "These are the indices of the points of the initial cloud that belong to the first cluster: " << std::endl;
  int counter = 0;
  while (counter < clusters[0].indices.size ())
  {
    std::cout << clusters[0].indices[counter] << ", ";
    counter++;
    if (counter % 10 == 0)
      std::cout << std::endl;
  }
  std::cout << std::endl;
```

9. `pcl::RegionGrowing`类提供了一种方法得到彩色的聚类点云团。这些点云团可以通过实例化`pcl::visualization::CloudViewer`来可视化，更多可视化细节可看 [The CloudViewer](https://pcl.readthedocs.io/projects/tutorials/en/latest/cloud_viewer.html#cloud-viewer)

```c++
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  pcl::visualization::CloudViewer viewer ("Cluster viewer");
  viewer.showCloud(colored_cloud);
  while (!viewer.wasStopped ())
  {
  }

  return (0);
}

// 对应的头文件为#include <pcl/visualization/cloud_viewer.h>
// 显示的效果跟pcl_viewer一样，只是少了一些按键功能
```



### 编译和运行代码

1、创建CMakeLists.txt文件

```cmake
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(region_growing_segmentation)

find_package(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (region_growing_segmentation region_growing_segmentation.cpp)
target_link_libraries (region_growing_segmentation ${PCL_LIBRARIES})
```

2、执行二进制`target`文件

```
$ ./region_growing_segmentation
```
