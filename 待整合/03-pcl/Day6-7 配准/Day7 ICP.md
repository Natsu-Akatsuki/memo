# ICP

# 代码

1、创建源文件 `iterative_closest_point.cpp` 

```c++
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

int
 main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>(5,1));
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);

  // Fill in the CloudIn data
  for (auto& point : *cloud_in)
  {
    point.x = 1024 * rand() / (RAND_MAX + 1.0f);
    point.y = 1024 * rand() / (RAND_MAX + 1.0f);
    point.z = 1024 * rand() / (RAND_MAX + 1.0f);
  }
  
  std::cout << "Saved " << cloud_in->size () << " data points to input:" << std::endl;
      
  for (auto& point : *cloud_in)
    std::cout << point << std::endl;
      
  *cloud_out = *cloud_in;
  
  std::cout << "size:" << cloud_out->size() << std::endl;
  for (auto& point : *cloud_out)
    point.x += 0.7f;

  std::cout << "Transformed " << cloud_in->size () << " data points:" << std::endl;
      
  for (auto& point : *cloud_out)
    std::cout << point << std::endl;

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cloud_in);
  icp.setInputTarget(cloud_out);
  
  pcl::PointCloud<pcl::PointXYZ> Final;
  icp.align(Final);

  std::cout << "has converged:" << icp.hasConverged() << " score: " <<
  icp.getFitnessScore() << std::endl;
  std::cout << icp.getFinalTransformation() << std::endl;

 return (0);
}
```



# 代码解构

1. 导入需要用到的头文件

```c++
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
```

2. 创建两个`pcl::PointCloud<pcl::PointXYZ>` boost shared pointers 并进行初始化and initializes them. The type 

```c++
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>(5,1));
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
```

3. 给点云对象赋值

```c++
  // Fill in the CloudIn data
  for (auto& point : *cloud_in)
  {
    point.x = 1024 * rand() / (RAND_MAX + 1.0f);
    point.y = 1024 * rand() / (RAND_MAX + 1.0f);
    point.z = 1024 * rand() / (RAND_MAX + 1.0f);
  }
  
  std::cout << "Saved " << cloud_in->size () << " data points to input:" << std::endl;
      
  for (auto& point : *cloud_in)
    std::cout << point << std::endl;
      
  *cloud_out = *cloud_in;   // ？？
  
  std::cout << "size:" << cloud_out->size() << std::endl;
  for (auto& point : *cloud_out)
    point.x += 0.7f;

```

4. 创建icp实例后，输入`source`和`target`

```c++
  std::cout << "Transformed " << cloud_in->size () << " data points:" << std::endl;
      
  for (auto& point : *cloud_out)
    std::cout << point << std::endl;

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
 icp.setInputSource(cloud_in);
  icp.setInputTarget(cloud_out);
```

5. 调用align方法进行配准，Final存储进行变换后的source点云

```c++
  pcl::PointCloud<pcl::PointXYZ> Final;
  icp.align(Final);

// 获取迭代得分和变换矩阵
  std::cout << "has converged:" << icp.hasConverged() << " score: " <<    
  icp.getFitnessScore() << std::endl;
  std::cout << icp.getFinalTransformation() << std::endl;
```



# 编译和执行结果

1、创建CMakeLists.txt文件

```cmake
project(iterative_closest_point)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (iterative_closest_point iterative_closest_point.cpp)
target_link_libraries (iterative_closest_point ${PCL_LIBRARIES})
```

2、执行二进制`target`文件

```
$ ./iterative_closest_point
```

3、可看到输出结果： 

``` 
Saved 5 data points to input:
(0.352222,-0.151883,-0.106395)
(-0.397406,-0.473106,0.292602)
(-0.731898,0.667105,0.441304)
(-0.734766,0.854581,-0.0361733)
(-0.4607,-0.277468,-0.916762)
size:5
Transformed 5 data points:
(1.05222,-0.151883,-0.106395)
(0.302594,-0.473106,0.292602)
(-0.0318983,0.667105,0.441304)
(-0.0347655,0.854581,-0.0361733)
(0.2393,-0.277468,-0.916762)
has converged:1 score: 1.1956e-13
           1 -2.30968e-07 -2.98023e-08          0.7
 -2.1793e-07            1 -7.82311e-08  -1.2368e-07
 7.45058e-08  4.09782e-08            1   3.8743e-08
           0            0            0            1
```



# 总结

- pcl点云的遍历，可以通过auto遍历获取
- 要输出点云的值，可以直接cout<<激光点