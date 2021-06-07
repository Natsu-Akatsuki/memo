 

# 读/写PCD文件以读/存点云数据

## 写PCD文件

https://pcl.readthedocs.io/projects/tutorials/en/latest/writing_pcd.html#writing-pcd

### 代码

1、创建文件 `pcd_write.cpp` 

```c++
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int
  main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;

  // Fill in the cloud data
  cloud.width    = 5;
  cloud.height   = 1;
  cloud.is_dense = false;
  cloud.points.resize (cloud.width * cloud.height);  // 激光点的个数

  for (auto& point: cloud)
  {
    point.x = 1024 * rand () / (RAND_MAX + 1.0f);
    point.y = 1024 * rand () / (RAND_MAX + 1.0f);
    point.z = 1024 * rand () / (RAND_MAX + 1.0f);
  }

  pcl::io::savePCDFileASCII ("test_pcd.pcd", cloud);
  std::cerr << "Saved " << cloud.size () << " data points to test_pcd.pcd." << std::endl;

  for (const auto& point: cloud)
    std::cerr << "    " << point.x << " " << point.y << " " << point.z << std::endl;

  return (0);
}
```



### 代码解构

1、第一个头文件包含了PCD I/O操作的`definitions` 定义；第二个包含了一些点云数据结构的定义，如`pcl::PointXYZ`

```c++
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
```

2、创建了一个点云模板类。每个激光点的类型是`pcl::PointXYZ`（一个含x, y, z`field`字段的结构体）

```
  pcl::PointCloud<pcl::PointXYZ> cloud;
```

3、用随机值给点云数据填值

```c++
  // Fill in the cloud data
  cloud.width    = 5;
  cloud.height   = 1;
  cloud.is_dense = false;
  cloud.points.resize (cloud.width * cloud.height);

  for (auto& point: cloud)   // c++11的range-based for loop语法
  {
    point.x = 1024 * rand () / (RAND_MAX + 1.0f);
    point.y = 1024 * rand () / (RAND_MAX + 1.0f);
    point.z = 1024 * rand () / (RAND_MAX + 1.0f);
  }
```

4、将数据保存到磁盘

```
  pcl::io::savePCDFileASCII ("test_pcd.pcd", cloud);
```

5、输出日志，看是否成功保存点云

```
  std::cerr << "Saved " << cloud.size () << " data points to test_pcd.pcd." << std::endl;

  for (const auto& point: cloud)
    std::cerr << "    " << point.x << " " << point.y << " " << point.z << std::endl;
```



### 编译和运行代码

1、创建CMakeLists.txt文件

```cmake
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcd_write)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pcd_write pcd_write.cpp)
target_link_libraries (pcd_write ${PCL_LIBRARIES})
```

2、执行二进制`target`文件

```
$ ./pcd_write
```

3、可看到输出结果

```
Saved 5 data points to test_pcd.pcd.
  0.352222 -0.151883 -0.106395
  -0.397406 -0.473106 0.292602
  -0.731898 0.667105 0.441304
  -0.734766 0.854581 -0.0361733
  -0.4607 -0.277468 -0.916762
```

4、可使用cat查看pcd文件的内容

```
$ cat test_pcd.pcd
# .PCD v.5 - Point Cloud Data file format
FIELDS x y z
SIZE 4 4 4
TYPE F F F
WIDTH 5
HEIGHT 1
POINTS 5
DATA ascii
0.35222 -0.15188 -0.1064
-0.39741 -0.47311 0.2926
-0.7319 0.6671 0.4413
-0.73477 0.85458 -0.036173
-0.4607 -0.27747 -0.91676
```



## 读PCD文件

### 代码

创建 pcd_read/pcd_read.cpp源文件

```c++
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int
main (int argc, char** argv)
{
  // 对象名(参数) 实例化 pcl::PointCloud<pcl::PointXYZ>::Ptr 类，argument需要是指针
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);  

  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd.pcd", *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }
  std::cout << "Loaded "
            << cloud->width * cloud->height
            << " data points from test_pcd.pcd with the following fields: "
            << std::endl;
  for (const auto& point: *cloud)
    std::cout << "    " << point.x
              << " "    << point.y
              << " "    << point.z << std::endl;

  return (0);
}
```



### 代码解构

1、创建 PointCloud<PointXYZ> `boost shared pointer` boost共享指针并进行初始化

```c++
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
```

2、从`disk`磁盘中读取点云数据

```c++
  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd.pcd", *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }
```





Alternatively, you can read a PCLPointCloud2 blob (available only in PCL 1.x). Due to the dynamic nature of point clouds, we prefer to read them as binary blobs, and then convert to the actual representation that we want to use.

```
pcl::PCLPointCloud2 cloud_blob;
pcl::io::loadPCDFile ("test_pcd.pcd", cloud_blob);
pcl::fromPCLPointCloud2 (cloud_blob, *cloud); //* convert from pcl/PCLPointCloud2 to pcl::PointCloud<T>
```

reads and converts the binary blob into the templated PointCloud format, here using pcl::PointXYZ as the underlying point type.

3、最后输出数据

```c++
  for (const auto& point: *cloud)
    std::cout << "    " << point.x
              << " "    << point.y
              << " "    << point.z << std::endl;
```



### 编译和运行代码

1、创建CMakeLists.txt文件

```
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcd_read)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pcd_read pcd_read.cpp)
target_link_libraries (pcd_read ${PCL_LIBRARIES})
```

2、执行二进制`target`文件

```
$ ./pcd_read
```

3、将看到相关的输出如下

```
Loaded 5 data points from test_pcd.pcd with the following fields: x y z
  0.35222 -0.15188 -0.1064
  -0.39741 -0.47311 0.2926
  -0.7319 0.6671 0.4413
  -0.73477 0.85458 -0.036173
  -0.4607 -0.27747 -0.91676
```

若待读取的pcd文件不存在，则会提示如下报错信息：

```
Couldn't read file test_pcd.pcd
```



## 自定义点云类型

- https://pcl.readthedocs.io/projects/tutorials/en/latest/adding_custom_ptype.html

```c++
/*

   * A point cloud type that has "ring" channel
     */
     struct PointXYZIR
     {
         PCL_ADD_POINT4D
         PCL_ADD_INTENSITY;
         uint16_t ring;
         EIGEN_MAKE_ALIGNED_OPERATOR_NEW
     } EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIR,  
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (uint16_t, ring, ring)
)
```



## 拓展知识点

### add_definitions(${PCL_DEFINITIONS})的作用？

- 参考：https://www.cnblogs.com/li-yao7758258/p/9041916.html、https://cmake.org/cmake/help/v3.18/command/add_definitions.html?highlight=add_definitions

-  该命令行可用于添加任意flags，但一般用在编译阶段，来提供`preprocessor definitions` 宏定义

- ```cmake
  add_definitions("-DTEST_ENABLE")
  ```

  这句话的意思就是如果源程序中定义了

  ```c++
  #ifdef TEST_ENABLE 
  ......
  #endif
  ```

  则中间的代码就会生效



### range-based loop

- 参考：https://www.cnblogs.com/Nothing-9708071624/p/10167982.html

- 想要拷贝元素：for(auto x:range)

​    想要修改元素：for(auto &&x:range)

​    想要只读元素：for(const auto& x:range)                                                                                                                       





## TODO

###  point_types 和 point_cloud.h的区别





# 拓展案例

## 1. [pcd_reader和loadPCDFile没有区别](https://blog.csdn.net/weixin_46098577/article/details/112280924)

## 2. 不同存储方式的读取速度比较

- 以100w个点为测试案例

|                                           | 读取  |                                                            |
| ----------------------------------------- | ----- | ---------------------------------------------------------- |
| savePCDFileASCII（ASCII码的格式保存点云） | 34s   |                                                            |
| savePCDFileBinary（二进制的格式保存点云） | 250ms |                                                            |
| savePCDFileBinaryCompressed               | 640ms | 导出文件相比原始二进制的会小些，但读取时间会更长，原因未知 |
| savePCDFile(默认ASCII码保存)              |       |                                                            |
