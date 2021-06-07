#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <chrono>

int
main(int argc, char **argv) {
    // 对象名(参数) 实例化 pcl::PointCloud<pcl::PointXYZ>::Ptr 类，argument需要是指针
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    auto start = std::chrono::steady_clock::now();

//  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd.pcd", *cloud) == -1) //* load the file
//  {
//    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
//    return (-1);
//  }
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("test_pcd_binary_1g.pcd", *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    return (0);
}