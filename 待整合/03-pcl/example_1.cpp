//
// Created by helios on 2021/2/4.
//
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <string>


using namespace std;

int read_pcd(const string &file_name, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {

    if (pcl::io::loadPCDFile<pcl::PointXYZI>(file_name, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }
}

int main() {

    // pcd 点云文件的读取
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    string file_name = "/home/helios/1611500404.4752169/pcd/0.pcd";
    if (read_pcd(file_name, cloud) != -1) {
        cout << "pcd 文件读取成功" << endl;
    }

    // 点云可视化
    pcl::visualization::CloudViewer viewer ("pointcloud viewer");
    viewer.showCloud(cloud);
    while (!viewer.wasStopped ())
    {
    }
    return 0;
}