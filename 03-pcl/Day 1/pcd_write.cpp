//
// Created by helios on 2021/2/2.
//

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int
main(int argc, char **argv) {
    pcl::PointCloud<pcl::PointXYZ> cloud;

    // Fill in the cloud data
    cloud.width = 40000000;
    cloud.height = 1;
    cloud.is_dense = false;
    cloud.points.resize(cloud.width * cloud.height);  // 激光点的个数

    for (auto &point: cloud) {
        point.x = 1024 * rand() / (RAND_MAX + 1.0f);
        point.y = 1024 * rand() / (RAND_MAX + 1.0f);
        point.z = 1024 * rand() / (RAND_MAX + 1.0f);
    }

//    pcl::io::savePCDFileASCII("test_pcd.pcd", cloud);
    pcl::io::savePCDFileBinary("test_pcd_binary_1g.pcd", cloud);
//    pcl::io::savePCDFileBinary("test_pcd_binary.pcd", cloud);
//    pcl::io::savePCDFileBinaryCompressed("test_pcd_binary_compressed.pcd", cloud);
    std::cerr << "Saved " << cloud.size() << " data points to pcd." << std::endl;

    // for (const auto &point: cloud)
    //     std::cerr << "    " << point.x << " " << point.y << " " << point.z << std::endl;

    return (0);
}