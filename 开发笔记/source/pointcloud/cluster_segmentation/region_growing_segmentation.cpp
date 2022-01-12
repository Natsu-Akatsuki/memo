#include <iostream>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/visualization/cloud_viewer.h>
#include <vector>

typedef pcl::PointXYZI PointType;
int main(int argc, char **argv) {
  pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
  if (pcl::io::loadPCDFile<PointType>("../data/livox_pointcloud.pcd", *cloud) ==
      -1) {
    std::cout << "Cloud reading failed." << std::endl;
    return (-1);
  }

  // 计算法向量
  pcl::search::Search<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

  pcl::NormalEstimation<PointType, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(cloud);
  normal_estimator.setKSearch(50);
  normal_estimator.compute(*normals);

  pcl::IndicesPtr indices(new std::vector<int>);
  pcl::PassThrough<PointType> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0.0, 1.0);
  pass.filter(*indices);

  // 基于区域生长的提取
  pcl::RegionGrowing<PointType, pcl::Normal> reg;
  reg.setMinClusterSize(50);
  reg.setMaxClusterSize(1000000);
  reg.setSearchMethod(tree);
  reg.setNumberOfNeighbours(30);
  reg.setInputCloud(cloud);
  reg.setInputNormals(normals);
  reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold(1.0);
  std::vector<pcl::PointIndices> clusters;
  reg.extract(clusters);

  std::cout << "Number of clusters is equal to " << clusters.size()
            << std::endl;

  // 获取彩色的聚类点云团
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
  pcl::visualization::CloudViewer viewer("Cluster viewer");
  viewer.showCloud(colored_cloud);
  while (!viewer.wasStopped()) {
  }

  return (0);
}