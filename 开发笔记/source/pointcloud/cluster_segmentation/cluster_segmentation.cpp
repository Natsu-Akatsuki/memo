#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <string>

typedef pcl::PointXYZI PointType;

int main(int argc, char **argv) {

  // Read in the cloud data
  std::string file_name = "../data/livox_pointcloud.pcd";
  pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>),
      nonground_pointcloud(new pcl::PointCloud<PointType>);
  if (pcl::io::loadPCDFile<PointType>(file_name, *cloud) == -1) {
    PCL_ERROR("Couldn't read file\n");
    return (-1);
  }

  std::cout << "PointCloud before filtering has: " << cloud->size()
            << " data points." << std::endl; //*

  // 进行下采样
  pcl::VoxelGrid<PointType> vg;
  pcl::PointCloud<PointType>::Ptr cloud_filtered(
      new pcl::PointCloud<PointType>);
  vg.setInputCloud(cloud);
  vg.setLeafSize(0.1f, 0.1f, 0.1f);
  vg.filter(*cloud_filtered);
  std::cout << "PointCloud after filtering has: " << cloud_filtered->size()
            << " data points." << std::endl; //*

  // 除地面
  pcl::SACSegmentation<PointType> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointCloud<PointType>::Ptr cloud_plane(new pcl::PointCloud<PointType>());

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100);
  seg.setDistanceThreshold(0.02); // 2cm

  // Segment the largest planar component from the remaining cloud
  seg.setInputCloud(cloud_filtered);
  seg.segment(*inliers, *coefficients);
  if (inliers->indices.size() == 0) {
    std::cout << "Could not estimate a planar model for the given dataset."
              << std::endl;
  }

  // Extract the planar inliers from the input cloud
  pcl::ExtractIndices<PointType> extract;
  extract.setInputCloud(cloud_filtered);
  extract.setIndices(inliers);
  extract.setNegative(false);

  // Get the points associated with the planar surface
  extract.filter(*cloud_plane);
  std::cout << "PointCloud representing the planar component: "
            << cloud_plane->size() << " data points." << std::endl;

  // Remove the planar inliers, extract the rest
  extract.setNegative(true);
  extract.filter(*nonground_pointcloud);
  *cloud_filtered = *nonground_pointcloud;

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
  tree->setInputCloud(cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointType> ec;
  ec.setClusterTolerance(0.02); // 2cm
  ec.setMinClusterSize(100);
  ec.setMaxClusterSize(25000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud_filtered);
  ec.extract(cluster_indices);

  return (0);
}