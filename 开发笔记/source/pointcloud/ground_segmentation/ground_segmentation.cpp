#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/impl/region_growing.hpp>
#include <pcl/segmentation/impl/sac_segmentation.hpp>
#include <pcl/visualization/cloud_viewer.h>

typedef pcl::PointXYZI PointType;
class GroundSegmentation {
public:
  int save_pointcloud(const std::string &file_name);
  int load_pointcloud(const std::string &file_name);
  int run();

  GroundSegmentation(float distance_thre, int iter_times = 100) {

    ground_pointcloud_.reset(new pcl::PointCloud<PointType>);
    nonground_pointcloud_.reset(new pcl::PointCloud<PointType>);
    input_pointcloud.reset(new pcl::PointCloud<PointType>);
    coefficients_.reset(new pcl::ModelCoefficients);

    seg_.setOptimizeCoefficients(true);
    seg_.setModelType(pcl::SACMODEL_PLANE);
    seg_.setMethodType(pcl::SAC_RANSAC);
    seg_.setDistanceThreshold(distance_thre);
    seg_.setMaxIterations(iter_times);
  }

private:
  pcl::PointCloud<PointType>::Ptr input_pointcloud;

  // ground segmentation parameter
  pcl::ModelCoefficients::Ptr coefficients_;
  pcl::SACSegmentation<PointType> seg_;
  pcl::PointCloud<PointType>::Ptr ground_pointcloud_;
  pcl::PointCloud<PointType>::Ptr nonground_pointcloud_;
};

int GroundSegmentation::load_pointcloud(const std::string &file_name) {

  if (pcl::io::loadPCDFile<PointType>(file_name, *input_pointcloud) == -1) {
    PCL_ERROR("Couldn't read file\n");
    return (-1);
  }
  return 0;
}

int GroundSegmentation::save_pointcloud(const std::string &file_name) {
  if (pcl::io::savePCDFile<PointType>(file_name, *nonground_pointcloud_,
                                      true) == -1) {
    PCL_ERROR("Couldn't read file\n");
    return (-1);
  }
  return 0;
}

int GroundSegmentation::run() {

  pcl::PointIndices::Ptr inliers_idx(new pcl::PointIndices);
  seg_.setInputCloud(input_pointcloud);
  seg_.segment(*inliers_idx, *coefficients_);

  // extract points
  pcl::ExtractIndices<PointType> extract;
  extract.setInputCloud(input_pointcloud);
  extract.setIndices(inliers_idx);

  extract.setNegative(false);
  extract.filter(*ground_pointcloud_);

  extract.setNegative(true);
  extract.filter(*nonground_pointcloud_);

  pcl::visualization::CloudViewer viewer("viewer");
  viewer.showCloud(ground_pointcloud_);
  while (!viewer.wasStopped()) {
  }
  return 0;
}

int main() {
  GroundSegmentation ground_segmentation(0.05, 100);
  ground_segmentation.load_pointcloud("../livox_pointcloud.pcd");
  ground_segmentation.run();
  ground_segmentation.save_pointcloud("livox_pointcloud_nonground.pcd");
  return 0;
}