#include <pcl/io/pcd_io.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

struct CustomPointType {
  PCL_ADD_POINT4D; // preferred way of adding a XYZ+padding
  float test;
  PCL_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16; // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(
    // (type, name, tag)
    CustomPointType, (float, x, x)(float, y, y)(float, z, z)(float, test, test))

int main(int argc, char **argv) {
  pcl::PointCloud<CustomPointType> cloud;
  cloud.points.resize(2);
  cloud.width = 2;
  cloud.height = 1;
  cloud[0].test = 1;
  cloud[1].test = 2;
  cloud[0].x = cloud[0].y = cloud[0].z = 0;
  cloud[1].x = cloud[1].y = cloud[1].z = 3;

  pcl::io::savePCDFile("output.pcd", cloud, true);
}