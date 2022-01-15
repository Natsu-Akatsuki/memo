#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <string>
typedef pcl::PointXYZI PointType;

constexpr int num_point_dims = 4;
bool readBinFile(const std::string &filename) {

  pcl::PointCloud<PointType>::Ptr pointcloud(new pcl::PointCloud<PointType>);
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file) {
    std::cerr << "Could not open the file " << filename << std::endl;
    return false;
  }

  using namespace std;

  // 获取文件大小：设置输入流的位置，挪到文件末尾和文件头
  file.seekg(0, std::ios::end);
  auto file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  auto start = std::chrono::steady_clock::now();

  // 点云个数
  size_t point_num =
      static_cast<unsigned int>(file_size) / sizeof(float) / num_point_dims;

  {
    PointType point;
    for (size_t i = 0; i < point_num; ++i) {
      // 将number of characters存放到某个char buffer中
      file.read((char *)&point.x, 3 * sizeof(float));
      // 需要使用intensity，不能直接使用 4 * sizeof(float)
      file.read((char *)&point.intensity, sizeof(float));
      pointcloud->push_back(point);
    }
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
  if (pcl::io::savePCDFile<PointType>("output.pcd", *pointcloud, true) == -1) {
    PCL_ERROR("Couldn't read file\n");
    return (-1);
  }

  file.close();
  return 0;
}

int main() {
  // pcl和file system都能够使用相对路径
  readBinFile("../data/rslidar16_pointcloud.bin");
  return 0;
}