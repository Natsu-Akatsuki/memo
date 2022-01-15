#include <chrono>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
typedef pcl::PointXYZI PointType;

ros::Subscriber pointcloud_sub;
ros::Publisher pointcloud_pub;

void pointcloud_cb(const sensor_msgs::PointCloud2ConstPtr &pointcloud_msg) {

  {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; i++) {
      pcl::PointCloud<PointType>::Ptr pointcloud_ros(
          new pcl::PointCloud<PointType>());
      pcl::fromROSMsg(*pointcloud_msg, *pointcloud_ros);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "fromROSMsg: elapsed time: " << elapsed_seconds.count()
              << "s\n";
  }

  {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; i++) {
      float point[24000];
      for (sensor_msgs::PointCloud2ConstIterator<float>
               iter_x(*pointcloud_msg, "x"),
           iter_y(*pointcloud_msg, "y"), iter_z(*pointcloud_msg, "z"),
           iter_i(*pointcloud_msg, "intensity");
           iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_i) {
        point[0] = *iter_x;
        point[1] = *iter_y;
        point[2] = *iter_z;
        point[3] = *iter_i;
      }
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "PointCloud2ConstIterator: elapsed time: "
              << elapsed_seconds.count() << "s\n";
  }

  pcl::PointCloud<PointType> pcl_msg;
  sensor_msgs::PointCloud2 ros_msg;
  pcl::toROSMsg(pcl_msg, ros_msg);
  ros_msg.header = pointcloud_msg->header;
  pointcloud_pub.publish(ros_msg);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "pointcloud_subscriber_publisher");
  ros::NodeHandle nh;
  pointcloud_sub =
      nh.subscribe<sensor_msgs::PointCloud2>("/livox/lidar", 1, pointcloud_cb);
  pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/output", 1);
  ros::spin();
}
