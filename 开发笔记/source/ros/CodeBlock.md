# CodeBlock

pub.getNumSubscribers()

## [rostopic](http://docs.ros.org/en/diamondback/api/rostopic/html/)

- [取消订阅与发布](https://github.com/ros/ros_comm/blob/noetic-devel/tools/topic_tools/sample/simple_lazy_transport.py)

```bash
 # blocking会直接阻塞所在的线程
 ros_type = rostopic.get_topic_class("主题名",blocking=True)[0]
 self.sub_ = rospy.Subscriber("主题名", self.ros_type, <回调函数>)
```

## 获取handle

```c++
// 适合于nodelet
private_nh_ = getPrivateNodeHandle();
nh_ = getNodeHandle();
```

## 参数服务器数据交互

```c++
private_nh_.param<std::string>("target_frame", target_frame_, "base_link");
private_nh_.param<bool>("use_height", use_height_, false);
private_nh_.param<int>("min_cluster_size", min_cluster_size_, 3);
private_nh_.param<int>("max_cluster_size", max_cluster_size_, 200);
private_nh_.param<float>("tolerance", tolerance_, 1.0);
```

## 订阅与发布

```c++
pointcloud_sub_ = private_nh_.subscribe("input", 1, &EuclideanClusterNodelet::pointcloudCallback, this);
cluster_pub_ = private_nh_.advertise<autoware_perception_msgs::DynamicObjectWithFeatureArray>("output", 10);
debug_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("debug/clusters", 1);
```

- [回调函数的写法](https://wiki.ros.org/roscpp_tutorials/Tutorials/UsingClassMethodsAsCallbacks)

## rosnode simple

```c++
#include <ros/ros.h>

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "/*node_name*/");
  // ros::NodeHandle nh;
  // <class> node;
  ros::spin();
  return 0;
}
```

## 功能函数

### 返回两点间距离

```c++
/**
 * 返回两点间距离
 * @param point0 
 * @param point1 
 * @return 
 */
double DataAssociation::getDistance(const geometry_msgs::Point& point0, const geometry_msgs::Point& point1)
{
  const double diff_x = point1.x - point0.x;
  const double diff_y = point1.y - point0.y;
  // const double diff_z = point1.z - point0.z;
  return std::sqrt(diff_x * diff_x + diff_y * diff_y);
}
```

### 计算点云体心

```c++
/**
 * 计算点云体心
 * @param pointcloud
 * @return
 */
geometry_msgs::Point DataAssociation::getCentroid(const sensor_msgs::PointCloud2& pointcloud)
{
  geometry_msgs::Point centroid;
  centroid.x = 0;
  centroid.y = 0;
  centroid.z = 0;
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(pointcloud, "x"), iter_y(pointcloud, "y"),
       iter_z(pointcloud, "z");
       iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
  {
    centroid.x += *iter_x;
    centroid.y += *iter_y;
    centroid.z += *iter_z;
  }
  centroid.x = centroid.x / ((double)pointcloud.height * (double)pointcloud.width);
  centroid.y = centroid.y / ((double)pointcloud.height * (double)pointcloud.width);
  centroid.z = centroid.z / ((double)pointcloud.height * (double)pointcloud.width);
  return centroid;
}
```
