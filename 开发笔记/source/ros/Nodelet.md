# Nodelet

## standalone nodelet

from tier4@euclidean_cluster_node.cpp

```c++
#include <nodelet/loader.h>
#include <ros/ros.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "euclidean_cluster_node");
  ros::NodeHandle private_nh("~");

  nodelet::Loader nodelet;
  nodelet::M_string remap(ros::names::getRemappings());
  nodelet::V_string nargv;
  std::string nodelet_name = ros::this_node::getName();
  nodelet.load(nodelet_name, "euclidean_cluster/euclidean_cluster_nodelet", remap, nargv);
  ros::spin();
  return 0;
}
```

from tier4@CMakeLists.txt

```cmake
add_executable(euclidean_cluster_node src/euclidean_cluster_node.cpp)
target_link_libraries(euclidean_cluster_node
  euclidean_cluster
  ${catkin_LIBRARIES}
)
```
