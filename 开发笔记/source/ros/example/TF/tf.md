# TF

## 静态TF

```xml
<!-- static_transform_publisher x y z yaw pitch roll 父 子坐标系 -->
frame_id child_frame_id
child_frame_id到frame_id
<launch>
	<node pkg="tf2_ros" type="static_transform_publisher" name="base_link_to_lidar" args="1.0, 0, 0, 0, 0, 0 lidar base_link" />
	<node pkg="tf2_ros" type="static_transform_publisher" name="camera_to_base_link" args="0 0 0 -1.570795 -1.570795 0 base_link camera"/>
    <node pkg="tf" type="static_transform_publisher" name="rack_link_to_base_link"  args="0.27 0 1.09 0 0 0 /base_link /rack_link 1000" />
</launch>
```

.. note:: 此处的to指relative to

.. attention:: TF2是没有时间间隔参数的

## 动态TF

- TF2，发布和监听的主题同时有`/TF`和`/tf_static`  

```
try{
	// ros::Time(0)：在buffer中可利用的最新的TF数据
    transformStamped = tfBuffer.lookupTransform("turtle2", "turtle1", ros::Time(0));
  } catch (tf2::TransformException &ex) {
    ROS_WARN("Could NOT transform turtle2 to turtle1: %s", ex.what());
  }
  

try{
	// ros::Time::now()：当前时刻的TF数据
	transformStamped = tfBuffer.lookupTransform("turtle2", "turtle1", ros::Time::now());
  } catch (tf2::TransformException &ex) {
	ROS_WARN("Could NOT transform turtle2 to turtle1: %s", ex.what());
  }
```





```
/**
 * 计算TF变换
**/
bool LidarApolloInstanceSegmentation::transformCloud(
  const sensor_msgs::PointCloud2 & input,
  sensor_msgs::PointCloud2& transformed_cloud,
  float z_offset)
{
  // transform pointcloud to target_frame
  if (target_frame_ != input.header.frame_id) {
    try {
      geometry_msgs::TransformStamped transform_stamped;
      transform_stamped = tf_buffer_.lookupTransform(target_frame_, input.header.frame_id,
                                                     input.header.stamp, ros::Duration(0.5));
	  // 齐次变换矩阵                                                     
      Eigen::Matrix4f affine_matrix =
        tf2::transformToEigen(transform_stamped.transform).matrix().cast<float>();
      pcl_ros::transformPointCloud( , input, transformed_cloud);
      transformed_cloud.header.frame_id = target_frame_;
    } catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
      return false;
    }
  } else {
    transformed_cloud = input;
  }
  
  // move pointcloud z_offset in z axis
  sensor_msgs::PointCloud2 pointcloud_with_z_offset;
  Eigen::Affine3f z_up_translation(Eigen::Translation3f(0, 0, z_offset));
  Eigen::Matrix4f z_up_transform = z_up_translation.matrix();
  pcl_ros::transformPointCloud(z_up_transform, transformed_cloud, transformed_cloud);

  return true;
}  
```



# 变换矩阵

```
void pcl_ros::transformPointCloud
( 	const Eigen::Matrix4f & transform,
	const sensor_msgs::PointCloud2 & in,
	sensor_msgs::PointCloud2 & out 
)	
```





# 注意事项

- 一个子系只能有一个父系；一个父系可以有多个子系