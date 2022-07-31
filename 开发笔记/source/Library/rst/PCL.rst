.. role:: raw-html-m2r(raw)
   :format: html


PCL
===

`Install <https://pointclouds.org/downloads/>`_
---------------------------------------------------

.. prompt:: bash $,# auto

   $ sudo apt install libpcl-dev

CMake
-----

.. code-block:: cmake

   find_package(PCL QUIET)
   include_directories(${PCL_INCLUDE_DIRS})
   target_link_libraries(<target_name> ${PCL_LIBRARIES})

Header
------

.. code-block:: c++

   #include <pcl/io/pcd_io.h>
   #include <pcl/point_types.h>
   #include <pcl/filters/filter.h>

   // 有关transform
   #include <pcl/common/transforms.h>

Construct
---------

`智能指针和原始数据的相互转换 <https://blog.csdn.net/h287850870/article/details/80988552>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
   pcl::PointCloud<pcl::PointXYZ> cloud;
   cloud = *cloud_ptr;
   cloud_ptr = cloud.makeShared;

创建点云智能指针对象
^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   // pcl::PointCloud<pcl::PointXYZI>::Ptr pc_ptr_ = nullptr // 创建时初始化不用nullptr
   pcl::PointCloud<pcl::PointXYZI>::Ptr pc_ptr_(new pcl::PointCloud<pcl::PointXYZI>);

`Custom PointCloud Msg <https://github.com/RobustFieldAutonomyLab/LeGO-LOAM/blob/master/LeGO-LOAM/include/utility.h>`_
--------------------------------------------------------------------------------------------------------------------------


* 官方例程：\ `detail <https://pcl.readthedocs.io/projects/tutorials/en/latest/adding_custom_ptype.html#example>`_
* 解读

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220112190723590.png" alt="image-20220112190723590" style="zoom:80%;" />`

.. code-block:: cpp

   #include <pcl/memory.h>
   #include <pcl/pcl_macros.h>
   #include <pcl/point_types.h>
   #include <pcl/point_cloud.h>
   #include <pcl/io/pcd_io.h>

   struct CustomPointType
   {
       PCL_ADD_POINT4D
       PCL_ADD_INTENSITY;
       float roll;
       float pitch;
       float yaw;
       double time;
       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   } EIGEN_ALIGN16;

   POINT_CLOUD_REGISTER_POINT_STRUCT (CustomPointType,
                                      (float, x, x) (float, y, y)
                                      (float, z, z) (float, intensity, intensity)
                                      (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                      (double, time, time)
   )

Downsample
----------

.. code-block:: cpp

   typedef pcl::PointXYZ PointType;
   pcl::PointCloud<PointType>::Ptr fcloud;

   /// Applies Voxel Grid filter to cloud.
   /// in: leaf_size (size of voxel, in meters), cloud (to be filtered)
   /// out: (in class) fcloud (filtered cloud)
   void applyVoxelFilter(float leaf_size, pcl::PointCloud<PointType>::Ptr cloud);

   void applyVoxelFilter(float leaf_size, pcl::PointCloud<PointType>::Ptr cloud,
                         pcl::PointCloud<PointType>::Ptr fcloud) {
     fcloud.reset(new pcl::PointCloud <PointType>);

     pcl::VoxelGrid <PointType> vg;
     vg.setInputCloud(cloud);
     vg.setLeafSize(leaf_size, leaf_size, leaf_size);
     vg.filter(*fcloud);
   }

Geometry
--------

`KDtree <https://pcl.readthedocs.io/projects/tutorials/en/latest/kdtree_search.html#kdtree-search>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   #include <pcl/point_cloud.h>
   #include <pcl/kdtree/kdtree_flann.h>

   // 建树
   pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
   kdtree.setInputCloud(cloud);

   // 构建搜索点
   pcl::PointXYZ searchPoint;
   searchPoint.x = 1024.0f * rand () / (RAND_MAX + 1.0f);
   searchPoint.y = 1024.0f * rand () / (RAND_MAX + 1.0f);
   searchPoint.z = 1024.0f * rand () / (RAND_MAX + 1.0f);

   // K nearest neighbor search
   int K = 10;
   std::vector<int> pointIdxNKNSearch(K);  // 该向量有大小
   std::vector<float> pointNKNSquaredDistance(K);

   if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
   // todo

   std::vector<int> pointIdxRadiusSearch;
   std::vector<float> pointRadiusSquaredDistance;
   // 含半径约束的搜索
   if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)

Transform
^^^^^^^^^

.. code-block:: c++

   #include <pcl/common/transforms.h>

   // 可以含强度信息
   void Calibration::lidarToCameraPoints(pcl::PointCloud<PointType>::Ptr &pts){
     pcl::transformPointCloud(*pts, *pts, lidar_extrinsic_);
   }

Filter
------

Statistical Outlier Removal
^^^^^^^^^^^^^^^^^^^^^^^^^^^

对每个点找近邻点，该点称为核心点；认为\ **邻域点到核心点的距离差**\ （这个统计量）服从正态分布，若邻域点的距离差大于某个阈值则剔除掉该点

.. code-block:: c++

   #include <pcl/point_types.h>
   #include <pcl/filters/statistical_outlier_removal.h>

   typedef pcl::PointXYZ PointT
   int main(int argc, char **argv) {

     pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud <PointT>);
     pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud <PointT>);
     // Create the filtering object
     pcl::StatisticalOutlierRemoval <PointT> sor;
     sor.setInputCloud(cloud);
     // 样本数/领域点为50，标准差因子为1, query点的标准差大于1m时则认为是离群点
     sor.setMeanK(50);
     sor.setStddevMulThresh(1.0);
     sor.filter(*cloud_filtered);

     return (0);
   }

Crop Filter
^^^^^^^^^^^

.. code-block:: c++

   #include <pcl/filters/crop_box.h>
   #include <pcl/point_cloud.h>
   #include <pcl/point_types.h>
   #include <pcl_conversions/pcl_conversions.h>
   typedef pcl::PointXYZRGB PointT;

   // camera frame right, bottom, forward
   constexpr float min_range[3] = {-2.5, -2.0, 0.0};
   constexpr float max_range[3] = {2.5, 2.0, 3.0};
   constexpr float leaf_size = 0.01;

   pcl::PointCloud<PointT>::Ptr pointcloud_pcl(new pcl::PointCloud<PointT>);
   pcl::CropBox<PointT> crop;
   crop.setMin(Eigen::Vector4f(min_range[0], min_range[1], min_range[2], 1.0));
   crop.setMax(Eigen::Vector4f(max_range[0], max_range[1], max_range[2], 1.0));
   crop.setInputCloud(pointcloud_pcl);
   crop.setKeepOrganized(true);
   crop.filter(*pointcloud_pcl);

Subset
^^^^^^

一般调用pcl的分割算法后，返回的是相关的索引。因此需要根据索引去进一步提取感兴趣的点云。

.. code-block:: c++

   #include <pcl/filters/extract_indices.h>

   pcl::PointCloud<pcl::PointXYZ>::Ptr input(new pcl::PointCloud<pcl::PointXYZ>);
   pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
   // Create the filtering object
   pcl::ExtractIndices<pcl::PointXYZ> extract;
   // Extract the inliers
   extract.setInputCloud(input);
   // 点云索引
   extract.setIndices(index_ptr);
   // true：反相提取点云
   extract.setNegative(false);
   extract.filter(*output);

IO
--

`Write <https://pcl.readthedocs.io/projects/tutorials/en/latest/writing_pcd.html#writing-pcd>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211022000542975.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211022000542975.png
   :alt: image-20211022000542975



* 旧式接口

.. code-block:: c++

   // 以下均为以前的API，现已统一用savePCDFile来替代
   pcl::io::savePCDFileASCII("file.pcd", cloud);
   pcl::io::savePCDFileBinary("file.pcd", cloud);
   pcl::io::savePCDFileBinaryCompressed("file.pcd",cloud);

`Read <https://pcl.readthedocs.io/projects/tutorials/en/latest/reading_pcd.html#reading-pcd>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   typedef pcl::PointXYZ PointType;
   pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
   if (pcl::io::loadPCDFile<PointType>("file.pcd", *cloud) == -1) {
       PCL_ERROR("Couldn't read file\n");
       return (-1);
   }

Visualize
---------

Pseudo Color
^^^^^^^^^^^^


* 基于强度信息进行\ `分段线性拉伸 <https://blog.csdn.net/huqiang_823/article/details/81054507>`_\ 得到RGB，增强点云间的对比度，提高可视化效果，具体代码参考\ `实例 <https://github.com/Livox-SDK/livox_horizon_loam/blob/master/src/laserMapping.cpp#L168>`_

.. code-block:: c++

   void RGBpointAssociateToMap(PointType const *const pi,
                               pcl::PointXYZRGB *const po) {
     Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
     Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
     po->x = point_w.x();
     po->y = point_w.y();
     po->z = point_w.z();
     int reflection_map = pi->curvature * 10;
     if (reflection_map < 30) {
       int green = (reflection_map * 255 / 30);
       po->r = 0;
       po->g = green & 0xff;
       po->b = 0xff;
     } else if (reflection_map < 90) {
       int blue = (((90 - reflection_map) * 255) / 60);
       po->r = 0x0;
       po->g = 0xff;
       po->b = blue & 0xff;
     } else if (reflection_map < 150) {
       int red = ((reflection_map - 90) * 255 / 60);
       po->r = red & 0xff;
       po->g = 0xff;
       po->b = 0x0;
     } else {
       int green = (((255 - reflection_map) * 255) / (255 - 150));
       po->r = 0xff;
       po->g = green & 0xff;
       po->b = 0;
     }
   }

rviz
^^^^

.. code-block:: python

   def bgr_to_hex(color_np):
       """
       Args:
           color_np:{n,3} [b,g,r]
           b = color_np[:, 0]
           g = color_np[:, 1]
           r = color_np[:, 2]
       """

       rgb_arr = np.array((color_np[:, 2] << 16) | (color_np[:, 1] << 8) | \
                          (color_np[:, 0] << 0), dtype=np.uint32)
       # 实测上只能用这种方式修改，不能使用astype转换
       rgb_arr.dtype = np.float32
       return rgb_arr

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210911215651517.png" alt="image-20210911215651517" style="zoom: 67%;" />`

ROS
---

Convert
^^^^^^^

PCL to ROS
~~~~~~~~~~

.. code-block:: c++

   pcl::PointCloud<pcl::PointXYZRGB> colored_pointcloud;
   sensor_msgs::PointCloud2 output_msg;

   pcl::toROSMsg(colored_pointcloud, output_msg);
   output_msg.header = input.header;
   instance_pointcloud_pub_.publish(output_msg);

`Transform <http://docs.ros.org/en/indigo/api/pcl_ros/html/namespacepcl__ros.html#a34090d5c8739e1a31749ccf0fd807f91>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* ``ROS Pointcloud`` + ``ROS TF`` + ``Eigen TF``

.. code-block:: c++

   bool LidarApolloInstanceSegmentation::transformCloud(const sensor_msgs::PointCloud2& input, sensor_msgs::PointCloud2& transformed_cloud, float z_offset)
   {
     // transform pointcloud to target_frame
     if (target_frame_ != input.header.frame_id)
     {
       try
       {
         geometry_msgs::TransformStamped transform_stamped;
         // 得到target_frame_->input_frame的坐标系变换；input_frame在target_frame_的位姿；将input_frame的点云转换到target_frame_的坐标变换
         transform_stamped =
             tf_buffer_.lookupTransform(target_frame_, input.header.frame_id, input.header.stamp, ros::Duration(0.5));
         Eigen::Matrix4f affine_matrix = tf2::transformToEigen(transform_stamped.transform).matrix().cast<float>();
         pcl_ros::transformPointCloud(affine_matrix, input, transformed_cloud);
         transformed_cloud.header.frame_id = target_frame_;
       }
       catch (tf2::TransformException& ex)
       {
         ROS_WARN("%s", ex.what());
         return false;
       }
     }
     else
     {
       transformed_cloud = input;
     }

     // move pointcloud z_offset in z axis
     // 点云z数据 + z_offset
     sensor_msgs::PointCloud2 pointcloud_with_z_offset;
     Eigen::Affine3f z_up_translation(Eigen::Translation3f(0, 0, z_offset));
     Eigen::Matrix4f z_up_transform = z_up_translation.matrix();
     pcl_ros::transformPointCloud(z_up_transform, transformed_cloud, transformed_cloud);

     return true;
   }

.. note:: PCL有一个点云TF的接口；对于对点的坐标进行变换的话，不是用遍历的方案，而是采用矩阵相乘的方式


Q&A
---

反射强度与材料的关系
^^^^^^^^^^^^^^^^^^^^


* 以下资料参考速腾16线数据手册

.. list-table::
   :header-rows: 1

   * - 材料
     - 反射强度
   * - 黑色介质+漫反射
     - 约等于0
   * - 白色介质+漫反射
     - 小于100
   * - 半反透介质+镜面反射
     - 大于100
   * - 全反射
     - 等于255


Tool
----

CloudCompare
^^^^^^^^^^^^

`Install <http://www.cloudcompare.org/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. prompt:: bash $,# auto

   # 方法一：可以直接使用apt安装，但是不支持pcd点云文件的导入
   $ sudo apt install cloudcompare
   # 方法二：使用snap安装，但是需要更换到edge版本
   $ sudo snap install cloudcompare
   $ sudo snap refresh --edge cloudcompare

Q&A
~~~


* `为什么cloudcompare没有撤销操作 <http://www.danielgm.net/cc/forum/viewtopic.php?t=1257>`_
* `CloudCompare支持的文件格式 <https://www.cloudcompare.org/doc/wiki/index.php?title=FILE_I/O>`_

Reference
~~~~~~~~~


* `官方教程 <http://www.cloudcompare.org/tutorials.html>`_\ ：包括剔除点云（仅支持2D裁剪）、配准（自动配准、交互式配准：自己选配准点）

Reference
---------


* `pcl official wiki <https://pcl.readthedocs.io/projects/tutorials/en/latest/>`_
