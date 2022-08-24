ROS1
====

`Install <http://wiki.ros.org/noetic/Installation>`_
--------------------------------------------------------

Deployment
----------

..

   There is likely **a lot** more **learning curve** than catkin_make install + copy binaries. @\ `answers.ros <https://answers.ros.org/question/226581/deploying-a-catkin-package/>`_


Metapackage
-----------

作用不大，基本用不上


* 方案一：\ `answer.ros.org <https://answers.ros.org/question/322340/creating-metapackage/>`_

.. prompt:: bash $,# auto

   catkin_create_pkg <MY_META_PACKAGE> --meta


* 方案二：\ `ros-wiki教程 <http://wiki.ros.org/catkin/package.xml#Metapackages>`_

直接添加一些细节到 ``CMakeLists`` 和 ``package.xml`` 上



* `命令行rqt-bag <http://wiki.ros.org/rqt_bag>`_\ ：一个rosbag录制和回放，查看topic的图形化程序

Tutorial
--------

Beginner
^^^^^^^^

CLI
~~~~~


* ``rosrun``

.. prompt:: bash $,# auto

   # 对于python文件，或需要添加可执行权限
   $ rosrun <package_name> <executable>


* ``rosbag``

.. prompt:: bash $,# auto

   # 回放
   $ rosbag play <包名>
   # 只发布自己想要的主题
   $ rosbag play school.bag --topic /rslidar_points

   # 主题重映射
   $ rosbag play school.bag /rslidar_points:=/velodyne_points
   # --clock     # publish the clock time
   # -r <n>      # 以n倍速播放

   # 录制
   $ rosbag record <主题名>

   # 裁剪
   # 这种时刻指的是ros时间戳，类似 1576119471.511449
   $ rosbag filter <输入包名> <输出包名> "t.to_sec() < 某个时刻 and t.to_sec() > 某个时刻"

   # 压缩和解压
   $ rosbag compress/decompress <待压缩的包名>


* ``rosnode`` （\ `官方文档 for python <http://docs.ros.org/en/hydro/api/rosnode/html/>`_\ ）

.. code-block:: python

   # rosnode list
   import rosnode
   node_list = rosnode.get_node_names()

   # rosnode kill
   node_list = rosnode.get_node_names()
   _, _ = rosnode.kill_nodes(node_list)


* ``rostopic``

.. prompt:: bash $,# auto

   $ rostopic list       # 查看当前发布和订阅的主题
   $ rostopic type <topic_name> # 查看主题的类型
   $ rostopic echo <topic_name> # 查看主题中的数据


`发布数据时带时间戳 <http://wiki.ros.org/ROS/YAMLCommandLine#Headers.2Ftimestamps>`_：

.. prompt:: bash $,# auto

   $ rostopic pub /topic_name topic_type [args...]  # 发布数据
   # options
   # -r: 指定发布的频率
   # -f: 从yaml文件中读取args
   # -s: 需配合-r模式使用，可使用auto和now这两个词的substitution

   # example
   $ rostopic pub -s -r 4 /clicked_point geometry_msgs/PointStamped "header: auto
   point:
     x: 0.0
     y: 0.0
     z: 0.0"
   $ rostopic pub -s --use-rostime -r 4 /clicked_point geometry_msgs/PointStamped "header:
     seq: 0
     stamp: now
     frame_id: ''
   point:
     x: 0.0
     y: 0.0
     z: 0.0"

.. attention::  ``-s`` 实测只能替换命令行中的keyword；使用上 ``-f`` 时，只能替换第一次的数据


.. code-block:: python

   # rostopic list
   import rospy
   topic_list = rospy.get_published_topics()


*
  roslaunch ：\ `官方文档 for python <http://docs.ros.org/en/kinetic/api/roslaunch/html/index.html>`_\ ，\ `官方文档 wiki with example <http://wiki.ros.org/roslaunch/API%20Usage>`_\ （该API支持中文路径）

*
  rospack：\ `官方文档 for python <http://docs.ros.org/en/independent/api/rospkg/html/python_api.html>`_

.. prompt:: bash $,# auto

   # 返回某个包的绝对路径
   $ rospack find <pkg>

.. tabs::

   .. code-tab:: c++ C++

      // 获取某个package的绝对路径
      #include <ros/package.h>
      std::string path = ros::package::getPath("package_name");

   .. code-tab:: python Python

      # 获取某个package的绝对路径
      from rospkg import RosPack
      rp = RosPack()
      path = rp.get_path('package_name')


- roswtf

.. prompt:: bash $,# auto

   # 可用于知悉哪些节点的哪些主题没有订阅成功
   $ roswtf

Namespace
~~~~~~~~~~

.. tabs::

   .. code-tab:: c++ C++

      // c++中描述的节点名不包含命名空间，无'/'(e.g即没有/ns/node，只有node)
      ros::init(argc, argv, "节点名");

   .. code-tab:: xml launch

      <!-- launch中的节点名也不包含命名空间 -->
      <node pkg="talker" type="talker" name="talker"/>

      <!-- 可以通过group tag或ns attribute来添加命名空间 -->
      <!-- 前者可同时给多个节点附上一个命名空间，后者则针对具体的一个，前者可被后者覆盖-->
      <group ns="namespaceA">
         <node ns='namespaceB' pkg="talker" type="talker" name="talker"/>
      </group>


* topic

  * 当创建的节点有命名空间时，base类型（e.g. ``node_name``\ ，而非\ ``/.../node_name``\ ）的topic会附上节点的命名空间
  * 当创建的句柄带有(``~``)时，base类型的topic除了附上节点的命名空间，还会附上节点名作为命名空间


Node
~~~~~

.. code-block:: cpp

   #include <ros/ros.h>

   int main(int argc, char* argv[])
   {
   ros::init(argc, argv, "/*node_name*/");
   // ros::NodeHandle nh;
   // <class> node;
   ros::spin();
   return 0;
   }

Parameter Server
~~~~~~~~~~~~~~~~
.. code-block:: c++

   private_nh_.param<std::string>("target_frame", target_frame_, "base_link");
   private_nh_.param<bool>("use_height", use_height_, false);
   private_nh_.param<int>("min_cluster_size", min_cluster_size_, 3);
   private_nh_.param<int>("max_cluster_size", max_cluster_size_, 200);
   private_nh_.param<float>("tolerance", tolerance_, 1.0);

Subscriber and Publisher
~~~~~~~~~~~~~~~~~~~~~~~~~

* `取消订阅与发布 <https://github.com/ros/ros_comm/blob/noetic-devel/tools/topic_tools/sample/simple_lazy_transport.py>`_
* 设置有订阅才发布主题

.. code-block:: c++

   if (pub_.getNumSubscribers() < 1) return;

* 订阅器和发布器

.. code-block:: c++

   // 已知主题名
   pointcloud_sub_ = private_nh_.subscribe("input", 1, &EuclideanClusterNodelet::pointcloudCallback, this);
   cluster_pub_ = private_nh_.advertise<autoware_perception_msgs::DynamicObjectWithFeatureArray>("output", 10);
   debug_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("debug/clusters", 1);

   // 未知主题名
   // blocking会直接阻塞所在的线程
   ros_type = rostopic.get_topic_class("主题名",blocking=True)[0]
   self.sub_ = rospy.Subscriber("主题名", self.ros_type, <回调函数>)


*
  `回调函数的写法 <https://wiki.ros.org/roscpp_tutorials/Tutorials/UsingClassMethodsAsCallbacks>`_

*
  同时接收多个数据进行处理

.. code-block:: c++

   // 导入相关头文件
   #include "message_filters/subscriber.h"
   #include "message_filters/sync_policies/approximate_time.h"
   #include "message_filters/synchronizer.h"

   // 声明回调函数
   void objectsCallback(
       const autoware_perception_msgs::DynamicObjectWithFeatureArray::ConstPtr & input_object0_msg,
    const autoware_perception_msgs::DynamicObjectWithFeatureArray::ConstPtr & input_object1_msg);

   // 定义订阅器
   message_filters::Subscriber<autoware_perception_msgs::DynamicObjectWithFeatureArray> object0_sub_(pnh_, "input/object0", 1);
   message_filters::Subscriber<autoware_perception_msgs::DynamicObjectWithFeatureArray> object1_sub_(pnh_, "input/object1", 1);

   // 定义同步机制
   typedef message_filters::sync_policies::ApproximateTime<
       autoware_perception_msgs::DynamicObjectWithFeatureArray, autoware_perception_msgs::DynamicObjectWithFeatureArray>
       SyncPolicy;

   typedef message_filters::Synchronizer<SyncPolicy> Sync
   Sync sync_;


TF2
~~~~

* roslaunch发布静态TF

.. code-block:: xml

   <!-- static_transform_publisher x y z yaw pitch roll 父 子坐标系 -->
   <node pkg="tf2_ros" type="static_transform_publisher" name="camera_to_lidar" args="0, 0, 0, 0, 0, 0 lidar camera" />


* 查看\ ``TF``\ 树

.. prompt:: bash $,# auto

   $ rosrun rqt_tf_tree rqt_tf_tree

   # sudo apt install ros-noetic-tf2-tools
   $ rosrun tf2_tools view_frames.py

`Time <http://wiki.ros.org/roscpp/Overview/Time>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: cpp

   // 获取ROS系统下的时间戳
   ros::Time begin = ros::Time::now();

   // Timer 回调函数
   publish_timer_ = nh_.createTimer(ros::Duration(1.0 / publish_rate), &Callback, this);

Intermidiate
^^^^^^^^^^^^^^^^

Callback Function
~~~~~~~~~~~~~~~~~~~~

- `sensor_msgs::ImageConstPtr <https://docs.ros.org/en/diamondback/api/sensor_msgs/html/namespaces.html>`_\ 是什么类型数据？

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220206215812571.png" alt="image-20220206215812571" style="zoom: 67%;" />`

.. code-block:: c++

   // 共享指针应用案例
   void TensorrtYolo::callback(const sensor_msgs::Image::Ptr& in_image_msg)
   void TensorrtYolo::callback(const sensor_msgs::Image::Ptr in_image_msg)

`Conda <https://github.com/RoboStack/ros-noetic>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
对于跨版本而言，体验一般，毕竟不能直接使用apt安装的二进制包

`cv_bridge <http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 将数据从opencv->ros时，一般采用 ``bgr`` 编码方式（opencv原本的数据默认即bgr通道，不管是读还是写）

.. code-block:: c++

   cv_bridge::CvImagePtr cv_ptr;
   try
   { // 不提供第二个参数时将等效于"passthrough"，不对图片进行变换
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
   }
   catch (cv_bridge::Exception& e)
   {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
   }

   // Update GUI Window
    image_pub_.publish(cv_ptr->toImageMsg());

Spin, Subscriber and Publisher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

暂时没找到权威的资料，以下为结合参考资料和自己理解的版本


* ``queue_size``\ 对应的是\ ``publisher queue size``\ （待发布数据的缓存队列）和\ ``subscriber queue size``\ （待处理的接收数据的缓存队列）

* `rospy和roscpp spin的区别？ <https://get-help.robotigniteacademy.com/t/what-is-rospy-spin-ros-spin-ros-spinonce-and-what-are-they-for/58>`_

  ``rospy.spin()`` 只是起阻塞作用（自旋锁/忙等），防止主进程结束

  roscpp中的 ``spin`` 和 ``spinonce`` 一方面起阻塞作用，另一方面用于调用回调函数

* 发布器的数据处理逻辑？

  调用\ ``pubish()``\ 时，发布器线程（\ ``publisher thread``\ ）会将相关的原始数据放到发布器队列（\ ``publisher queue``\ ），如果队列已满则丢弃旧的数据

  自旋线程\ ``spinner thread``\ 根据发布器队列中对应的数据，对数据进行序列化和进行发布

  默认情况下发布器队列是共用的

* 订阅器的数据处理逻辑？

  接收器线程（\ ``receiver thread``\ ）接收到的\ ``序列化数据``\ 之放到各自的订阅器队列中（\ ``subscriber queue``\ ）中，如果队列已满则丢弃旧的数据

  自旋线程（\ ``spinner thread``\ ）根据订阅器队列中对应的数据，对数据进行反序列化和调用相关的回调函数

* 回调函数队列和发布器队列/订阅器队列的区别？

  都是队列（先进先出）

  发布器队列存储的是待发布的数据（ the publisher queue is another queue like callback queue, but the queue is for queuing published message which is filled every time ``publish()`` function is called.）（估计暂未进行序列化）

* `基于多线程的回调函数 <http://wiki.ros.org/roscpp/Overview/Callbacks%20and%20Spinning>`_

.. code-block:: c++

   ros::MultiThreadedSpinner spinner(4); // Use 4 threads
   spinner.spin(); // spin() will not return until the node has been shutdown

* 只处理最新的数据

  在ros中，可能会遇到一些很耗时的操作，比如点云配准，图像特征提取。这样的话，回调函数的处理时间就会变得很长。如果发布端发布数据的频率高于订阅端处理的速度，同时订阅端没有限制地处理所有的数据的话，就会使订阅端一直处理较旧的数据。最终的数据和数据的处理之间的时延将会很高。希望处理最新的数据的话，就需要将发布器和订阅器的队列长度设置为1。

  如下为图像处理时队列长度不为1的效果图（左为输出效果，右为输入图像，可看出有较大的时延）（实测：inference时间和ros image数据传输耗时为ms级别）

.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/latency.gif
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/latency.gif
   :alt: img


* rospy回调函数的多线程处理机制

.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/rospy-cb-multithread.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/rospy-cb-multithread.png
   :alt: rospy-cb-multithread.png

``rospy`` 中处理回调函数时会派生出一个新的线程去执行（线程名与主题名相同）；如果有n个回调函数（处理的是不同的topic）则会派生出n个线程；如果有回调函数处理相同的topic则共用一个线程

.. attention:: 此处阐述的是 ``rospy`` 的回调函数的机制，在 ``roscpp`` 中会有所不同

* 参考资料：

  `知乎 <https://zhuanlan.zhihu.com/p/375418691>`_

  `外语博客 <https://levelup.gitconnected.com/ros-spinning-threading-queuing-aac9c0a793f>`_

  `ROS1 订阅器和发布器官方资料 <http://wiki.ros.org/rospy/Overview/Publishers%20and%20Subscribers>`_

  `队列长度的设置 csdn <https://blog.csdn.net/qq_32618327/article/details/121650164>`_

数据同步
~~~~~~~~


* 在ROS里面对点云数据和GNSS数据进行融合，可能采用如下的方式进行融合：用单个变量存储待融合的数据。这种融合方法，并不能保证融合时两个数据的时间戳是接近的。

.. code-block:: c++

   void gnss_callback(ros_gnss_data) {
     gnss_data_ = ros_gnss_data;
   }

   void gnss_callback(ros_lidar_data) {
     lidar_data_ = ros_lidar_data;
     fuse(lidar_data_, gnss_data_);
   }


`Network Setup <http://wiki.ros.org/ROS/NetworkSetup>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. code-tab:: bash 主机配置

      $ export ROS_MASTER_URI=http://<master_machine_ip>:11311

   .. code-tab:: bash 从机配置

      $ export ROS_MASTER_URI=http://<master_machine_ip>:11311

`基于ros环境导入某个package下的python包 <https://roboticsbackend.com/ros-import-python-module-from-another-package/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

步骤一：创建python模板，相应的目录树如下

.. code-block:: plain

   └── directory_name        # 一般可以设置为package_name
       ├── CMakeLists.txt
       ├── package.xml
       ├── setup.py
       └── src
           └── module_name      # 一般设置为module_name
               ├── import_me_if_you_can.py
               └── __init__.py

步骤二：编写 ``setup.py`` 文件

.. code-block:: python

   ## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

   from distutils.core import setup
   from catkin_pkg.python_setup import generate_distutils_setup

   # fetch values from package.xml
   setup_args = generate_distutils_setup(
       packages=['module_name'],
       package_dir={'': 'src'},
       requires=['rospy']
   )

   setup(**setup_args)

步骤三：编写 ``CMakeLists.txt`` 文件

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.13)
   project(project_name)
   find_package(catkin REQUIRED COMPONENTS rospy)
   # 调用当前CMakeLists文件所在目录下的setup.py
   catkin_python_setup()
   catkin_package()


Rviz
~~~~~~~~~~~~

* `应用案例官方教程 <https://github.com/autolaborcenter/rviz_navi_multi_goals_pub_plugin.git>`_\ （含display, panel, tool的自定义设置）

* （自定义插件）继承rviz:: Panel类

.. code-block:: c++

   class TeleopPanel: public rviz:: Panel{

       ...

   }

* `给自定义插件添加icon <https://answers.ros.org/question/213971/how-to-add-an-icon-to-a-custom-display-type-in-rviz/>`_：只需要在icon\ **s**\ /class\ **es**\ 目录下添加icon.png文件即可（icon文件名需同插件名）

* rviz Qt (for python)：**用完一圈之后，不推荐使用这个rviz的python api，一是文档太少，难以进行开发，二是坑很多**\ 。比如退出Qt应用程序后，rviz节点将成为僵尸节点（即不能被rosnode kill掉，只能使用rosnode cleanup清理），而在实测中c++中不存在这个问题，进程可以退出得很干净；另外实测不能够在Qt中的rviz中添加图像面板，否则会有段错误（暂时没有解决方案）

Advance
^^^^^^^^^^

Nodelet
~~~~~~~~

ros节点的通信是进程的通信，采用ros tcp的方法。当节点间传输的数据体量较大，通信（比如要反序列和序列化）的开销将比较大。因此若\ **希望减少节点间通讯的开销来提高实时性**\ ，这就需要用到nodelet技术。具体例子，比如跑一些点云的预处理模块，涉及到采集的点云转换成ros数据，点云滤波去离群点，点云裁剪，点云去地面等节点，这些节点允许放在一个进程作为单独的线程去跑（ ``ros nodelet`` 程序能将一个 ``node`` 进程节点转换为 ``nodelet`` 线程节点），然后想提高实时性就可以用到。总体给人一种将进程通信转换为线程通信的感觉。


* `CLI <http://wiki.ros.org/nodelet#Helper_tools>`_

.. prompt:: bash $,# auto

   $ rospack plugins --attrib=plugin nodelet   # 显示.xml文件
   $ rosrun nodelet declared_nodelets          # 显示ros工作空间中已有的nodelet

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210810223516109.png" alt="image-20210810223516109" style="zoom:67%; " />`

* plugin文件


nodelet与plugin密切相关，其中ros中的插件（`plugin <[pluginlib](http://wiki.ros.org/pluginlib/Tutorials/Writing%20and%20Using%20a%20Simple%20Plugin>`_\）即动态库中的可动态加载的类


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210811003457276.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210811003457276.png
   :alt: image-20210811003457276


:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210515175724200.png" alt="image-20210515175724200" style="zoom:67%; " />`

* nodelet的启动


步骤一：一般而言，每个nodelet需要一个 ``NodeletManager`` 来管理，启动 ``NodeletManager`` 的方法：

.. prompt:: bash $,# auto

   # 同时重命名NodeletManager
   $ rosrun nodelet nodelet manager __name:=nodelet_manager

等价于在launch文档中的：

.. code-block:: xml

   <node pkg="nodelet" type="nodelet" name="nodelet_manager" args="manager"/>

步骤二：加载 ``nodelet`` 到 ``NodeletManager``

.. prompt:: bash $,# auto

   # Launch a nodelet of type pkg/type(包名/xml文件中的class name) on manager manager
   $ rosrun nodelet nodelet load nodelet_tutorial_math/Plus nodelet_manager

等价于在launch文档中的：

.. code-block:: xml

   <node pkg="nodelet" type="nodelet" name="Plus" args="load nodelet_tutorial_math/Plus nodelet_manager"/>

* nodelet launch文档解读

.. code-block:: xml

   <!--都需要启动nodelet包的nodelet可执行文件，不过相应的启动参数不一样-->
   <node pkg="nodelet" type="nodelet" name="euclidean_cluster_manager" args="manager" output="screen" />

   <node pkg="nodelet" type="nodelet" name="$(anon voxel_grid_filter)" args="load pcl/VoxelGrid euclidean_cluster_manager" output="screen">
   </node>

   <node pkg="nodelet" type="nodelet" name="$(anon euclidean_cluster)" args="load euclidean_cluster/voxel_grid_based_euclidean_cluster_nodelet euclidean_cluster_manager" output="screen">
   </node>

   <!--standalone nodelet，不需要加载到nodelet manager，相关于启动一个普通node-->
   <node pkg="nodelet" type="nodelet" name="Plus3" args="standalone nodelet_tutorial_math/Plus">
   </node>

* standalone nodelet template

.. tabs::

   .. code-tab:: c++ C++

      // from tier4@euclidean_cluster_node.cpp
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

   .. code-tab:: cmake CMake

      // from tier4@CMakeLists.txt
      add_executable(euclidean_cluster_node src/euclidean_cluster_node.cpp)

      target_link_libraries(euclidean_cluster_node
         euclidean_cluster ${catkin_LIBRARIES})

Q&A
---

rospy.init_node()为什么在主线程才能调用？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 因为\ ``rospy.init_node()``\ 时会引入（\ ``register``\ ）信号回调函数（\ ``signal handlers``\ ），而python中引入信号回调函数需要在主线程中完成（python特性）；不引入信号回调函数则可以在非主线程中调用\ ``rospy.init_node()``


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210909214309037.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210909214309037.png
   :alt: image-20210909214309037

* Test Example

.. code-block:: python

   import rospy
   import threading

   class Node(threading.Thread):
       def __init__(self):
           threading.Thread.__init__(self)

       def run(self):
           rospy.init_node('node')
           rospy.spin()

   if __name__ == '__main__':
       node = Node()
       node.start()
       node.join()

   # >>> ValueError: signal only works in main thread


* 官方实现

.. code-block:: python

   # File "/opt/ros/noetic/lib/python3/dist-packages/rospy/core.py", line 623, in register_signals
   # #687
   def register_signals():
       """
       register system signal handlers for SIGTERM and SIGINT
       """
       _signalChain[signal.SIGTERM] = signal.signal(signal.SIGTERM, _ros_signal)
       _signalChain[signal.SIGINT]  = signal.signal(signal.SIGINT, _ros_signal)

TroubleShooting
---------------

* `ros wiki trouble shooting <http://roswiki.autolabor.com.cn/rospy(2f)Troubleshooting.html>`_ ：含\ ``ctrl+C``\ 和\ ``import``\ 等问题

Tools
-----

.. list-table::
   :header-rows: 1

   * - 工具
     - 描述
   * - `RTUI <https://github.com/eduidl/rtui>`_
     - ROS TUI（用于快速查看节点/主题的相关信息），pip安装