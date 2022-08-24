Migration
==================
Executor
********
- `ROS1 <http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29>`_
- `ROS2 <https://docs.ros.org/en/humble/Concepts/About-Executors.html>`_
- `ROS2 Node Tutorial <https://roboticsbackend.com/write-minimal-ros2-cpp-node/>`_

.. tabs::

   .. code-tab:: c++ ROS1

      #include "ros/ros.h"

      int main(int argc, char **argv) {
         ros::init(argc, argv, "talker");
         ros::NodeHandle n;

         ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
         ros::Rate loop_rate(10);

         int count = 0;
         while (ros::ok()) {
            ros::spinOnce();
            loop_rate.sleep();
         }

         return 0;
      }

   .. code-tab:: c++ ROS1-OOP

      #include "ros/ros.h"

      // 类
      int main(int argc, char **argv) {
         ros::init(argc, argv, "tensorrt_yolo");
         object_recognition::TensorrtYoloNodelet tensorrt_yolo;
         ros::spin();
         return 0;
      }


   .. code-tab:: c++ ROS2

      #include "rclcpp/rclcpp.hpp"

      int main(int argc, char *argv[]) {
         // Some initialization.
         rclcpp::init(argc, argv);
         // TODO
         // Instantiate a node.
         rclcpp::Node::SharedPtr node = ...;
         // Run the executor.
         rclcpp::spin(node);

         // spin部分可拓展为：
         // rclcpp::executors::SingleThreadedExecutor executor;
         // executor.add_node(node);
         // executor.spin();

         // Shutdown and exit.
         // TODO
         return 0;
      }

Header
********

.. tabs::

   .. code-tab:: c++ ROS1

      #include "ros/ros.h"
      #include "std_msgs/String.h"  // MSG
      #include <image_transport/image_transport.h>

   .. code-tab:: c++ ROS2

      #include "rclcpp/rclcpp.hpp"
      #include "std_msgs/msg/string.hpp"
      #include <image_transport/image_transport.hpp>



Logger
********

.. tabs::

   .. code-tab:: c++ ROS1


      NODELET_INFO("Initializing nodelet TemplatePackageNodelet..."); // only for nodelet
      ROS_INFO("Publishing: '%s'", message.data.c_str());

   .. code-tab:: c++ ROS2

      RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());

   .. code-tab:: python ROS2-Python

      # 给某个节点设置日志等级
      rclpy.logging.set_logger_level('image_publisher', rclpy.logging.LoggingSeverity.WARN)

Launch
*******

.. tabs::

   .. code-tab:: python ROS2-Python

      from launch import LaunchDescription
      from launch_ros.actions import Node

      def generate_launch_description():
         return LaunchDescription([
            Node(
                  package='turtlesim',
                  namespace='turtlesim1',
                  executable='turtlesim_node',
                  name='sim'
            ),
            Node(
                  package='turtlesim',
                  namespace='turtlesim2',
                  executable='turtlesim_node',
                  name='sim'
            ),
            Node(
                  package='turtlesim',
                  executable='mimic',
                  name='mimic',
                  remappings=[
                     ('/input/pose', '/turtlesim1/turtle1/pose'),
                     ('/output/cmd_vel', '/turtlesim2/turtle1/cmd_vel'),
                  ]
            )
         ])

CMake
********
- `ROS2-ii <https://zhuanlan.zhihu.com/p/438191834>`_

.. tabs::

   .. code-tab:: cmake ROS1
      
      # TODO

   .. code-tab:: cmake ROS2-ii


      # ament_cmake升级版

      # 导入宏包
      find_package(ament_cmake_auto REQUIRED)

      # 根据package.xml导入相关的依赖
      ament_auto_find_build_dependencies()

      # 等价于add_library + 导入相关头文件（from ros package） + 导入相关依赖库（from ros package）
      # add_executable()
      # target_include_directories()
      # target_link_libraries()
      ament_auto_add_executable(main src/main.cpp)

      # 导出相关的配置文件和进行安装
      ament_auto_package()


Publisher and Subscriber
********************************
- `ROS1 <http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29>`_
- `ROS2 <https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html>`_

.. tabs::

   .. code-tab:: cmake ROS1

      ros::Publisher publisher = n.advertise<std_msgs::String>("topic", 10);

   .. code-tab:: cmake ROS2

      rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
      publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);


Image Transport
****************
- `ROS1 <https://wiki.ros.org/image_transport>`_
- `ROS2 Migraion <https://github.com/ros-perception/image_common/wiki/ROS2-Migration>`_

.. tabs::

   .. code-tab:: c++ ROS1

      // Use the image_transport classes instead.
      #include <ros/ros.h>
      #include <image_transport/image_transport.h>

      void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
      // ...
      }

      ros::NodeHandle nh;
      image_transport::ImageTransport it(nh);
      image_transport::Subscriber sub = it.subscribe("in_image_base_topic", 1, imageCallback);
      image_transport::Publisher pub = it.advertise("out_image_base_topic", 1);

      // 类方法（其API需要bind）
      image_transport::Subscriber sub =
         it.subscribe("in/image", 1, std::bind(&TensorrtYoloNodelet::callback, this, _1));

   .. code-tab:: c++ ROS2

      #include "rclcpp/rclcpp.hpp"
      #include <image_transport/image_transport.hpp>

      image_transport::Subscriber image_sub_ = image_transport::create_subscription(
         this, "in/image", std::bind(&TensorrtYoloNodelet::callback, this, _1), "raw",
         rmw_qos_profile_sensor_data);
      image_transport::Publisher pub = image_transport::create_publisher(this, "out/image");