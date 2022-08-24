
ROS2
====

`Install <https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html>`_
----------------------------------------------------------------------------------------

for humble(test in ubuntu22.04)

Debian Package
^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ sudo apt update && sudo apt install curl gnupg lsb-release
   $ sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   $ echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null$ sudo apt update

   $ sudo apt install ros-humble-desktop

   # 添加环境变量到~/.bashrc
   $ echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

源码编译
^^^^^^^^

.. prompt:: bash $,# auto

   # 安装相关开发工具
   $ sudo apt update && sudo apt install -y \
     build-essential \
     cmake \
     git \
     python3-colcon-common-extensions \
     python3-flake8 \
     python3-flake8-blind-except \
     python3-flake8-builtins \
     python3-flake8-class-newline \
     python3-flake8-comprehensions \
     python3-flake8-deprecated \
     python3-flake8-docstrings \
     python3-flake8-import-order \
     python3-flake8-quotes \
     python3-pip \
     python3-pytest \
     python3-pytest-cov \
     python3-pytest-repeat \
     python3-pytest-rerunfailures \
     python3-rosdep \
     python3-setuptools \
     python3-vcstool \
     wget

   # 获取源代码
   $ mkdir -p ~/ros2_humble/src
   $ cd ~/ros2_humble
   $ wget https://raw.githubusercontent.com/ros2/ros2/humble/ros2.repos
   $ vcs import src < ros2.repos

   # 安装相关依赖
   $ sudo rosdep init
   $ rosdep update
   $ rosdep install --from-paths src --ignore-src -y --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers"

   # 编译工程
   $ cd ~/ros2_humble/
   $ colcon build --symlink-install

   # 添加环境变量
   $ echo "source ~/ros2_humble/install/local_setup.bash" >> ~/.bashrc

验证
^^^^


* 验证安装是否成功

.. prompt:: bash $,# auto

   $ ros2 run demo_nodes_cpp talker


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/L3PpIaqrMi15ctrj.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/L3PpIaqrMi15ctrj.png
   :alt: img


Tutorial
--------

Beginner
^^^^^^^^

CLI
~~~

大体上就是将ros1的命令行进行拆分

.. prompt:: bash $,# auto

   # ros1: rosnode info
   $ ros2 node info
   # 创建一个ros2包
   $ ros2 pkg create
   $ ros2 run <pkg_name> <node_name>
   $ ros2 node list
   $ ros2 topic list

   # 录制数据（ros2导出的是一个文件夹）
   $ ros2 bag play <包目录>
   $ ros2 bag record -a

Code
~~~~


* 更好的内存管理：订阅器、发布器对象这些都有智能指针

.. code-block:: c++

   #include "rclcpp/rclcpp.hpp"
   #include "std_msgs/msg/string.hpp"

   // 该函数在类中（继承了Node类）
   // 日志
   RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());

   // 继承
   class MinimalPublisher : public rclcpp::Node

   // 发布器
   rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
   publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);

`自定义消息/服务类型 <https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html#>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* 
  步骤一：构建消息/服务文件（\ ``.msg`` / ``.srv``\ ）（加上action的话，统称为\ `interface <https://docs.ros.org/en/humble/Concepts/About-ROS-Interfaces.html>`_\ ）

* 
  步骤二：CMakeLists.txt

.. code-block:: cmake

   # 导入特定的功能包
   find_package(rosidl_default_generators REQUIRED)

   # attention: 一定需要为${PROJECT_NAME}
   rosidl_generate_interfaces(${PROJECT_NAME}
     "msg/Num.msg"
     "srv/AddThreeInts.srv"
   )


* 步骤三：package.xml

.. code-block:: xml

   <build_depend>rosidl_default_generators</build_depend>
   <exec_depend>rosidl_default_runtime</exec_depend>

   <!-- TODO -->
   <member_of_group>rosidl_interface_packages</member_of_group>


* 步骤四：相关调用

`Roslaunch <https://docs.ros.org/en/foxy/How-To-Guides/Launch-file-different-formats.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Intermidiate
^^^^^^^^^^^^

`Rosbag convert <https://ternaris.gitlab.io/rosbags/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* 实现ROS1和ROS2包的相互转换

.. prompt:: bash $,# auto

   $ pip install rosbags

   # ros1 -> ros2 Convert "foo.bag", result will be "foo/"
   $ rosbags-convert foo.bag
   # ros1 -> ros2  Convert "foo.bag", save the result as "bar"
   $ rosbags-convert foo.bag --dst /path/to/bar

   # ros2 -> ros1 Convert "bar", result will be "bar.bag"
   $ rosbags-convert bar
   # ros2 -> ros1 Convert "bar", save the result as "foo.bag"
   $ rosbags-convert bar --dst /path/to/foo.bag

Advance
^^^^^^^

`ros1_bridge <https://github.com/ros2/ros1_bridge/blob/master/README.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用ros1_bridge可实现ros2和ros1主题的双向交互

`Nodelet <https://docs.ros.org/en/humble/Tutorials/Intermediate/Composition.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* ros2中的nodelet称为\ ``component``
* ``component``\ 是被编译到共享库的，所以没有主函数
* 
  其一般为 ``rclcpp::Node``\ 的子类

* 
  CLI

.. prompt:: bash $,# auto

   # 查看当前工作空间现有的component
   $ ros2 component types

   # 启动一个component的container（运行一个component container进程）
   $ ros2 run rclcpp_components component_container
   # 查看已启动的container
   $ ros2 component list
   # /ComponentManager

   # Load a component into a container node
   #container_node_name package_name plugin_name
   $ ros2 component load /ComponentManager composition composition::Talker

Real-time Programming
~~~~~~~~~~~~~~~~~~~~~


* `实时系统的介绍，ROS2考虑实时系统的原因 <https://design.ros2.org/articles/realtime_background.html>`_

Reference
^^^^^^^^^


* `初级教程：for CLI <https://docs.ros.org/en/foxy/Tutorials/Beginner-CLI-Tools.html>`_
* 
  `初级教程：for client library <https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html>`_\ （\ `订阅器和发布器 <https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html>`_\ 、\ `参数服务器 <https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-CPP.html>`_\ ）

* 
  进阶教程：（\ `nodelet <https://docs.ros.org/en/humble/Tutorials/Intermediate/Composition.html>`_\ ）
