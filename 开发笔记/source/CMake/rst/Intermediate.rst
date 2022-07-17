.. role:: raw-html-m2r(raw)
   :format: html


Intermediate
============


.. raw:: html

   <p align="right">Author: kuzen, Natsu-Akatsuki</p>



* ``build system``\ 和\ ``build tools``\ 是不同的概念
* ``build tool``\ 的作用单元是一系列的package，能够构建包的依赖关系图从而根据依赖关系，为每个包调用特定的\ ``build system``

Build System
------------

Make
^^^^

`make uninstall <https://gitlab.kitware.com/cmake/community/-/wikis/FAQ#can-i-do-make-uninstall-with-cmake>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

默认不提供make uninstall，需要自己定义。相关内容等价于：

.. prompt:: bash $,# auto

   # 但并不能删除相关的文件夹
   $ xargs rm < install_manifest.txt

CMake
^^^^^

Install
~~~~~~~


* ``apt`` 下载

.. prompt:: bash $,# auto

   # linux 18.04对应3.10版本
   $ sudo apt-get install cmake


* `源码下载 <https://cmake.org/download/>`_

.. prompt:: bash $,# auto

   # e.g.
   $ wget https://github.com/Kitware/CMake/releases/download/v3.18.3/cmake-3.18.3.tar.gz 
   # 要安装cmake-qt-gui时需要添加如下option
   $ ./bootstrap --qt-gui

CMake参数
~~~~~~~~~

.. prompt:: bash $,# auto

   # Wno-dev非gcc的编译参数，常应用于屏蔽PCL的警告
   $ cmake -Wno-dev

使用conda下的cmake文件
~~~~~~~~~~~~~~~~~~~~~~

在conda环境安装了相关包之后，需要conda activate才能使用其cmake文件，如果不activate的话，需要类似如下类型的参数配置

.. prompt:: bash $,# auto

   # 以pybind11为例 
   -Dpybind11_DIR=${env_path}/share/cmake/pybind11

判断一个路径对应的是否是一个文件夹
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. prompt:: bash $,# auto

   if(IS_DIRECTORY "...")

调用系统可执行文件
~~~~~~~~~~~~~~~~~~


* `find_program <https://cmake.org/cmake/help/latest/command/find_program.html>`_\ ：类似which，找到某个可执行文件的路径
* execute_process：执行某个可执行文件

.. prompt:: bash $,# auto

   # 判断某个可执行文件是否存在
   find_program(GDOWN_AVAIL "gdown")
   if (NOT GDOWN_AVAIL)
     message("...")
   endif()

   # 执行某个可执行文件
   execute_process(COMMAND mkdir [args...])
   execute_process(COMMAND gdown [args...])

引入外部项目
~~~~~~~~~~~~


* `FetchContent <https://cmake.org/cmake/help/latest/module/FetchContent.html>`_

该command为3.11的特性，会在configure time时导入(pollute)文件

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.14)

   # 导入FetchContent module
   include(FetchContent)
   # 配置等下Fetch时的配置参数
   FetchContent_Declare(
     mycom_toolchains
     URL  https://intranet.mycompany.com//toolchains_1.3.2.tar.gz
   )
   # 触发下载(Fetch)
   FetchContent_MakeAvailable(mycom_toolchains)

.. hint:: 对于cmake，configure time是指生成cache文件的时间段；有三种time，分别是配置期(configure time)，编译期(build time)和安装期(install time)；配置期的命令包括add_subdirectory, include, file


.. todo:: 暂未清楚不同期导入文件所带来的结果


`获取上层目录 <https://cmake.org/cmake/help/latest/command/get_filename_component.html?highlight=get_filename_component>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

   get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)

.. note:: 在include_directory填路径时使用".."也能生效


ROS Build Tool
--------------

ROS编译工具根据迭代顺序依次有： ``catkin_make``\ ，\ ``catkin_make_isolated``\ ， ``catkin_tools`` ， ``ament_tools``\ ，\ ``colon``

catkin_make
^^^^^^^^^^^

CLI
~~~

.. prompt:: bash $,# auto

   # 单独编译某些package
   $ catkin_make -DCATKIN_WHITELIST_PACKAGES="package1;package2"
   # 等价于：
   $ catkin_make --only-pkg-with-deps
   # 撤销白名单设置
   $ catkin_make -DCATKIN_WHITELIST_PACKAGES=""

   # 使用ninja进行编译（编译速度会更快，但报错信息无高亮，日志可读性差）
   $ catkin_make --use-ninja

.. note:: 要屏蔽某些包被编译，可以创建一个名为 `CATKIN_IGNORE`的文件到这些包所在的目录下


`catkin-tools <https://catkin-tools.readthedocs.io/en/latest/index.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Install <https://catkin-tools.readthedocs.io/en/latest/installing.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. prompt:: bash $,# auto

   # 添加ROS仓库
   $ ...

   $ sudo apt-get update
   $ sudo apt-get install python3-catkin-tools

CLI
~~~


* build（编译）

.. prompt:: bash $,# auto

   # 跳过对某些已编译包的编译（实际上只是检查）
   $ catkin build --start-with <pkg>
   # 编译当前所处的包
   $ catkin build --this


* config（配置参数）

.. prompt:: bash $,# auto

   # 配置编译参数
   $ catkin config -DPYTHON_EXECUTABLE=/opt/conda/bin/python3 \
   -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.8 \
   -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.8.so
   # 追加配置参数
   $ catkin config -a <配置参数>
   # 移除配置参数
   $ catkin config -r <配置参数>

   # 使用catkin_make参数
   $ catkin config --catkin-make-args [args]

   # 配置白名单（或黑名单）
   $ catkin config --whitelist/blacklist <pkg>
   # 取消白名单配置
   $ catkin config --no-whitelist


* `缓存Environment来提高编译速度 <https://catkin-tools.readthedocs.io/en/latest/verbs/catkin_config.html?highlight=cache#accelerated-building-with-environment-caching>`_

.. prompt:: bash $,# auto

   $ catkin config/build --env-cache
   $ catkin config/build --no_env_cache


* clean（清理中间文件）

.. prompt:: bash $,# auto

   # 指定删除某个package
   $ catkin clean <package_name>
   # 删除所有 product 
   $ catkin clean --deinit
   # 移除非src文件夹下的包的编译产物 
   $ catkin clean --orphans

.. note:: catkin clean 默认删除 ``devel`` , ``log`` 等目录，但隐藏目录 ``.catkin_tools`` , ``.catkin_workspace`` 不会清除



* `profile <https://catkin-tools.readthedocs.io/en/latest/verbs/catkin_profile.html>`_\ ：尚未明晰可用的场景

`Deploy a catkin package <https://answers.ros.org/question/226581/deploying-a-catkin-package/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`colcon <https://colcon.readthedocs.io/en/released/user/quick-start.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Install <https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html#>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. prompt:: bash $,# auto

   # 安装
   $ sudo apt install python3-colcon-common-extensions

   # 配置跳转
   $ echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> ~/.bashrc \
   && echo "export _colcon_cd_root=/opt/ros/humble/" >> ~/.bashrc

   # 配置命令行Tab补全
   $ echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc

   # 配置clean拓展插件
   $ git clone https://github.com/ruffsl/colcon-clean
   $ python3 setup.py install --user

CLI
~~~


* `build <https://colcon.readthedocs.io/en/released/user/how-to.html>`_

.. prompt:: bash $,# auto

   # 编译工作空间的所有pkg
   $ colcon build
   # 只编译部分包
   $ colcon build -packages-select <pkg_name>
   # 使用符号链接而不是复制文件进行安装
   $ colon build --symlink-install

   # option:
   # --cmake-args -DCMAKE_BUILD_TYPE=Debug
   # --event-handlers console_direct+   编译时显示所有编译信息
   # --event-handlers console_cohesion+  编译完一个包后才显示它的编译信息
   # --packages-select <name-of-pkg>  编译某个特定的包（不包含其依赖）
   # --packages-up-to <name-of-pkg>   编译某个特定的包（包含其依赖）
   # --packages-above <name-of-pkg>  重新编译某个包（和依赖这个包的相关包）

   # source devel/setup.bash的等价命令
   $ source install/local_setup

.. note:: 暂未发现其支持像 ``catkin build`` 中的 ``context-aware`` 功能



* list

.. prompt:: bash $,# auto

   # 显示当前工作空间的所有包的信息
   $ colcon list
   # List all packages in the workspace in topological order and visualize their dependencies
   $ colcon graph

Debug
-----

could not find a package configuration file（catkin build）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210912141918386.png" alt="image-20210912141918386" style="zoom: 80%; " />`

检查一：检查一波 ``package.xml`` 是否写好了\ ``build tag``

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/AYu9WKlHPlES5yu7.png!thumbnail" alt="img" style="zoom:67%; " />`

检查二：若使用catkin build的话检查一波是否将find_package(catkin REQUIRED...)放置于第三方库find_package的前面（具体原因未知，此为经验性结论）

/usr/bin/ld: cannot find
^^^^^^^^^^^^^^^^^^^^^^^^


* 在使用TensorRT部署时（make）出现如下的一些报错

.. prompt:: bash $,# auto

   /usr/bin/ld: cannot find -lnvonnxparser
   /usr/bin/ld: cannot find -lnvinfer_plugin 
   /usr/bin/ld: cannot find -lcudnn

一种解决方案为使用环境变量 ``LIBRARY_PATH`` 。此前认为时需要修改环境变量 ``LD_LIBRARY_PATH`` ，添加动态库链接搜索路径，但实际上该环境变量只影响运行期（runtime）链接器 ``ld.so`` 的搜索路径。而不影响编译期（complie time）链接器 ``/usr/bin/ld`` 的搜索路径。要影响编译期链接的话，需要修改环境变量 ``LIBRARY_PATH``

.. prompt:: bash $,# auto

   env LIBRARY_PATH=/usr/local/cuda/lib64:${HOME}/application/TensorRT-8.0.0.3/lib make

另一种解决方案为在CMakeLists上增设：

.. code-block:: cmake

   # e.g.
   link_directories(/usr/local/cuda/lib64/ $ENV{HOME}/application/TensorRT-8.0.0.3/lib)


* 
  拓展资料


  * `ld和ld.so命令的区别 <https://blog.csdn.net/jslove1997/article/details/108033399>`_
  * `stackoverflow answer <https://stackoverflow.com/questions/61016108/collect2-error-ld-returned-1-exit-status-lcudnn>`_


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/U9PWBBMXKy4vBo31.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/U9PWBBMXKy4vBo31.png!thumbnail
   :alt: img



.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/FvUyBNAT1nHvGPiG.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/FvUyBNAT1nHvGPiG.png!thumbnail
   :alt: img



* `查找动态链接库的顺序 for runtime <https://man7.org/linux/man-pages/man8/ld.so.8.html>`_

No CMAKE_CXX_COMPILER could be find
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ sudo apt install build-essential

未定义的引用（undefined reference）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该种错误发生在\ **链接**\ 时期。一般来说有以下几种情况。一种是没下载相关的链接库（可locate检测一下）；一种是库的冲突，比如ros的opencv库与从源码编译安装到系统的opencv库发生冲突，至依赖被覆盖而使目标文件无法成功链接到库。可卸载安装到系统的opencv库（如用sudo make uninstall来卸载）；一种是已下载但没找到，添加相关搜素路径即可

imported target \"...\" references the file \"...\" but this file does not exist
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`locate 定位相关位置后，使用软链接 <https://blog.csdn.net/weixin_45617478/article/details/104513572>`_

no such file or directory：没有找到头文件的路径，导入头文件失败
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在已有头文件的情况下，可直接添加绝对路径进行搜索；\ `或者头文件名不对，进行修改即可 <https://github.com/RobustFieldAutonomyLab/LeGO-LOAM/issues/219>`_

.. code-block:: cmake

   # e.g. include/utility.h:13:10: fatal error: opencv2/cv.h: No such file or directory #include <opencv2/cv.h>
   include_directories(
      include
      绝对路径   # e.g. /home/helios/include
   )

目标文件命名冲突（for catkin）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

rslidar和velodyne package的目标文件重名


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/M5KhRzVvmtcWapDQ.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/M5KhRzVvmtcWapDQ.png!thumbnail
   :alt: img


找不到cuda库和tensorrt库相关文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在autoware中，使用有关深度学习的cmake时，不能直接通过find_package找到cuda库和tensorRT；autoware配置环境时是使用deb包来安装的，会随带着将cmake等文件也安装到系统路径中；而如果使用的是local的安装方式，则find_package失效时，可参考如下方法进行添加：

.. code-block:: cmake

   include_directories($ENV{HOME}/application/TensorRT-7.2.3.4/include/)
   link_directories($ENV{HOME}/application/TensorRT-7.2.3.4/lib)
   `

`Failed to compute shorthash for libnvrtc.so <https://blog.csdn.net/xzq1207105685/article/details/117400187>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在CMakeList.txt开头添加\ ``find_package(PythonInterp REQUIRED)``

`ROS中编译通过但是遇到可执行文件找不到的问题 <https://blog.csdn.net/u014157968/article/details/86516797>`_\ ：指令顺序的重要性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* ``catkin_package``\ 要放在\ ``add_executable``\ 前，\ `案例（松灵底盘） <https://github.com/agilexrobotics/agx_sdk/issues/1>`_

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/BdZu0UoMbhAAPawe.png!thumbnail" alt="img" style="zoom:50%; " />`


* `为什么有些情况即使顺序不对，catkin_make也能编译成功？ <https://jbohren-ct.readthedocs.io/en/pre-0.4.0-docs/migration.html>`_

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/0EA9e6jBjsZnVsIF.png!thumbnail" alt="img" style="zoom:67%; " />`

opencv库兼容性问题
^^^^^^^^^^^^^^^^^^


* 不同版本的opencv库或有功能相同但名字不同的问题，在编译时可能会出现未声明等报错，这时候就需要查文档就行修改。

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/Sz3d8VYj2wt2TNqb.png!thumbnail" alt="img" style="zoom:50%; " />`

实例：\ `kalibr 16.04/14.04 <https://github.com/ethz-asl/kalibr>`_ -> `kalibr 20.04 <https://github.com/ori-drs/kalibr>`_


* CheckLists

.. list-table::
   :header-rows: 1

   * - 16.04(apt version)
     - 20.04(apt version 4.2)
   * - CV_LOAD_IMAGE_COLOR (icv::imread)
     - cv:: IMREAD_COLOR



* 一般来说可以尝试先将\ ``CV_``\ 转化为\ ``cv::``\ 来进行替换

boost库的升级换代
^^^^^^^^^^^^^^^^^


* 有关模块


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210918004819514.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210918004819514.png
   :alt: image-20210918004819514



.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210918005720515.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210918005720515.png
   :alt: image-20210918005720515



* 有关函数

.. code-block:: c++

   // for 16.04
   boost::this_thread::sleep(boost::chrono::microseconds(SmallIterval)); 
   // for 20.04
   std::this_thread::sleep_for(std::chrono::microseconds(SmallIterval));

.. note:: 在编译时有些函数不存在，可能是因为更新换代而被取代了，这时候查一下google和相关文档即可


ambigious candidate
^^^^^^^^^^^^^^^^^^^

..

   Reference to 'shared_ptr' is ambiguous candidate found by name lookup is 'boost::shared_ptr' candidate found by name lookup is 'pcl::shared_ptr'


pcl库和boost都有自己的share_ptr实现，而\ `源程序 <https://github.com/fverdoja/Fast-3D-Pointcloud-Segmentation>`_\ 使用了using这种方法，使得编译器不知道该调用哪个share_ptr

.. code-block:: c++

   using namespace boost;
   using namespace pcl;

   void removeText(shared_ptr<visualization::PCLVisualizer> viewer); // ERROR
   void removeText(pcl::shared_ptr<visualization::PCLVisualizer> viewer); // TRUE

Tools
-----

`catkin-lint <https://fkie.github.io/catkin_lint/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

静态查看catkin工程错误

.. prompt:: bash $,# auto

   # 安装
   $ sudo apt install catkin-lint
   # example
   $ catkin_lint -W0 .


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210912200754563.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210912200754563.png
   :alt: image-20210912200754563


.. note:: catkin_lint相关提示信息仅供参考，不一定准确


`ccmake <https://cmake.org/cmake/help/latest/manual/ccmake.1.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

cmake TUI程序，在\ **终端**\ 交互式地配置选项

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210925215521631.png" alt="image-20210925215521631" style="zoom:67%; " />`

cmake-gui
^^^^^^^^^

cmake GUI程序，在\ **图形化界面**\ 交互式地配置选项

Reference
---------


* `colon的诞生背景 <https://design.ros2.org/articles/build_tool.html>`_
