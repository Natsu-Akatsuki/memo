
Beginner
========

Grammar
-------


* 
  指令大小写无关（如 ``add_library`` 等价于 ``ADD_LIBRARY``\ ）

* 
  参数和变量、OPTION大小写敏感（如REQUIRED不能写成required）

Alias target
^^^^^^^^^^^^


* 
  ``target``\ ，根据上下文，应该指的是\ ``library``\ 这种target，而不是executable file；且是 ``alias target`` ；采用 ``taget-based`` 的方法可以不用再\ ``include_directory`` ，只需要 ``target_link_libraries`` 就能完成编译（\ `link <https://github.com/ttroy50/cmake-examples/tree/master/01-basic/H-third-party-library>`_\ ）

* 
  案例1：


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/wbtoJSQAxXyl23X8.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/wbtoJSQAxXyl23X8.png!thumbnail
   :alt: img


等价于


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/X74TytKWlvFw0Xst.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/X74TytKWlvFw0Xst.png!thumbnail
   :alt: img



* `案例2 <https://github.com/fzi-forschungszentrum-informatik/Lanelet2/issues/39>`_


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/srnzrPDtnm75OZuv.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/srnzrPDtnm75OZuv.png!thumbnail
   :alt: img



* target-based的target（library）采用的是alias targets，其生成的方式可参考 `cmake-examples <https://github.com/ttroy50/cmake-examples/blob/master/01-basic/D-shared-library/README.adoc>`_\ , `Fexui example <https://github.com/ArthurSonzogni/FTXUI/blob/master/cmake/ftxui_export.cmake>`_


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/uK5A6MiUUP6Ylf96.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/uK5A6MiUUP6Ylf96.png!thumbnail
   :alt: img


Macro
-----

`add_dependencies <https://cmake.org/cmake/help/latest/command/add_dependencies.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 指明依赖：即指明在target生成前，需要先生成某些\ ``target``

.. code-block:: cmake

   add_dependencies(<target> [<target-dependency>]...)

add_definitions
^^^^^^^^^^^^^^^


* 添加宏

.. code-block:: cmake

   add_definitions(-DPERFORMANCE_LOG)

add_executable
^^^^^^^^^^^^^^


* 构建可执行文件

.. code-block:: cmake

   add_executable(target_name 文件名)

add_library
^^^^^^^^^^^


* 构建库文件，构建的 ``library`` 可以不写全名字，如只写hello，cmake会自动补全为 ``libhello.so`` 或 ``libhello.a``

.. code-block:: cmake

   add_library(target_name STATIC 文件名)     # 静态库
   add_library(target_name SHARED 文件名)     # 动态库
   add_library(target_name OBJECT 文件名)     # 生成目标文件但是不进行链接


* `复用目标文件，防止多次编译 <https://www.anycodings.com/1questions/1992095/cmake-reuse-object-files-built-for-a-lib-into-another-lib-target>`_\ ：\ `官方资料 <https://gitlab.kitware.com/cmake/community/-/wikis/doc/tutorials/Object-Library>`_

cuda
^^^^

.. code-block:: cmake

   cuda_add_executable(target_name <...cu>)
   cuda_add_library(target_name <library_name>)

`compile_options <https://cmake.org/cmake/help/latest/command/target_compile_options.html?highlight=target_compile_options>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 编译项

.. code-block:: cmake

   # 设置相关标准
   set(CMAKE_CXX_STANDARD 17) # 具有最强的覆盖作用
   add_compile_options(-std=c++14)

   # 设置DEBUG时的编译选项
   SET(CMAKE_BUILD_TYPE "RELEASE")
   SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O0 -g")
   # 设置DEBUG时的编译选项
   SET(CMAKE_BUILD_TYPE "DEBUG")
   SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O3 -Wall")

   # 设置GDB编译项
   add_compile_options("-O0" "-g") # for gdb（注意一项对应一个编译项）
   target_compile_options(<target_name> PUBLIC "-O0" "-g")

   # 保留中间产物
   target_compile_options(<target_name> PUBLIC "-save-temps")

   # 屏蔽deprecated消息
   set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations")

   # -Wno-deprecated
   # -march=native：使用本机的编译指令（代码运行速度或会提高）

   # 设置优化项
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

.. note:: 该选项会覆盖CMAKE_BUILD_TYPE


.. note::  ``add_compile_options()`` 作用于所有编译器， ``CMAKE_CXX_FLAGS`` 或 ``CMAKE_C_FLAGS`` 分别只针对c++，c编译器


`optimization <https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#Optimize-Options>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. prompt:: bash $,# auto

   # -O0：(default) 屏蔽所有的优化
   # -0g：suppresses many optimization passes
   # -O3：优化等级为3

   # CMAKE_BUILD_TYPE:
   # -O3：Release
   # -O0：for Debug
   # -Os：for MinRelSize

`warning <https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html#Warning-Options>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

   # -Wconversion: e.g double -> float (narrowing conversion)

configure_file
^^^^^^^^^^^^^^


* 拷贝一个文件，并用cmake文件的变量替换输入文件中形如\ ``@VAR@``\ 或\ ``${VAR}``\ 的变量
* 让普通文件使用CMake的变量

.. code-block:: cmake

   configure_file(
     ${PROJECT_SOURCE_DIR}/header.hpp.in
     ${PROJECT_SOURCE_DIR}/include/global_definition/header.hpp)


* 用例可参考任老的仓库（\ `detail <https://github.com/Little-Potato-1990/localization_in_auto_driving/blob/master/lidar_localization/cmake/global_defination.cmake>`_\ ）

`execute_process <https://blog.csdn.net/qq_28584889/article/details/97758450>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 执行命令行

.. code-block:: cmake

   # 相关待执行的命令； 存储标准输出的变量
   execute_process(COMMAND python -c "from sysconfig import get_paths;print(get_paths()['include'])" OUTPUT_VARIABLE DUMMY)
   execute_process(COMMAND python3 -c "import torch; print(f'{torch.utils.cmake_prefix_path}/Torch', end='')" OUTPUT_VARIABLE Torch_DIR)

file
^^^^


* 使用通配符找文件

.. code-block:: cmake

   # e.g. file(GLOB source_files ${TENSORRT_INSTALL_DIR}/samples/common/*.cpp)
   file(GLOB <outPUT-var> [<globbing-expr>...])

`find_library <https://cmake.org/cmake/help/latest/command/find_library.html?highlight=find_library#find-library>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cmake

   # find_library (<VAR> name1 [path1 path2 ...])
   find_library(NVPARSERS NAMES nvparsers)
   find_library(NVCAFFE_PARSER NAMES nvcaffe_parser)
   find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)


* 要添加搜索路径，可修改\ ``CMAKE_LIBRARY_PATH``\ 或\ ``CMAKE_PREFIX_PATH``

.. code-block:: cmake

   # e.g.
   set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "$ENV{HOME}/application/TensorRT-8.0.0.3/lib")

`find_package <https://cmake.org/cmake/help/v3.18/command/find_package.html?highlight=find_package>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: plain

   find_package(<PackageName> [version] [EXACT] [QUIET] [MODULE]
                [REQUIRED] [[COMPONENTS] [components...]]
                [OPTIONAL_COMPONENTS components...]
                [NAMES name1 [name2 ...]]
                # If the NAMES option is given the names following it are used instead of <PackageName>
                [NO_POLICY_SCOPE])


* 指定路径

.. code-block:: cmake

   find_package(PCL REQUIRED
   PATHS  库路径
   NO_DEFAULT_PATH)  # 只在PATHS路径下寻找，不使用默认的搜索路径


* 
  ``find_package``\ 宏执行后会产生相关的变量，例如，\ ``<package_name>_INCLUDE_DIRS``
  or ``<package_name>_INCLUDES`` or
  ``<package_name>_INCLUDE_DIR`` 具体看相关模块的设计

* 
  cmake modules 指文件\ ``FindXXX.cmake``\ ，要指定 cmake
  module的搜索路径，可以配置如下参数；不过它也有默认的搜索路径即cmake安装路径下的Module目录（e.g.
  /usr/share/cmake-3.16/Modules），在默认路径下没找到，才去CMAKE_MODULE_PATH下找

.. code-block:: cmake

   set(CMAKE_MODULE_PATH 路径名)
   # set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/share/cmake/geographiclib/")


* ``find_packaege``\ 还有一种\ ``Config Mode``\ ，当没找到\ ``FindXXX.cmake``\ 时将按特定的规则进行搜寻，具体可参考\ `英文文档 <https://cmake.org/cmake/help/latest/command/find_package.html#search-procedure>`_\ 和\ `中文说明 <https://zhuanlan.zhihu.com/p/50829542>`_\ （PATH环境变量也会起作用），
* 该种模式下找的是\ ``... LibConfig.cmake``\ 或\ ``...lib_config.cmake``\ 。可添加的搜索路径为

.. code-block:: plain

   <package>_DIR
   CMAKE_PREFIX_PATH
   CMAKE_FRAMEWORK_PATH
   CMAKE_APPBUNDLE_PATH
   PATH


* ``find_package``\ 中如果find的包是\ ``catkin``\ ，则\ ``components``\ 用于将\ ``components``\ 涉及的包的环境变量都统一到\ ``catkin_ prefix``\ 的环境变量中。\ `用与节省敲代码的时间(typing time) <http://wiki.ros.org/catkin/CMakeLists.txt#Why_Are_Catkin_Packages_Specified_as_Components.3F>`_

function
^^^^^^^^


* 自定义函数

.. code-block:: cmake

   # abstract from https://github.com/tier4/AutowareArchitectureProposal.iv/blob/use-autoware-auto-msgs/perception/object_recognition/detection/lidar_centerpoint/CMakeLists.txt

   function(download FILE_NAME GFILE_ID FILE_HASH)
   # https://drive.google.com/file/d/GFILE_ID/view
   message(STATUS "Checking and downloading ${FILE_NAME}")
   set(FILE_PATH ${DATA_PATH}/${FILE_NAME})
   if(EXISTS ${FILE_PATH})
       file(MD5 ${FILE_PATH} EXISTING_FILE_HASH)
       if(NOT ${FILE_HASH} EQUAL ${EXISTING_FILE_HASH})
       message(STATUS "... file hash changes. Downloading now ...")
       execute_process(COMMAND gdown --quiet https://drive.google.com//uc?id=${GFILE_ID} -O ${FILE_PATH})
       endif()
   else()
       message(STATUS "... file doesn't exists. Downloading now ...")
       execute_process(COMMAND gdown --quiet https://drive.google.com//uc?id=${GFILE_ID} -O ${FILE_PATH})
   endif()
   endfunction()

   # default model
   download(pts_voxel_encoder_default.onnx 1_8OCQmrPm_R4ZVh70QsS9HZo6uGrlbgz 01b860612e497591c4375d90dff61ef7)

include_directories
^^^^^^^^^^^^^^^^^^^


* 添加链接库文件搜索路径（文件夹）

方法一：

.. code-block:: cmake

   # 当前包的头文件目录要放在前面
   include_directories(
    include  # 相对于当前CMakeLists所在的文件目录
    ${catkin_INCLUDE_DIRS}
   )


* 控制追加的路径是放在原来的前面还是后面（设置参数 ON）

.. code-block:: cmake

   set(cmake_include_directorirs_before ON)
   set(cmake_include_directorirs_after ON)

.. hint::  ``#include <file.h>`` 时对应的位置是相对于 ``include_directories`` 中导入的路径例如： ``include_directories`` 的路径是/include/；头文件在/include/package_name/header.h则最终的编写应为 ``#include <package_name/header.h>``


方法二：

.. code-block:: cmake

   target_include_directories（target_name
       PUBLIC
           头文件目录）

方法二的头文件路径仅适用特定的 ``target`` ，方法一的适用于所有 ``target``

`link_directories <https://cmake.org/cmake/help/latest/command/link_directories.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 添加链接库文件搜索路径（文件夹），不起链接作用

.. code-block:: cmake

   link_directories(dir_path)

list
^^^^


* 正则移除\ ``catkin_LIBRARIES``\ 中的系统pcl库

.. code-block:: cmake

   # remove pcl installed from apt
   list(FILTER catkin_LIBRARIES EXCLUDE REGEX /usr/lib/x86_64-linux-gnu/libpcl*)
   list(FILTER catkin_LIBRARIES EXCLUDE REGEX optimized)
   list(FILTER catkin_LIBRARIES EXCLUDE REGEX debug)

   list(FILTER catkin_INCLUDE_DIRS EXCLUDE hREGEX /usr/include/pcl-1.8)

`message <https://cmake.org/cmake/help/latest/command/message.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cmake

   message(STATUS|WARNING|FATAL|SEND_ERROR ${})# 这种形式一定要加STATUS这些option
   message("...")

   # 显示列表数据时带分隔符;
   message("${...}")
   # 替换分隔符
   string(REPLACE ";"  ", " new_str "${old_str}")


* `彩色输出 <https://stackoverflow.com/questions/18968979/how-to-get-colorized-output-with-cmake>`_

`properties <https://cmake.org/cmake/help/v3.18/manual/cmake-properties.7.html#target-properties>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 修改属性

.. code-block:: cmake

   set_target_properties(target1 target2 ...
                         PROPERTIES prop1 value1
                         prop2 value2 ...)
   set_property(<GLOBAL                      |
                 DIRECTORY [<dir>]           |
                 TARGET    [<target1> ...]   |
                 SOURCE    [<src1> ...]
                           [DIRECTORY <dirs> ...]
                           [TARGET_DIRECTORY <targets> ...] |
                 INSTALL   [<file1> ...]     |
                 TEST      [<test1> ...]     |
                 CACHE     [<entry1> ...]    >
                [APPEND] [APPEND_STRING]
                PROPERTY <name> [<value1> ...])


* 修改文件生成名前/后缀等

.. code-block:: cmake

   set_target_properties(lib_cpp PROPERTIES PREFIX "")               #  指定前缀
   set_target_properties(lib_cpp PROPERTIES OUTPUT_NAME "lib_cpp")   #  指定文件名
   set_target_properties(lib_cpp PROPERTIES SUFFIX ".so")            #  指定后缀
   set_target_properties(lib_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})  # 指定库的输出路径
   set_target_properties(lib_cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})  # 指定可执行文件的输出路径

原来默认生成 ``lib_cpp.cpython-37m-x86_64-linux-gnu.so`` 现在是 ``lib_cpp.so`` ；更多属性配置可参考\ `link <https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html#target-properties>`_

target_link_libraries
^^^^^^^^^^^^^^^^^^^^^


* 链接库

.. code-block:: cmake

   target_link_libraries(target_name library_name)


* `有关关键词option： private、public、target的区别 <%5Bhttps://leimao.github.io/blog/CMake-Public-Private-Interface/%5D(https://leimao.github.io/blog/CMake-Public-Private-Interface/>`_\ )：

本质是用于描述一个链接是否能被继承


* ``private``\ (default)：目标文件A所链接过的库不会被目标文件B 继承
* ``public``\ ：目标文件A所链接过的库可被目标文件B 继承
* ``interface``\ ：目标文件A所链接过的库不可被目标文件B继承，但是目标文件C链接B时可链接到目标文件A的链接库

有如下案例：比如给定三个文件，分别为可执行文件A ``eat_apple`` ；库A ``fruit`` (有size和color两个函数)；库B ``apple_libraries`` (有apple_size这个函数，该实现依赖 ``fruit库`` 调用了size函数) 。如果，在实现库B到库A的link时，采用private关键词；eat_apple中调用了apple_size这个函数，依赖了apple_libraries库。此时可执行文件A到库B的link无论使用哪种关键词，都会有link错误。因为前面采用了private关键词，库A到库B的link不会被可执行文件A继承。需要将库B到库A的privatelink改为public link才行。


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/GVwiCAlL2biYLEkP.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/GVwiCAlL2biYLEkP.png!thumbnail
   :alt: img


`option <https://cmake.org/cmake/help/v3.20/command/option.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cmake

   option(<variable> "<help_text>" [value])


* 拓展：\ `option和set的区别？ <https://stackoverflow.com/questions/36358217/what-is-the-difference-between-option-and-set-cache-bool-for-a-cmake-variabl>`_\ ，option只能布尔型，默认是OFF；某些场景下可以相互替换

set
^^^

设置变量

.. code-block:: cmake

   set(SOURCES
       src/Hello.cpp
       src/main.cpp
   )
   message(${SOURCES})   # src/Hello.cppsrc/main.cpp
   set(ENV{变量名} 值)    # 获取环境变量（注意ENV需要大写）
   message($ENV{HOME})   # 使用环境变量

.. hint:: 单个variable有多个arguments时，用分号将argument进行concatenate后再进行赋值；然而message显示时，不会出现分号；使用一个变量时，不同于 bash可以不加上{}，在 CMakelists中一定要加上


include
^^^^^^^


* 
  导入额外的cmake文件

* 
  方法一：

.. code-block:: cmake

   include(<file|module> [OPTIONAL] [RESULT_VARIABLE <var>]
                         [NO_POLICY_SCOPE])

从某个\ **文件**\ (CMakeLists.txt)或模块(.cmake)中导入cmake代码；未指定地址时，首先在内置的模块库目录下寻找( ``CMake builtin module directory`` )，其次在\ **CMAKE_MODULE_PATH**\ 中寻找

.. code-block:: cmake

   set(VTK_CMAKE_DIR "${VTK_SOURCE_DIR}/CMake")
   set(CMAKE_MODULE_PATH ${VTK_CMAKE_DIR} ${CMAKE_MODULE_PATH})
   include(vtkCompilerChecks)  # /VTK-8.2.0/CMake/vtkCompilerChecks.cmake


* 方法二：导入CMakeLists.txt，source_dir对应CMakeLists.txt的所在\ **目录**

.. code-block:: cmake

   add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])


* ``include``\ 和\ ``add_subdirectory``\ 的区别？(\ `details <https://stackoverflow.com/questions/48509911/cmake-add-subdirectory-vs-include>`_\ )

add_subdirectory会有不同的变量作用域；

外面的编译选项会传递到add_subdirectory中（子工程可以覆盖它）

install
^^^^^^^


* 可以安装的内容：编译产生的target文件（即可执行文件、库文件）；其他文件
* 若要指定安装路径：

方法一：命令行

.. prompt:: bash $,# auto

   cmake .. -DCMAKE_INSTALL_PREFIX=install

方法二：cmake-gui等图形界面进行：


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/fCeDn3uR7Aeffvas.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/fCeDn3uR7Aeffvas.png!thumbnail
   :alt: img



* 指定安装的内容和相对路径：

.. code-block:: cmake

   # 安装可执行文件，并安装到到指定目录：${CMAKE_INSTALL_PREFIX}/bin
   install (TARGETS <target_name>
       DESTINATION bin)

   # 安装库文件，并安装到指定目录：${CMAKE_INSTALL_PREFIX}/lib
   install (TARGETS <target_name>
       LIBRARY DESTINATION lib)

   # 安装库文件（挪整个文件夹），并安装到指定目录：${CMAKE_INSTALL_PREFIX}/include
   install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
       DESTINATION include)

   # 安装配置文件，拷贝到：${CMAKE_INSTALL_PREFIX}/etc
   install (FILES <file_name>
       DESTINATION etc)

   # ROS2
   install(
     DIRECTORY include/
     DESTINATION include
   )

   # ROS_PKG/lib
   # ROS_PKG/bin
   # ROS_PKG/include
   install(
     TARGETS my_library
     EXPORT my_libraryTargets
     LIBRARY DESTINATION lib
     ARCHIVE DESTINATION lib
     RUNTIME DESTINATION bin
     INCLUDES DESTINATION include
   )


* ``make install``\ 后 CMake 会生成\ ``install_manifest.txt``\ 文件（含安装的文件路径，到时可基于这个文件删除安装文件）

.. code-block:: cmake

   e.g.
   /usr/local/include/ceres/autodiff_cost_function.h
   /usr/local/include/ceres/autodiff_first_order_function.h
   /usr/local/include/ceres/autodiff_local_parameterization.h

.. hint:: 默认安装路径： ``/usr/local/include`` ; ``/usr/local/bin`` ; ``/usr/local/lib/cmake``


`ament <https://docs.ros.org/en/foxy/How-To-Guides/Ament-CMake-Documentation.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* ``ament_target_dependencies`` 比 ``target_link_libraries`` 更具优势，It will also ensure that the include directories of all dependencies are ordered correctly when using overlay workspaces

.. code-block:: cmake

   find_package(ament_cmake REQUIRED)
   find_package(rclcpp REQUIRED)
   # 链接库
   ament_target_dependencies()
   # 导出库
   ament_export_targets()
   ament_export_dependencies()
   # The project setup is done by ament_package() and this call must occur exactly once per package. ament_package() installs the package.xml, registers the package with the ament index, and installs config (and possibly target) files for CMake so that it can be found by other packages using find_package
   ament_package()

`ament auto <https://zhuanlan.zhihu.com/p/438191834>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ament的高级封装

.. code-block:: cmake

   find_package(ament_cmake_auto REQUIRED)
   # automatically link the dependency according to the xml (without find_package)
   ament_auto_find_build_dependencies()

   # 生成目标文件
   ament_auto_add_library(listener_node SHARED src/listener_node.cpp)
   ament_auto_add_executable(listener_node_exe src/listener_main.cpp)

   # replace the export, install and ament_package command
   ament_auto_package()

catkin_package
^^^^^^^^^^^^^^


* 
  `官方文档 wiki <http://wiki.ros.org/catkin/CMakeLists.txt#catkin_package.28.29>`_\ 、\ `官方文档 api <https://docs.ros.org/en/groovy/api/catkin/html/dev_guide/generated_cmake_api.html#catkin_package>`_

* 
  作用：安装\ ``package.xml``\ ；生成可被其他package调用的配置文件(即.config或.cmake文件)。使其他包\ ``find_package``\ 时可以获取这个包的相关信息，如依赖的头文件、库、CMake变量

.. code-block:: cmake

   catkin_package(
     INCLUDE_DIRS include
     CATKIN_DEPENDS cloud_msgs
     DEPENDS PCL
   )
   add_executable(imageProjection src/imageProjection.cpp)
   add_executable(featureAssociation src/featureAssociation.cpp)
   add_executable(mapOptmization src/mapOptmization.cpp)
   add_executable(transformFusion src/transformFusion.cpp)


* 实测其并不会将当前的include等文件夹拷贝到devel目录中
* 必须要在声明targets前（即使用\ ``add_library()``\ 或\ ``add_executable()``\ 前）调用该宏

Module CheatSheet
-----------------

EIGEN
^^^^^

.. code-block:: cmake

   find_package(Eigen3 REQUIRED)
   include_directories(${EIGEN3_INCLUDE_DIRS})

OpenCV
^^^^^^

.. code-block:: cmake

   find_package(OpenCV REQUIRED)
   include_directories(${OpenCV_INCLUDE_DIRS})
   target_link_libraries(<target> ${OpenCV_LIBS})

Variables CheatSheet
--------------------

python
^^^^^^

`FindPythonLibs <https://cmake.org/cmake/help/v3.10/module/FindPythonLibs.html>`_ / `FindPythonInterp <https://cmake.org/cmake/help/v3.10/module/FindPythonInterp.html?highlight=python_executable>`_

.. prompt:: bash $,# auto

   -DPYTHON_EXECUTABLE=/opt/conda/bin/python3
   -DPYTHON_EXECUTABLE=$(python -c "import sys;print(sys.executable)")

   -DPYTHON_INCLUDE_DIR=$(python -c "from sysconfig import get_paths;print(get_paths()['include'])")
   -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.8.so

   -DPYBIND11_PYTHON_VERSION=3.7
   -DPYTHON_VERSION=3.7

compiler
^^^^^^^^

.. prompt:: bash $,# auto

   # 指定使用c++14标准
   set(CMAKE_CXX_FLAGS "-std=c++14")

`ros <http://docs.ros.org/en/kinetic/api/catkin/html/user_guide/variables.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

path
^^^^

.. list-table::
   :header-rows: 1

   * - Variable
     - Info
   * - CMAKE_SOURCE_DIR
     - The root source directory
   * - CMAKE_CURRENT_SOURCE_DIR
     - The current source directory **if using sub-projects and directories**.
   * - PROJECT_SOURCE_DIR
     - 当前CMake工程的源文件路径（.cmake文件所在路径）
   * - PROJECT_BINARY_DIR
     - 当前工程的build目录
   * - CMAKE_BINARY_DIR
     - 执行cmake命令的所在目录
   * - CMAKE_CURRENT_BINARY_DIR
     - The build directory you are currently in.
   * - `LIBRARY_OUTPUT_PATH <https://cmake.org/cmake/help/v3.18/variable/LIBRARY_OUTPUT_PATH.html?highlight=library_output_path>`_ （deprecated）LIBRARY_OUTPUT_DIRECTORY
     - 库的输出路径（要设置在add_library之前）
   * - CMAKE_PREFIX_PATH
     - find_packaeg 搜索.cmake .config的搜索路径（初始为空）
   * - EXECUTABLE_OUTPUT_PATH
     - 可执行文件的输出路径


Reference
---------


* `github例程 <https://github.com/ttroy50/cmake-examples>`_
* `定义和术语 <https://cmake.org/cmake/help/latest/manual/cmake-language.7.html>`_
* `官网 <https://cmake.org/cmake/help/latest/index.html>`_
