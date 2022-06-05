.. role:: raw-html-m2r(raw)
   :format: html


Standard
========

C++
---

命名规范
^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 范例
     - 应用场景
     - 描述
   * - **camelCased**
     - functionName\ :raw-html-m2r:`<br />`\ methodName
     - 函数名和方法代表执行某些行为，所以命名一般是动态的
   * - **CamelCased**
     - ClassName\ :raw-html-m2r:`<br />`\ IMUName
     - 一般是名词
   * - **__XXXX**
     - 系统保留
     - ``__builtin_expect`` (一般开发者不需要修改这方面内容)
   * - **ALL_CAPITALS**
     - CONSTANT
     - ``PI``
   * - **ALL_CAPITALS**
     - 宏名
     - CHECK_CUDA_ERROR
   * - **under_scored**
     - namespace_name\ :raw-html-m2r:`<br />`\ package_name\ :raw-html-m2r:`<br />`\ topic_name\ :raw-html-m2r:`<br />`\ service_name
     - 
   * - **under\ *scored*\ **
     - member_varibale
     - 
   * - **g_under_scored**
     - g_global_variable


----

**NOTE**


* Compound names of over three words are a clue that your design may be unnecessarily confusing.

----

命名空间规范
^^^^^^^^^^^^


* 不建议在头文件使用\ ``using-directive``\ ：会干扰导入这个头文件的目标文件的命名空间
* 可以在源文件中使用\ ``using-directives`` ，但更推荐使用\ ``using-declarations``\ ，只导入自己想要的变量

.. code-block:: cpp

   using namespace std;  // Bad, because it imports all names from std::
   using std::list;  // I want to refer to std::list as list
   using std::vector;  // I want to refer to std::vector as vector

函数名
^^^^^^

.. code-block:: cpp

   checkForErrors() // good
   errorCheck() // bad
   dumpDataToFile() // good
   dataFile() // bad

宏
^^

#if vs. #ifdef
^^^^^^^^^^^^^^

使用前者，因为可以如此使用来屏蔽宏


* CLI

.. prompt:: bash $,# auto

   $ cc -c lurker.cpp -DDEBUG=0


* 源程序

.. code-block:: cpp

   cudaEvent_t start_, stop_;
   cudaStream_t stream_ = 0;
   CHECK_CUDA_ERROR(cudaEventCreate(&start_));
   CHECK_CUDA_ERROR(cudaEventCreate(&stop_));

   CHECK_CUDA_ERROR(cudaEventDestroy(start_));
   CHECK_CUDA_ERROR(cudaEventDestroy(stop_));

   #if PERFORMANCE_LOG
     float generateFeaturesTime = 0.0f;
     CHECK_CUDA_ERROR(cudaEventRecord(start_, stream_));
   #endif

   #if PERFORMANCE_LOG
     CHECK_CUDA_ERROR(cudaEventRecord(stop_, stream_));
     CHECK_CUDA_ERROR(cudaEventSynchronize(stop_));
     CHECK_CUDA_ERROR(cudaEventElapsedTime(&generateFeaturesTime, start_, stop_));
     std::cout<<"TIME: generateVoxels: "<< generateVoxelsTime <<" ms." <<std::endl;
     std::cout<<"TIME: generateFeatures: "<< generateFeaturesTime <<" ms." <<std::endl;
     std::cout<<"TIME: doinfer: "<< doinferTime <<" ms." <<std::endl;
     std::cout<<"TIME: doPostprocessCuda: "<< doPostprocessCudaTime <<" ms." <<std::endl;
   #endif

拓展资料
^^^^^^^^


* `Google C++ style guide <https://google.github.io/styleguide/cppguide.html>`_
* `ros2 code style <https://docs.ros.org/en/foxy/Contributing/Code-Style-Language-Versions.html>`_
* `ros1 cpp code style <http://wiki.ros.org/CppStyleGuide>`_

Python
------


* ``package_name``
* ``ClassName``
* ``method_name``
* ``field_name``
* ``_private_something``
* ``self.__really_private_field``
* ``_global``
* **4** space indentation

ROS
---

Package
^^^^^^^


* 全部由小写字母、数字、_构成，开头为字母
* 不使用连续的\ ``_``\ ，即\ ``__``
* 
  至少两个字符长

* 
  更多细节参考 `REP-144 <https://www.ros.org/reps/rep-0144.html>`_

* 
  所有python源代码都放置在ros包目录下

* 要被调用的模块/包放置在/src目录下
* python脚本名等于节点名
* 脚本顶部有\ ``#!/usr/bin/env python``
* 目录结构的参考案例

.. code-block::

   global_planner # ros package
    |- src/ # 需要重用的代码
       |- global_planner # python package
         |- __init__.py
         |- file.py
    |- scripts/          
       |- ros相关的脚本文件（不被其他包调用）
    |- launch/      # 存放launch文件
    |- scripts/     # 含ros的相关不需要重用的代码   
    |- msg/      # 存放msg文件  
    |- srv/       # 存放srv文件
    |- urdf/       # 存放urdf / xacro等模型文件
    CMakesLists.txt
    package.xml
    setup.py

Topic
^^^^^


* 
  更多细节参考 `wiki <http://wiki.ros.org/Names>`_

* 
  所有topic应放置在\ ``private namespace``\ ；\ ``global topic``\ 需要有文档说明

* 私人命名空间的命名建议：方便别人在launch文档中进行重映射

  * ``input``\ : 该主题是要订阅的主题
  * ``output``\ : 该主题是要发布的主题
  * ``debug``\ : 该发布的主题是用来debug的 (e.g. for visulization)

举例：有一个节点订阅点云并对它进行下采样，那它的主题应设计为：


* ~input/points_original
* ~output/points_filtered
