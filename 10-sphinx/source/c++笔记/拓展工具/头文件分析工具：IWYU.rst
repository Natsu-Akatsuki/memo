.. role:: raw-html-m2r(raw)
   :format: html


头文件分析工具：\ `IWYU <https://github.com/include-what-you-use/include-what-you-use>`_
==========================================================================================

01. what？
----------

可以用于分析一个源程序中的头文件是否冗余的工具

02. how to use?(base case)
--------------------------


1. `安装相关的llvm依赖 <https://apt.llvm.org/>`_\ （以版本号为12.0为例）

.. code-block:: bash

   # 使用脚本的方式进行安装，安装最新的稳定版本
   $ sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
   $ sudo apt install llvm-12-dev libclang-12-dev clang-12


2. 编译和安装\ ``IWYU``\ （以clang的版本号为12.0为例）

.. code-block:: bash

   $ git clone https://github.com/include-what-you-use/include-what-you-use.git
   $ cd include-what-you-use
   $ git checkout clang_12.0
   $ mkdir build && cd build
   $ cmake -G "Unix Makefiles" -DCMAKE_PREFIX_PATH=/usr/lib/llvm-12 ..
   $ make
   # 默认安装在/usr/local/bin/include-what-you-use
   $ sudo make install


3. 基础用例


* 构建\ ``main.cpp``\ ：

.. code-block:: c++

   #include <iostream>
   #include <fcntl.h>
   #include <signal.h>
   #include <stdio.h>
   #include <stdlib.h>
   #include <string.h>
   #include <stdint.h>
   #include <sys/mman.h>
   #include <sys/stat.h>
   #include <unistd.h>
   #include <sys/user.h>
   #include <execinfo.h>

   int main() {
       std::cout << "Hello, World!" << std::endl;
       return 0;
   }


* 构建\ ``CmakeLists``\ ：

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.13)
   project(iwyu_base_case)

   set(CMAKE_CXX_STANDARD 20)

   add_executable(iwyu_base_case main.cpp)
   # 主要是添加下面这一句
   set_property(TARGET iwyu_base_case PROPERTY CXX_INCLUDE_WHAT_YOU_USE "/usr/local/bin/include-what-you-use")


* 执行make时则会提供如下信息

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210729223215252.png" alt="image-20210729223215252" style="zoom:67%;" />`


* (optional) 参考其修改意见以修改相关文件
.. code-block:: bash

   $ make -k CXX=/usr/local/bin/include-what-you-use 2> iwyu.out
   $ python /usr/local/bin/fix_includes.py < iwyu.out
