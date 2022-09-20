.. role:: raw-html-m2r(raw)
   :format: html


Tools
=====

Ccache
------

CLI
^^^

.. prompt:: bash $,# auto

   # 安装(ubuntu 20.04 version 3.7.7)
   $ sudo apt install ccache
   # 指定最大缓存量
   $ ccache -M 1G
   # 清除缓存
   $ ccache -C

`Install from Source <https://github.com/ccache/ccache/blob/master/doc/INSTALL.md>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ apt install libhiredis-dev asciidoctor

   $ wget -c https://github.com/ccache/ccache/releases/download/v4.6/ccache-4.6.tar.gz 
   # 解压缩和路径跳转
   $ mkdir build
   $ cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release ..

CMake
^^^^^

.. code-block:: cmake

   find_program(CCACHE_FOUND ccache)
    if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
   endif(CCACHE_FOUND)

GDB
---

LLDB的文档不太全，通用性较低，暂时还是倾向于使用GDB

Configuration
^^^^^^^^^^^^^


* `自定义配置文件：GEF <https://gef.readthedocs.io/en/master/>`_\ ；\ `GDB 配置文件 <https://github.com/cyrus-and/gdb-dashboard>`_
* `保存历史命令 <https://github.com/hellogcc/100-gdb-tips/blob/master/src/save-history-commands.md>`_\ ，默认的历史命令行导出路径为\ ``~/.gdb_history``

.. prompt:: bash $,# auto

   $ echo "set history save on" >> ~/.gdbinit


* `不显示线程启/闭信息 <https://stackoverflow.com/questions/10937289/how-can-i-disable-new-thread-thread-exited-messages-in-gdb>`_

.. prompt:: bash $,# auto

   $ echo "set print thread-events off" >> ~/.gdbinit


* `修改prompt <https://sourceware.org/gdb/onlinedocs/gdb/Prompt.html>`_

.. prompt:: bash $,# auto

   $ echo "set extended-prompt \w (gdb) " >> ~/.gdbinit


* 隐藏启动时的提示信息和版权信息，\ `details <https://stackoverflow.com/questions/63918429/permanently-disable-gdb-startup-text>`_

.. prompt:: bash $,# auto

   # 方案一（CLI）：设置别名
   $ alias gdb="gdb -q"

   # 方案二（配置文档）：注意不是gdbinit (from gdb 11)
   $ echo "set startup-quietly on" >> ~/.gdbearlyinit

CLI
^^^


* `GDB reference card <https://users.ece.utexas.edu/~adnan/gdb-refcard.pdf>`_\ ；\ `GDB cheatsheet <https://darkdust.net/files/GDB%20Cheat%20Sheet.pdf>`_

.. list-table::
   :header-rows: 1

   * - 命令行
     - abbreviation / example
     - 作用
   * - python-interactive [command]
     - pi
     - 进入Python交互模式
   * - python [command]
     - py [command]
     - 执行Python命令行
   * - break [line]
     - break 23
     - 打断点
   * - info vtlb
     - —
     - 查看虚函数表
   * - print :raw-html-m2r:`<variable_name>`
     - —
     - 查看变量
   * - info threads
     - —
     - 查看线程信息
   * - info locals [variable_name]
     - —
     - 查看函数栈的局部变量


`Altering <https://sourceware.org/gdb/onlinedocs/gdb/Altering.html#Altering>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

修改一个变量的值（\ ``CLion``\ 中对应的快捷键为\ ``F2``\ ）

.. code-block:: cpp

   (gdb) set var width=47

Breakpoint
^^^^^^^^^^

.. prompt:: bash $,# auto

   # 给某行打断点
   (gdb) break linenum
   (gdb) break filename:linenum


   (gdb) break filename:function

Disassemble
^^^^^^^^^^^

使用 ``disassemble`` 进一步看出现 ``dump core`` 出现的汇编位置

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/JQptKjWwdwGZWkXZ.png!thumbnail" alt="img" style="zoom:67%; " />`

Frame
^^^^^

Backtrace
~~~~~~~~~

.. prompt:: bash $,# auto

   # 查看调用栈
   (gdb) backtrace
   (gdb) where
   (gdb) info stack

Frame
~~~~~

.. prompt:: bash $,# auto

   # 切换到某一帧
   (gdb) f <num>

   # 查看该帧的局部变量
   (gdb) info locals

   # 查看形参
   (gdb) info args

Library
^^^^^^^

.. prompt:: bash $,# auto

   # 查看链接的动态库
   $ info share


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220810232749699.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220810232749699.png
   :alt: image-20220810232749699


`Pretty Printer <https://sourceware.org/gdb/onlinedocs/gdb/Pretty-Printing.html#Pretty-Printing>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* `启动和关闭的区别 <https://sourceware.org/gdb/onlinedocs/gdb/Pretty_002dPrinter-Example.html#Pretty_002dPrinter-Example>`_
* `gdb command <https://sourceware.org/gdb/onlinedocs/gdb/Pretty_002dPrinter-Commands.html#Pretty_002dPrinter-Commands>`_

.. prompt:: bash $,# auto

   # 查看已有的pretty printer，包括关闭的
   (gdb) info pretty-printer
   Print the list of installed pretty-printers. This includes disabled pretty-printers, which are marked as such.

   # 关闭pretty printer
   (gdb) disable pretty-printer

   # 启动pretty printer
   (gdb) enable pretty-printer

Python
^^^^^^

.. prompt:: bash $,# auto

   $ gdb python 

   (gdb) set args <python文件名>
   (gdb) run (gdb)

`Custom GDB Command <https://sourceware.org/gdb/onlinedocs/gdb/Python-API.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* 一般会使用regex来判断输入变量的类型是否符合需求

.. code-block:: python

   def vec_lookup_function(val):
       lookup_tag = val.type.tag
       if lookup_tag == None:
           return None

       regex = re.compile("^.*vector_base<.*,.*>$")
       if regex.match(lookup_tag):
           return VectorPrinter(val)

       return None

Practice
~~~~~~~~


* `pretty printer for std vector <https://hgad.net/posts/object-inspection-in-gdb/>`_\ ：正则，迭代读数据（dereference）部分很OK

`ROS <http://wiki.ros.org/roslaunch/Tutorials/Roslaunch%20Nodes%20in%20Valgrind%20or%20GDB>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   # 追加tag
   launch-prefix="gdb -ex run --args"

   # option:
   # -ex <command> 执行给定的GDB command

`Signal <https://github.com/hellogcc/100-gdb-tips/blob/master/src/index.md#%E4%BF%A1%E5%8F%B7>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* `查看gdb如何处理信号 <https://github.com/hellogcc/100-gdb-tips/blob/master/src/info-signals.md>`_\ （\ ``Pass to program``\ 即让程序执行完信号回调函数后，程序才暂停）

Practice
^^^^^^^^

诊断\ `rviz段错误 <https://segmentfault.com/a/1190000015238799>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* ROS rviz增加 ``camera`` 或 ``image`` display时，会出现段错误（segmentation fault）

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/fmh1yBcmwUYtSSwt.png!thumbnail" alt="img" style="zoom:67%; " />`

步骤一：执行程序

.. prompt:: bash $,# auto

   $ gdb python
   (gdb) run <py_file>.py

步骤二：添加display触发异常


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/svJsNayoXZXXaa1v.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/svJsNayoXZXXaa1v.png!thumbnail
   :alt: img


步骤三：查看调用栈的情况，可定位到是哪个函数产生段错误（加上full会\ **同时输出局部变量**\ ）

.. prompt:: bash $,# auto

   (gdb) bt full

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/GncFdU91N5TBJGVo.png!thumbnail" alt="img" style="zoom:67%; " />`

`GDB无响应 <https://stackoverflow.com/questions/8978777/why-would-gdb-hang>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

原因未知，可通过下发相关信号解决

.. prompt:: bash $,# auto

   $ kill -CONT <pid of the process>

Extension
^^^^^^^^^

`GDBGUI <https://www.gdbgui.com/gettingstarted/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

暂时没感觉新颖的地方


* Install

.. prompt:: bash $,# auto

   $ pip install gdbgui


* Usage（\ `Youtube <https://www.youtube.com/channel/UCUCOSclB97r9nd54NpXMV5A>`_\ ）

.. prompt:: bash $,# auto

   $ gdbgui

Reference
^^^^^^^^^


* `GDB小技巧 <https://github.com/hellogcc/100-gdb-tips>`_

TODO
^^^^


* 了解\ `Frame Filter <https://chromium.googlesource.com/native_client/nacl-gdb/+/refs/heads/upstream/gdb/python/lib/gdb/command/frame_filters.py>`_\ ，并看未来如何用得上

ClangBuildAnalyzer
------------------

Install
^^^^^^^

.. prompt:: bash $,# auto

   # 安装ClangBuildAnalyzer
   $ git clone https://github.com/aras-p/ClangBuildAnalyzer.git
   $ cd ClangBuildAnalyzer
   $ make -f projects/make/Makefile
   $ cd build
   $ sudo cp ClangBuildAnalyzer /usr/local/bin/

   # 安装 clang
   $ sudo apt install clang-12

Usage
^^^^^


* cmake导入相关参数

.. code-block:: cmake

   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ftime-trace")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-trace")


* `分析结果 <https://github.com/aras-p/ClangBuildAnalyzer#usage>`_

.. prompt:: bash $,# auto

   $ ClangBuildAnalyzer --all <artifacts_folder> <capture_file>

LLDB
----

CLI
^^^

.. list-table::
   :header-rows: 1

   * - 命令行
     - abbreviation / example
     - 作用
   * - script
     - script import sys
     - 调用内置的Python解析器并执行（若无参数则进入交互模式）
   * - command
     - —
     - Commands for managing custom LLDB commands
   * - command script
     - —
     - —
   * - process attach
     - process attach --pid 123
     - 调试一个正在运行的进程


Python
^^^^^^


* `LLDB 22.04导入内置Python出问题 <https://bugs.launchpad.net/ubuntu/+source/llvm-defaults/+bug/1972855>`_

.. prompt:: bash $,# auto

   # 查看内置Python解析器的位置
   $ lldb -P
   $ sudo mkdir -p /usr/lib/local/lib/python3.10/
   $ ln -s /usr/lib/llvm-14/lib/python3.10/dist-packages/ /usr/lib/local/lib/python3.10/dist-packages


* 自定义lldb命令（基于Python）

.. prompt:: bash $,# auto

   # 导入外置python模块
   (lldb) command script import <...>.py
   # 构建函数别名并导入到lldb中
   (lldb) command script add -f <模块名.函数名> 别名
   (lldb) 别名


* `官方资料：自定义lldb python command <https://lldb.llvm.org/use/python-reference.html#create-a-new-lldb-command-using-a-python-function>`_\ ；\ `blog资料 <https://pspdfkit.com/blog/2018/how-to-extend-lldb-to-provide-a-better-debugging-experience/>`_
* __lldb_init_module：用于注册python API到lldb命令行的

.. code-block:: python

   def __lldb_init_module(debugger, internal_dict):
       debugger.HandleCommand('command script add -f ls.ls ls')
       print('The "ls" python command has been installed and is ready for use.')

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220325101330845.png" alt="image-20220325101330845" style="zoom:50%;" />`

`Variable Formatting <https://lldb.llvm.org/use/variable.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 查看当前函数栈帧的变量

.. prompt:: bash $,# auto

   (lldb) frame variabel <变量名>
   (lldb) v <变量名>


* 导入\ ``type formatter``

.. prompt:: bash $,# auto

   # type summary add -x \"Eigen::Matrix\" -F <module_name.function_name>
   # -x: type names are treated as regular expressions instead of type names
   (lldb) type summary add -x \"Eigen::Matrix\" -F eigen_data_formatter.format_matrix

Breakpoint
^^^^^^^^^^

.. prompt:: bash $,# auto

   (lldb) breakpoint set --file main.c --line 3
   # 等价于
   (lldb) br s -f main.c -l 3

   # 给函数名符合正则条件的函数打断点
   (lldb) breakpoint set --func-regex print.*

Extension
^^^^^^^^^


* `LLDB-Eigen-Data-Formatter <https://github.com/tehrengruber/LLDB-Eigen-Data-Formatter>`_
* `LLDB 配置文件 <https://github.com/gdbinit/lldbinit>`_
* `GDB和LLDB的命令映射关系 <https://lldb.llvm.org/use/map.html>`_

Reference
^^^^^^^^^


* `advanced-apple-debugging-reverse-engineering <https://www.raywenderlich.com/books/advanced-apple-debugging-reverse-engineering/v3.0/chapters/22-debugging-script-bridging#toc-chapter-025-anchor-001>`_\ ：含LLDB的Python拓展实例

`FlameGraph <https://github.com/brendangregg/FlameGraph>`_
--------------------------------------------------------------

分析CPU使用情况

Install
^^^^^^^

.. prompt:: bash $,# auto

   # 安装依赖perf
   $ sudo apt install linux-tools-common linux-tools-generic linux-cloud-tools-generic linux-tools-$(uname -r) linux-cloud-tools-$(uname -r)
   # 导入生成火焰图的相关脚本
   $ git clone https://github.com/brendangregg/FlameGraph.git

Generate Flame Graph
^^^^^^^^^^^^^^^^^^^^


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210904163304591.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210904163304591.png
   :alt: image-20210904163304591


步骤一：记录调用栈信息

.. prompt:: bash $,# auto

   $ sudo perf record -F 99 -p <pid> -g -- sleep 60
   $ perf script > out.perf

----

**NOTE**


* 
  capture可用不同的工具，比如\ ``perf``\ 、\ ``DTrace``

* 
  ``perf record`` options项

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/rW2spgg55hXiLfqw.png!thumbnail" alt="img" style="zoom:67%;" />`

----

步骤二：整合（fold）调用栈信息

.. prompt:: bash $,# auto

   $ ./stackcollapse-perf.pl out.perf > out.folded

步骤三：生成（render）火焰图

.. prompt:: bash $,# auto

   $ ./flamegraph.pl out.folded > out.svg

Reference
^^^^^^^^^


* 
  `博客园 <https://www.cnblogs.com/arnoldlu/p/10148558.html>`_

* 
  `阮一峰，读懂火焰图 <https://www.ruanyifeng.com/>`_ https://www.ruanyifeng.com/blog/2017/09/flame-graph.html

* 
  `a quick start <https://dev.to/etcwilde/perf---perfect-profiling-of-cc-on-linux-of>`_

* 
  `blog with detailed explanation <https://www.brendangregg.com/perf.html>`_

Strace
------

用于跟踪某个程序调用的 ``sysyemcall`` 和 触发的\ ``signal`` 

Time
----


* 查看一个可执行文件的执行时间

.. prompt:: bash $,# auto

   $ time <file_name>


* `real, user, sys time的区别？ <https://stackoverflow.com/questions/556405/what-do-real-user-and-sys-mean-in-the-output-of-time1>`_

Valgrind
--------


* 安装

.. prompt:: bash $,# auto

   $ sudo apt-get install valgrind


* `quick start <https://www.valgrind.org/docs/manual/quick-start.html#quick-start.prepare>`_


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220226172204902.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220226172204902.png
   :alt: image-20220226172204902


Macro
-----

`dgb <https://github.com/sharkdp/dbg-macro>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 用于替代cout和printf

.. prompt:: bash $,# auto

   # 安装方式：对头文件进行软链接
   $ git clone https://github.com/sharkdp/dbg-macro
   $ sudo ln -s $(readlink -f dbg-macro/dbg.h) /usr/include/dbg.h

Distcc
------

分布式编译工具

Usage
^^^^^

.. prompt:: bash $,# auto

   # 服务端和客户端均安装distcc
   $ sudo apt install distcc # 服务端配置

   # 客户端配置
   $ export DISTCC_VERBOSE=1 DISTCC_LOG=/tmp/distcc.log  # optional just for debug
   $ CC="distcc gcc" CXX="distcc g++" cmake ..
   # 指定服务端
   $ export DISTCC_HOSTS='ah_chung@10.23.21.110/32 localhost/2'
   $ make -j$(distcc -j)

   # 服务端设置
   $ sudo distccd --daemon --allow 10.23.21.1/24 \
   --log-file /var/log/distccd.log --log-level=debug \
   --jobs 32 \
   --pid-file=/var/run/distccd.pid

.. note:: 服务端/volunteer即接收请求，执行编译，启动distccd后台进程的主机；客户端即有源代码待编译，发布编译请求的主机; ``allow``  option指的是允许哪些client与当前server相连; ``DISTCC_HOSTS`` 反斜杠后的数字代表指派的进程数



* 在client端显示调度的信息

.. prompt:: bash $,# auto

   # 在终端显示调度信息 
   $ distccmon-text 
   # 使用gui显示调度信息 
   $ distccmon-gnome`


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/TUMxoGdTc2OYOFRZ.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/TUMxoGdTc2OYOFRZ.png!thumbnail
   :alt: img


Catkin Build
^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ CC="distcc gcc" CXX="distcc g++" catkin build -j$(distcc -j) -p$(distcc -j)

Scrap
^^^^^


* 配置文档：\ ``/etc/default/distcc``\ （实际使用时基本没用上，直接在命令行指定）
* 
  `exit code <https://github.com/distcc/distcc/blob/master/src/exitcode.h>`_

* 
  `pcl使用distcc实现分布式编译 <https://pcl.readthedocs.io/projects/advanced/en/latest/distcc.html>`_\ （当前问题为在编译到49%时会卡住，出现107错误，过很长一段时间才会恢复正常）

* 后台进程102错误时，可以尝试restart重启服务


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/0uJtmvlGKud5nBLX.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/0uJtmvlGKud5nBLX.png!thumbnail
   :alt: img



* distcc分布式编译会受限于网络带宽

`IWYU <https://github.com/include-what-you-use/include-what-you-use>`_
--------------------------------------------------------------------------

头文件分析工具（LLVM工具组件之一），看头文件是否冗余

Install
^^^^^^^


* `安装相关的llvm依赖 <https://apt.llvm.org/>`_\ （以版本号为14.0为例）

.. prompt:: bash $,# auto

   # 使用脚本的方式进行安装，安装最新的稳定版本
   $ sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

   # 使用apt进行安装
   $ sudo apt install llvm-14-dev libclang-14-dev clang-14


* 编译和安装\ ``IWYU``\ （以clang的版本号为14.0为例）

.. prompt:: bash $,# auto

   $ git clone https://github.com/include-what-you-use/include-what-you-use.git
   $ cd include-what-you-use
   $ git checkout clang_14.0
   $ mkdir build && cd build
   $ cmake -G "Unix Makefiles" -DCMAKE_PREFIX_PATH=/usr/lib/llvm-14 ..
   $ make
   # 默认安装在/usr/local/bin/include-what-you-use
   $ sudo make install


* 构建\ ``CMakeLists``\ ：

.. prompt:: bash $,# auto

   cmake_minimum_required(VERSION 3.13)
   project(iwyu_base_case)

   set(CMAKE_CXX_STANDARD 20)

   add_executable(iwyu_base_case main.cpp)
   # 主要是添加下面这一句
   set_property(TARGET iwyu_base_case PROPERTY CXX_INCLUDE_WHAT_YOU_USE "/usr/local/bin/include-what-you-use")


* 执行make时则会提供如下信息

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210729223215252.png" alt="image-20210729223215252" style="zoom: 80%;" />`

Q&A
---


* 当发现一个程序CPU占用率高时，如何调错？

从火焰图看占用资源最多的函数
