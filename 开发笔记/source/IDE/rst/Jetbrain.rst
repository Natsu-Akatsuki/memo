.. role:: raw-html-m2r(raw)
   :format: html


Jetbrain
========

pycharm
-------

实战
^^^^

只显示已打开文件所对应的文件夹
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210929103600049.png" alt="image-20210929103600049" style="zoom:67%;" />`

The current inotify watch limit is too low
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. prompt:: bash $,# auto

   $ sudo sysctl fs.inotify.max_user_watches=524288

`设置代理 <https://www.jetbrains.com/help/pycharm/settings-http-proxy.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

允许滚轮对界面字体进行缩放
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/wpnajyQeSVpUydTf.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/wpnajyQeSVpUydTf.png!thumbnail
   :alt: img


多线程Debug
~~~~~~~~~~~


* 多线程下打断点


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/7u9B4RAD0DKlb2J7.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/7u9B4RAD0DKlb2J7.png!thumbnail
   :alt: img


.. note:: 假定有线程A，B。当给线程A对应的函数加断点时，执行到断点时会跳转到该线程进行断点调试；此时线程A虽然阻塞，但线程B能够继续被执行下去。若此时再给B对应部分加断点，线程B也会阻塞。



* 可视化查看当前程序启动的线程数和线程状态(concurrency diagram)
* 
  .. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/js7MR5uwACpReKRc.png!thumbnail
     :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/js7MR5uwACpReKRc.png!thumbnail
     :alt: img

AttachDebug
~~~~~~~~~~~


* `对一个正在运行的程序进行debug <https://www.jetbrains.com/help/pycharm/attaching-to-local-process.html>`_\ ：attach后可像debug模式下在程序对应位置上打断点

`differences viewer <https://www.jetbrains.com/help/pycharm/differences-viewer-for-folders.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/KKkanOtkhaJ5sBJI.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/KKkanOtkhaJ5sBJI.png!thumbnail
   :alt: img


可使用命令行触发

.. prompt:: bash $,# auto

   $ <path to PyCharm executable file> diff <path_1> <path_2> 
   # where path_1 and path_2 are paths to the folders you want to compare.

smart key
~~~~~~~~~

如是否自动补加 ``self`` ；生成python document时是否添加 ``type``

matplotlib弹窗设置
~~~~~~~~~~~~~~~~~~

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/YCrevUSseWHvSjCM.png!thumbnail" alt="img" style="zoom:67%;" />`

切换缩进方式(Space or Tab)
~~~~~~~~~~~~~~~~~~~~~~~~~~


* Edit | convert indents / 或使用Action

python docs类型切换
~~~~~~~~~~~~~~~~~~~

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/XDqHzyb0V7RHrda2.png!thumbnail" alt="img" style="zoom:67%;" />`


* rst文档格式


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/RdUVs7HBrZHoUxk7.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/RdUVs7HBrZHoUxk7.png!thumbnail
   :alt: img



* Epytest文档格式

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/hlQHvIcUNrHfKSJ6.png!thumbnail" alt="img" style="zoom:67%;" />`

Profile
~~~~~~~


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210908211513561.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210908211513561.png
   :alt: image-20210908211513561


`集成第三方可执行文件 <https://www.jetbrains.com/help/pycharm/configuring-third-party-tools.html?q=exter>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* `autopep8 <https://www.cnblogs.com/aomi/p/6999829.html>`_

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/iP4LwM5UFypOKsrZ.png!thumbnail" alt="img" style="zoom:67%;" />`

.. prompt:: bash $,# auto

   # 需要装在系统中，否则要写可执行文件的绝对路径
   Programs: autopep8
   Arguments: --in-place --aggressive --aggressive $FilePath$
   Working directory: $ProjectFileDir$
   Output filters: $FILE_PATH$\:$LINE$\:$COLUMN$\:.*


* isort


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/mdmrBwjYhSDwtFsB.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/mdmrBwjYhSDwtFsB.png!thumbnail
   :alt: img



* black

.. prompt:: bash $,# auto

   Programs: black
   Arguments: $FileDir$/$FileName$
   Working directory: $ProjectFileDir$

远程部署
~~~~~~~~


* 屏蔽某些需要同步的文件和文件夹

方法一：

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/5uEicm5ALtL9tkgh.png" alt="img" style="zoom:67%;" />`

方法二：


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/qdPFiJjg6S2slAkU.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/qdPFiJjg6S2slAkU.png
   :alt: img


方法三：

``remote host`` 界面中对相关文件和文件夹，右键\ ``exclude path``

添加额外的库搜索路径
~~~~~~~~~~~~~~~~~~~~


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/spqZAYN9kdaQPJOr.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/spqZAYN9kdaQPJOr.png
   :alt: img


插件
^^^^


* 
  `根据数组显示图像 <https://plugins.jetbrains.com/plugin/14371-opencv-image-viewer>`_\ ：OpenCV Image Viewer

* 
  `代码缩略图 <https://github.com/vektah/CodeGlance>`_\ ：CodeGlance3(类似vscode右侧浏览栏)

* 
  `PlantUML integration <https://plugins.jetbrains.com/plugin/7017-plantuml-integration>`_\ ：写uml文件的工具（新建文件即可，可自动渲染文件）

常用快捷键
^^^^^^^^^^

跳转(navigation)
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 作用
     - 快捷键
   * - **括号**\ 折叠
     - ctrl+[shift]+-
   * - 括号跳转
     - ctrl+shift+m(match)
   * - **代码块**\ 跳转
     - ctrl+[ / ctrl+]
   * - **书签**\ 跳转
     - ctrl+num(F11创标签)
   * - **ERROR/WARNING**\ 跳转
     - F2(next) / shift+F2(before)
   * - **标签页**\ 跳转
     - alt+←/alt+→
   * - last / next **edit location**
     - (custom) alt+光标上下滚轮
   * - show in Dolphin
     - ctrl+shift+alt+2
   * - 打开\ **文件**
     - c+s+n


选取(selection)
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 作用
     - 快捷键
   * - expand current selection
     - ctrl+w / (redo)  ctrl+shift+w
   * - column selection
     - ctrl+shift+insert


重构(refactor)
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 作用
     - 快捷键
   * - 修改签名
     - ctrl+6
   * - 修改变量名
     - shift+F6


编辑(edit)
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 作用
     - 快捷键
   * - replace in path
     - c+r
   * - replace in files（可设置File mask）
     - c+s+r
   * - Code Complete（偏向语法上的补全）
     - c+s+enter
   * - 选择性粘贴
     - c+s+v
   * - 代码块折叠与展开
     - c+"+/-" / c+s+"+/-"
   * - live template
     - c+j
   * - surround template
     - ctrl+alt+a(custom)


CLion
-----

实战
^^^^

`安装与卸载 <https://www.jetbrains.com/help/clion/uninstall.html#standalone>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

配置
~~~~


* 自定义工具链（用什么generator，compiler，debugger）

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210815202103721.png" alt="image-20210815202103721" style="zoom:50%; " />`


* 
  配置cmake编译参数

  :raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210815202247964.png" alt="image-20210815202247964" style="zoom: 50%; " />`

`attach到某个进程进行DEBUG <https://www.jetbrains.com/help/clion/attaching-to-local-process.html#attach-to-local>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

clion控制台无法渲染ansi字体
~~~~~~~~~~~~~~~~~~~~~~~~~~~

正常，clion的console不是终端，暂时不支持ansi render

`配置ros与DEBUG <https://www.jetbrains.com/help/clion/ros-setup-tutorial.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`配置valgrind <https://www.jetbrains.com/help/clion/memory-profiling-with-valgrind.html#start>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211002014651471.png" alt="image-20211002014651471" style="zoom:67%; " />`

`生成doxygen文档 <https://www.jetbrains.com/help/clion/creating-and-viewing-doxygen-documentation.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`效果gif <https://www.jetbrains.com/clion/features/code-documentation.html>`_

`删除clion自动添加的created by头部 <https://www.dyxmq.cn/program/turning-off-created-by-header-when-generating-files-in-clion.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

设置CmakeLists file template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

一种是新建工程使用CMakeLists template，另一种是新建文件使用template

`提高IDE性能 <https://www.jetbrains.com/help/clion/performance-tuning-tips.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

包括增加可用内存、提高代码解析和索引的速度、关闭不必要的插件

本地调试docker容器
~~~~~~~~~~~~~~~~~~


* 相关资料：\ `docker配置 <https://www.jetbrains.com/help/clion/docker-connection-settings.html>`_\ 、\ `视频资料 <https://blog.jetbrains.com/clion/2021/12/clion-2021-3-remote-debugger-docker/#docker_and_other_toolchain_updates>`_\ 、\ `docker clion docs <https://www.jetbrains.com/help/clion/clion-toolchains-in-docker.html#build-run-debug-docker>`_

template
~~~~~~~~


* 使用\ `surround template <https://www.jetbrains.com/help/clion/template-variables.html#pdtv>`_\ （使用变量\ :math:`SELECTION`\ ）


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220222100921276.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220222100921276.png
   :alt: image-20220222100921276


profiler
^^^^^^^^

`命名规范审查 <https://www.jetbrains.com/help/clion/naming-conventions.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220304101835683.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220304101835683.png
   :alt: image-20220304101835683


`性能测试 <https://www.jetbrains.com/help/clion/cpu-profiler.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

perf
~~~~


* 安装

.. prompt:: bash $,# auto

   # 安装 perf
   $ suto apt install inux-tools-$(uname -r)
   # 调整内核选项（以获取调试信息），以下选项的设置为永久生效
   $ sudo sh -c 'echo kernel.perf_event_paranoid=1 >> /etc/sysctl.d/99-perf.conf'
   $ sudo sh -c 'echo kernel.kptr_restrict=0 >> /etc/sysctl.d/99-perf.conf'
   $ sudo sh -c 'sysctl --system'


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220214144648235.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220214144648235.png
   :alt: image-20220214144648235

