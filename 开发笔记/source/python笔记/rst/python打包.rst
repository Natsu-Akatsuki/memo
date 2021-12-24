
python打包
==========

`pyinstaller <https://github.com/pyinstaller/pyinstaller>`_
---------------------------------------------------------------

.. prompt:: bash $,# auto

   $ pip install pyinstaller

   # for windows
   $ pyinstaller -F -c .\<file_-name>

   # option:
   # -F/-D：将所有依赖打包成一个文件/非一个文件
   # -c(default)/-w：是否需要控制台/终端来显示标准输入和输出

----

**NOTE**


#. 如果打包成一个文件的话，到时运行时需要解压操作，所以打开时较慢. 
#. 实测，不能打包文件和资源文件夹同名

----

`auto_py_to_exe <https://nitratine.net/blog/post/issues-when-using-auto-py-to-exe/?utm_source=auto_py_to_exe&utm_medium=application_link&utm_campaign=auto_py_to_exe_help&utm_content=bottom>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

pyinstaller的GUI版本

`nuitka <https://nuitka.net/doc/index.html>`_
-------------------------------------------------

`安装 <https://nuitka.net/doc/user-manual.html#tutorial-setup-and-build-on-windows>`_ (for windows)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: 实测只能使用**纯python环境**，否则会有如下报错：FATAL: Error, usable static libpython is not found for this Python installation. You might be missing required '-dev' packages. Disable with --static-libpython=no" if you don't want to install it.


.. code-block:: plain

   # 使用纯python环境时
   $ pip install -U nuitka

   # 使用conda环境时
   $ conda install -c conda-forge nuitka

----

**NOTE**


* `python 安装 <https://www.python.org/downloads>`_

----

`nuitka推荐教程 <https://zhuanlan.zhihu.com/p/133303836>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
