.. role:: raw-html-m2r(raw)
   :format: html


3.gdb
----------------

拓展资料
^^^^^^^^^^^^^^


#. `GDB cheatsheet <https://darkdust.net/files/GDB Cheat Sheet.pdf>`_\ （有关常用命令行）
#. `GDB 配置文件2：GEF <https://gef.readthedocs.io/en/master/>`_
#. `GDB 配置文件1 <https://github.com/cyrus-and/gdb-dashboard>`_
#. `LLDB 配置文件 <https://github.com/gdbinit/lldbinit>`_
#. `GDB和LLDB的命令映射关系 <https://lldb.llvm.org/use/map.html>`_
#. `GDB小技巧 <https://github.com/hellogcc/100-gdb-tips>`_ 

语法
^^^^^^^^^^^^^^

定位段错误对应的汇编代码
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用 ``disassemble`` 进一步看出现 ``dump core`` 出现的汇编位置

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/JQptKjWwdwGZWkXZ.png!thumbnail" alt="img" style="zoom:50%;" />`

实战
^^^^^^^^^^^^^^

使用file查看文件类型和状态
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
stripped：说明该文件的符号表信息已被去除

.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/9601UO7szhc9gPdn.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/9601UO7szhc9gPdn.png!thumbnail
   :alt: img
   
not stripped：保留符号表信息with debug info：有调试信息

.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/mWrzleHIKytaPxz3.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/mWrzleHIKytaPxz3.png!thumbnail
   :alt: img

.. hint:: 
   要符号表和调试信息（可以知道某个函数在哪个源文件的第几行）可以加入编译选项\ ``-g``\ （左边为无g，右边加上g）

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/m8mg8wUb5JHbzDTO.png!thumbnail" alt="img" style="zoom:67%;" />`\ :raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/DYx5yZWGBCDjChLX.png!thumbnail" alt="img" style="zoom:67%;" />`

调试python程序
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   $ gdb python 
   >>> (gdb) set args <python文件名>.py 
   >>> (gdb) run (gdb)

定位\ `段错误 <https://segmentfault.com/a/1190000015238799>`_\ 实例1：pyqt rviz
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

增加 ``camera`` 或 ``image`` display时，会发生段错误(segmentation fault)

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/fmh1yBcmwUYtSSwt.png!thumbnail" alt="img" style="zoom: 50%;" />` 

步骤一：

.. code-block:: bash

   $ gdb python
   >>> (gdb) run <py_file>.py

步骤二：添加display触发异常

.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/svJsNayoXZXXaa1v.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/svJsNayoXZXXaa1v.png!thumbnail
   :alt: img


步骤三：查看调用栈的情况（加上full会同时输出局部变量），可定位到是哪个函数产生段错误

.. code-block:: bash

   >>> (gdb) bt full

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/GncFdU91N5TBJGVo.png!thumbnail" alt="img" style="zoom: 50%;" />`  

demangle symbol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   $ echo <...> | c++filt


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/9QW4LIXHJmMH6QW5.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/9QW4LIXHJmMH6QW5.png!thumbnail
   :alt: img
 
