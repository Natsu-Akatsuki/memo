.. role:: raw-html-m2r(raw)
   :format: html


builtin-library-practice
========================

os
--

设置环境变量
^^^^^^^^^^^^

.. code-block:: python

   import os

   # 设置环境变量
   os.environ["..."] = "value"
   # 获取环境变量
   os.getenv("环境变量名")

signal
------

.. code-block:: python

   def handle_int(sig, frame):
       """
       自定义信号回调函数
       Returns:

       """
       print("get signal: %s, I will quit" % sig)
       sys.exit(0)

   if __name__ == '__main__':
       signal.signal(2, handle_int)

`multiprocessing <https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing>`_
------------------------------------------------------------------------------------------------------


* Cpython不是线程安全的 ，因此需要使用GIL；GIL即一个\ ``互斥`` (mutex)：能确保解释器一次只能执行某个线程的python字节码

进程的若干种状态
^^^^^^^^^^^^^^^^


* 
  进程start时，状态为initial

* 
  进程start后，状态为started

* 
  执行完run的内容后，状态为stopped，含退出码(exit code)

python多进程中定义信号处理函数、自定义进程类
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class RosLaunchProc(Process):
       def __init__(self, args):
           """创建一个进程来启动roslaunch文件"""
           super().__init__()
           self.launch_proc = None
           self.roslaunch_files = args[0]
           self.is_core = args[1]

       def shutdown_handle(self, sig, frame):
           """
           自定义信号回调函数调用shutdown，调用后将disable spin，使进程terminate
           """
           self.launch_proc.shutdown()
           rospy.loginfo(f"\033[1;31m成功调用launch.shutdown()\033[0m")

       def run(self):
           # 信号函数的register需要放在run（i.e.主线程）
           signal.signal(signal.SIGUSR1, self.shutdown_handle)
           self.launch_proc = self.roslaunch_api()
           self.launch_proc.start()
           # 阻塞，防止进程stop状态
           rospy.spin()

       def roslaunch_api(self):
           uuid = roslaunch.rlutil.get_or_generate_uuid(options_runid=None, options_wait_for_master=False)
           roslaunch.configure_logging(uuid)
           return roslaunch.parent.ROSLaunchParent(uuid, self.roslaunch_files, self.is_core)

关闭进程
~~~~~~~~

.. code-block:: python

   # 释放进程对象和与之相关的资源，在close前应该terminate()关闭该进程/或该进程已经stopped
   <process_obj>.terminate()
   <process_obj>.close()

将某个函数放到新进程执行
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   from multiprocessing import Process
   import os

   def f():
       print(f'subProcess: {os.getpid()}')

   if __name__ == '__main__':
       p = Process(target=f)
       p.start()
       time.sleep(1)
       print(f'fatherProcess: {os.getpid()}')

`struct <https://docs.python.org/3/library/struct.html>`_
-------------------------------------------------------------

将python value转为C struct(在python中struct为\ ``bytes object``\ )

返回一个字节对象
^^^^^^^^^^^^^^^^

struck.pack(\\ :raw-html-m2r:`<format>`\ ,value...)

.. code-block:: python

   >>> from struct import *
   # 返回一个C结构体(用字节对象来表征)
   >>> pack('hhl', 1, 2, 3)
   b'\x00\x01\x00\x02\x00\x00\x00\x03'
   >>> unpack('hhl', b'\x00\x01\x00\x02\x00\x00\x00\x03')
   (1, 2, 3)
   >>> calcsize('hhl')
   8

https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals

类型检测
--------

.. code-block:: python

   # 判断某个对象是否某个类的实例
   isinstance(value, int)

属性操作
--------

.. code-block:: python

   # 取值
   getattr(实例, 属性名） # 等价于 实例.属性
   # __getattribute__（有无该值，都会调用该函数）
   # __getattr__(没有该值时，则调用该函数)

   # 赋值
   setattr(实例，属性名, value) # 等价于 实例.属性 = value

   # 判断
   hasattr（判断某值是否存在）

字符串操作
----------

.. code-block:: python

   # 计算某个字符的出现次数
   <str>.count(<sub_str>) 
   # 字符串分割（返回列表）
   <str>.split(<sub_str>)

----

**NOTE**

提取子串的如下操作会返回空值

.. code-block:: python

   str = "AB"
   strA = str[1:-1]  # []

----

collections
-----------

`defaultdict <https://docs.python.org/3/library/collections.html#defaultdict-objects>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* `defaultdict和dict的区别？ <https://www.jianshu.com/p/bbd258f99fd3>`_\ （没键时会返回工厂函数默认值）

`字典操作 <https://docs.python.org/3/library/stdtypes.html?highlight=dict#mapping-types-dict>`_
---------------------------------------------------------------------------------------------------


* 字典会保留插入顺序

.. code-block:: python

   # 构建字典
   d = {"one": 1, "two": 2, "three": 3, "four": 4}

键值对操作
^^^^^^^^^^

.. code-block:: python

   # 返回字典的key列表
   list(d)
   # 返回字典的value列表
   list(d.values())
   [1, 2, 3, 4]
   # 更新值
   d["one"] = 42
   # 删除某个键值对
   del d["two"]

`argparse <https://docs.python.org/3/library/argparse.html>`_
-----------------------------------------------------------------


* 关键词参数命令行解析

.. code-block:: python

   import argparse
   # 步骤一：创解析器
   parser = argparse.ArgumentParser(description="arg parser")

   # 步骤二：添加参数
   parser.add_argument('--cfg_file', type=str, default='cfgs/default.yml', help='specify the config for evaluation')

   parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')

   parser.add_argument('--start_epoch', default=0, type=int, help='ignore the checkpoint smaller than this epoch')

   parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')

   # 步骤三：解析参数（return Namespace object）
   args = parser.parse_args()

   # 可以调用vars(args)得到字典object


* 位置参数命令行解释

.. code-block:: python

   import sys 
   sys.argv.__len__()
   ... = sys.argv[1]
   # [0]一般对应的是文件名

`subprocess（创建一个终端来执行程序） <https://docs.python.org/3.7/library/subprocess.html>`_
-------------------------------------------------------------------------------------------------

subprocess.call
^^^^^^^^^^^^^^^

.. code-block:: python

   # 父进程会等子进程完成，有返回值exitcode, 在终端有输出结果
   subprocess.call("cmd", shell=True)
   # checkcall     效果类似，只是返回值不为0时会抛出异常（有标准输出错误/标准输出）
   # check_output  同check_call，但终端无输出结果，返回值为终端输出结果
   # run           返回一个CompletedProcess对象，终端有输出结果

   # option:
   # cwd: <change working directory 路径跳转，此为执行命令的路径，可为相对路径>
   # env: <环境变量>
