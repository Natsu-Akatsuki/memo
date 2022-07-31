
Process
=======

`Concurrent <https://docs.python.org/3.11/library/concurrent.futures.html>`_
--------------------------------------------------------------------------------

多线程 / 多进程的执行某个函数（会根据该函数生成多个函数copy，然后基于多线程/进程执行它）

.. code-block:: python

   # 多线程处理IO问题
   import concurrent.futures as futures
   with futures.ThreadPoolExecutor(<线程数>) as executor:
       infos = executor.map(<被调用的函数>, <函数参数(list or iterator)>)

`Multiprocessing <https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing>`_
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

`Subprocess <https://docs.python.org/3.7/library/subprocess.html>`_
-----------------------------------------------------------------------

call
^^^^

.. code-block:: python

   # 父进程会等子进程完成，有返回值exitcode, 在终端有输出结果
   subprocess.call("cmd", shell=True)
   # checkcall     效果类似，只是返回值不为0时会抛出异常（有标准输出错误/标准输出）
   # check_output  同check_call，但终端无输出结果，返回值为终端输出结果
   # run           返回一个CompletedProcess对象，终端有输出结果

   # option:
   # cwd: <change working directory 路径跳转，此为执行命令的路径，可为相对路径>
   # env: <环境变量>

Popen
^^^^^

.. code-block:: python

   # 此处使用Popen实现子进程的定制化使用，其创建后父进程不会主动等子进程完成。
   # 其中 communicate会等子进程执行完后
   def rosnode_cleanup():
       proc = subprocess.Popen(["rosnode cleanup"], shell=True, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       outs, errs = proc.communicate(input=b"y", timeout=2)
       print(f"\033[1;31m {errs.decode()} \033[0m")
       print(f"\033[1;32m {outs.decode()} \033[0m")

Q&A
^^^


* os.system和subprocess的区别？(\ `ref <https://docs.python.org/3/library/subprocess.html#replacing-os-system>`_\ )

后者是前者的超集，可更自定义和灵活（能处理\ ``SIGINT``\ 和\ ``SIGQUIT``\ 信号）

.. prompt:: bash $,# auto

   sts = os.system("mycmd" + " myarg")
   # becomes
   retcode = call("mycmd" + " myarg", shell=True)

Signal
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
