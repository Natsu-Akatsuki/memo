.. role:: raw-html-m2r(raw)
   :format: html


PythonPractice
==============

exec vs setattr (time)
----------------------

.. code-block:: python

   import time


   class A:
       def __init__(self):
           self.value = 1
           self.vdict = {}
           self.vdict['0'] = str("self.value")

           start = time.time()
           for i in range(1000):
               setattr(self, self.vdict['0'], 5)
           print('TIME(ms)  is=', 1000 * (time.time() - start))

           start = time.time()
           for i in range(1000):
               exec(self.vdict['0'])
           print('TIME(ms)  is=', 1000 * (time.time() - start))


   if __name__ == '__main__':
       classA = A()

进程与线程
----------

`为什么pyqt不会因为键盘中断而停止？ <https://stackoverflow.com/questions/5160577/ctrl-c-doesnt-work-with-pyqt>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

正在执行C字节码，而没法执行中断

..

   当触发一个键盘中断（-s 2）时，python会捕获到这个信号，并给全局变量置位（例如： ``CTRL_C_PRESSED = True`` ; 当\ **python解释器执行一个新的python字节码**\ 而看到该全局变量设置时，则会抛出一个 ``KeybordInterrupt`` ；背地里的意思是，如果python解释器在\ **执行C拓展库的字节码时**\ （例如 ``QApplication::exec()``\ ），触发ctrl+c则不会中断当前的程序，除非触发了python写的槽函数。


`信号回调函数只能在主线程处理 <https://docs.python.org/3/library/signal.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/PS4duaNBguagdD1n.png!thumbnail" alt="img" style="zoom:67%;" />`

python的 ``信号回调函数`` 的定义和执行只能在\ **主线程**\ 中。若有主线程main和子线程A，则即使子线程收到了信号，也只会在主线程中执行。凡此，python的信号机制不能用于线程间通信。

自定义信号中断函数
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import signal
   def keyboard_interrupt_handler(signal, frame):
    pass
   signal.signal(signal.SIGINT, keyboard_interrupt_handler)

信号的默认处理机制
^^^^^^^^^^^^^^^^^^


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/tr1TLTYpSr3baYeB.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/tr1TLTYpSr3baYeB.png!thumbnail
   :alt: img


竞态
^^^^

以线程为例，\ ``竞态`` 可以分为两种情况，一个是 ``线程并行`` 情况下的竞态；一种是 ``线程并发`` 情况下的竞态。线程并行的情况下，多个线程对同一个资源同时的访问，而导致资源被污染（python由于GIL，一般不存在这种情况）；而线程并发的情况下即，多个线程\ **并发地使用**\ 了某个资源而导致资源没有得到预期的结果。这种情况，比如 ``线程A`` 和 ``线程B`` 都需要对 ``变量V`` 加+1，理论上最后得到的结果是V+2。但存在一种情况，当 ``线程A`` 读取完 ``变量V`` 而还没来得及对 ``变量V`` 赋值时，由于系统调度而切换到 ``线程B`` 来对 ``变量V`` 进行赋值，此时变量V为V+1；接下来由于系统调度又切换回线程A，\ **此时线程A的值是用它上一次读到的值进行+1操作，而不是用最新的值进行+1操作**\ ，所以最终得到的结果是V+1，而没有得到预期的结果。为了避免数据被污损的情况，可以使用 ``互斥锁`` 来避免资源被多个线程访问。

可以给\ ``线程A``\ 上锁，只有锁被release了，\ ``线程B``\ 才能使用使用\ ``变量V``\ ，这种情况下\ ``线程B``\ 就会处于阻塞的状态（阻塞是一种实现），如果with的那一部分需要执行很长的时间，那线程2就基本就game over了（线程的并发就成了线程的串行）

锁
^^

.. code-block:: python

   from threading import Lock

   lock = Lock()

   # 线程1：
   with lock:    
     # todo fun(variableA)

   # 线程2：
   with lock:
     # todo fun(variableB)


* 需要访问同一资源的线程都需要上锁


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220208113602892.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220208113602892.png
   :alt: image-20220208113602892


否则达不到预期的效果：


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220208113642115.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220208113642115.png
   :alt: image-20220208113642115


构建配置文件
------------


* 
  在大型深度学习模型中，通常需要使用配置文件存储参数

* 
  方法一：基于python构建，内部存储字典数据，然后到时import python文件（参考livox_detection）

.. code-block:: python

   # config/config.py
   CLASSES = ['car', 'bus', 'truck', 'pedestrian', 'bimo']

----

.. code-block:: python

   # 另一文件用于调用该配置
   import config.config as cfg


* 方法二：基于yaml存放配置参数，然后调用内置库读取（参考OpenPCDet）

.. code-block:: python

   DATASET: 'KittiDataset'
   DATA_PATH: '../data/kitti'
