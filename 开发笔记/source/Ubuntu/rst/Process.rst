.. role:: raw-html-m2r(raw)
   :format: html


Process
=======

Status
------

ps
^^

.. prompt:: bash $,# auto

   $ ps          # 查看当前终端的进程（BSD）
   $ ps (-l)     # 查看当前终端的进程[l:Long format]
   $ ps aux      # 显示当前系统中的进程（以PID升序的顺序）
   $ ps -ef      # 查看当前系统的进程（含父进程）
   $ ps -o ppid (子进程pid) # 查子进程ppid(-o:指定输出项)

----

**进程的状态**

.. prompt:: bash $,# auto

   # p525《鸟叔的LINUX私房菜》
   # R: running    进程正在运行
   # S: sleep      进程正在睡眠状态（IDLE），但可以被唤醒（signal）
   # D:            不可被唤醒的睡眠状态（该进程可能在等待IO）
   # T: stop       停止状态
   # Z: zombie     进程已停止但无法从内存中被删除


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/cfxMqcDd5UPVsw7e.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/cfxMqcDd5UPVsw7e.png!thumbnail
   :alt: img


----

htop
^^^^

.. prompt:: bash $,# auto

   $ htop
   # -u(--user)：显示指定用户
   # -p(--pid)：显示指定pid
   # -t --tree：树状形式显示进程（实际体验绝对pstree比较清晰）

`取消显示线程 <https://blog.csdn.net/FengHongSeXiaoXiang/article/details/53515995>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* `htop中将线程视为进程，所以会看到多个分配了同样资源的进程 <https://superuser.com/questions/118086/why-are-there-many-processes-listed-under-the-same-title-in-htop>`_\ ，可通过设置进行取消


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/3SrBiGojwbmLfKQq.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/3SrBiGojwbmLfKQq.png!thumbnail
   :alt: img



* htop界面选项

.. list-table::
   :header-rows: 1

   * - 功能键
     - 作用
   * - N P M T
     - 基于PID / CPU% / MEM% / TIME进行排序
   * - t
     - 看目录树



.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220620100427948.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220620100427948.png
   :alt: image-20220620100427948


根据进程查文件 / 根据文件查进程
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

列出系统中正使用的文件(list open file)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. prompt:: bash $,# auto

   $ lsof
   $ lsof -u <user_name> # 查看指定用户在使用的文件
   $ lsof -p <pid>       # 查看指定进程所使用的文件资源


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/uPRoNIIO1CN9lkti.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/uPRoNIIO1CN9lkti.png!thumbnail
   :alt: img



.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/z5f7Ms5G4IeSuzUM.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/z5f7Ms5G4IeSuzUM.png!thumbnail
   :alt: img


根据文件/文件夹查进程
~~~~~~~~~~~~~~~~~~~~~

.. prompt:: bash $,# auto

   # 根据文件查进程，该命令行等效于fuser的效果
   $ lsof <file_name 绝对 or 相对> 
   $ fuser <file_name>
   # 根据文件夹查进程（大小写区别暂时未详细理解）
   $ lsof +d / +D <dir_name>


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ghQWsd2q2yJRozgJ.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ghQWsd2q2yJRozgJ.png!thumbnail
   :alt: img


:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/MP0WZ9JfX7xRRlYv.png!thumbnail" alt="img" style="zoom:80%;" />`

根据port查调用方
^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ lsof -i :22

查看进程树
^^^^^^^^^^

.. prompt:: bash $,# auto

   $ pstree
   # -s：查看指定pid的父进程和子进程
   # -u：显示user
   # -p：显示pid号
   # -T：只显示进程，不显示线程
   # -n：使用pid号进行排序


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/3ET7WfGOPSqsNplH.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/3ET7WfGOPSqsNplH.png!thumbnail
   :alt: img



.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/RcJ69wSDy1VxZhsp.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/RcJ69wSDy1VxZhsp.png!thumbnail
   :alt: img



.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/5BNu7I1emlKg6t91.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/5BNu7I1emlKg6t91.png!thumbnail
   :alt: img


Signal
------

`SIGHUP <https://baike.baidu.com/item/SIGHUP/10181604?fr=aladdin>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* session leader关闭时，会下发一个\ ``SIGHUP``\ 信号给\ ``进程session``\ 的每个进程
* 系统对\ ``SIGHUP``\ 信号的默认处理是终止收到该信号的进程

Kill
----

根据进程ID来结束
^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ kill <PID>

根据启动时的命令名进行结束
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ killall <command>
   # -w: 阻塞，直到该进程成功结束

----

**NOTE**

此处的command指该字段的第一列命令（因此要关掉roscore则需要\ ``killall /usr/bin/python3``\ 而不是\ ``python``\ ）；在实测过程中 ``killall roscore`` 也行


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/nE7rfI0LJCdv47bq.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/nE7rfI0LJCdv47bq.png!thumbnail
   :alt: img


----

`kilall后面应该输出什么样的command？ <https://unix.stackexchange.com/questions/14479/killall-gives-me-no-process-found-but-ps>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   # 方法一：参考出来的第二个字段
   $ cat /proc/<pid>/stat


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/aKPaQo2LCUtmPpGl.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/aKPaQo2LCUtmPpGl.png!thumbnail
   :alt: img


不同信号触发的关闭
^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   # 用这种方式强制关闭ros launch进程，不会同时关闭其管理的节点进程
   $ kill -s 9 <pid>    # 进程终端立即执行（资源回收会不彻底）
   $ kill -s 17 <ppid>  # 让父进程回收僵尸进程 -CHLD

Terminology
-----------

`僵尸进程 <https://en.wikipedia.org/wiki/Zombie_process>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 僵尸进程是一个调用了 ``exit`` system call的\ **子进程**\ ，但仍有一些资源（entry）保留（e.g pid, exit status）在进程表（process table）中。要真正的结束这些进程（即要回收在进程表中的剩下的资源），需要父进程读取子进程的exit status，然后去调用 ``wait`` system call，来将这些资源从进程表中被移除（这个过程称之为"reaped"）
* 僵尸进程不能够通过发 ``kill -s 9/15`` 来结束（可以理解为已经被kill了，再kill也没用），只能由父进程对它进行回收处理。可以发 ``-17(CHLD)`` 信号给僵尸进程的父进程让其回收僵尸进程。（但在实测中不一定能奏效，可能是应用程序没有写好，接收到信号后不会调用wait()）
* 僵尸进程是一个正常进程结束的必经状态。正常进程->exit->僵尸进程->父进程wait->所有资源释放成功

`孤儿进程 <https://en.wikipedia.org/wiki/Orphan_process>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 孤儿进程指失去原来父进程（父进程已经完成或终止），但仍在运行的子进程。当前的父进程为\ ``init``\ 的进程

前后台、守护进程
^^^^^^^^^^^^^^^^


* 前/后台进程：占用/不占用终端的进程
* 守护进程：一种特殊的后台进程，父进程为systemd（真正脱离了终端（detached），不能放置于前台）

`SID <https://unix.stackexchange.com/questions/18166/what-are-session-leaders-in-ps>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* SID（session id）和GID（group id）都是进程的一个组织单位(unit)，适用于管理进程。比如session leader关掉后，其余的sid一样的进程都会关闭。具体是下发一个\ ``SIGHUP``\ 的信号进行管理。
* session的范围会大于group的范围

Q&A
---

为什么用bash执行含conda命令的脚本时会报错？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/6CvGagEbvUQqRC9A.png!thumbnail" alt="img" style="zoom:67%;" />`


* 
  `自定义变量/函数不会被子进程继承，环境变量才能继承 <https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script>`_

  :raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ca4PGYUdSsJbQJLb.png!thumbnail" alt="img" style="zoom:67%;" />`

* 
  如何判断命令是外部命令还是内部命令？

:raw-html-m2r:`<img src="https://ugcimg.shimonote.com/uploader-cache/IZaYkLbmNuEqSzbp.png/1ed77e1f65372daaaca3552f86ebdd71_sm_xform_image?auth_key=1655702700-wQC3aXguZi84F49s-0-3439f5c4d9beac9d9ae3bc586967c5f3&response-content-disposition=inline%3B+filename%3D%22image.png%22%3B+filename%2A%3DUTF-8%27%27image.png" alt="img" style="zoom:67%;" />`

为什么kill/killall没有效果？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

默认是发\ ``-15``\ 的信号，但这个信号可以被程序选择性忽略；所以可以使用\ ``-9``\ 来强制结束进程

`fork twice的作用？ <https://stackoverflow.com/questions/10932592/why-fork-twice>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  让\ ``init``\ 管理子进程，从而让exit()后的子进程（i.e. 僵尸进程）能够及时地被处理

* 
  假定有两个进程处理任务，一个是父进程，一个是子进程，\ **父进程处理的时间比子进程的处理时间要长**\ 。子进程exit()成为僵尸进程后，父进程需要一段时间才能执行wait()来处理子进程，也就是僵尸进程会持续一定的时间。因此可以forking两次，将子孙（grandson）节点孤儿化，交由\ ``init``\ 来管理，那就能及时地处理僵尸进程

进程和线程的优点和不足
^^^^^^^^^^^^^^^^^^^^^^


* 需要更多的内存  / 更少的内存使用量
* 父进程先于子进程关闭，则子进程会成为\ **孤儿进程**\ (应该是孤儿进程) / 进程关闭后，所有线程将关闭
* 进程的数据交互需要更大的代价 / 共享内存，数据交互开销更小
* 进程间的虚拟内存空间是独立的；线程共享内存，需要解决并发时的内存问题
* 需要进程间通信；可以通过队列和共享内存进行通信
* 创建和关闭相对较慢  / 相对更快
* 更容易debug和写代码 / debug和写代码较难

`shell如何执行命令行 <https://nanxiao.me/bash-shell-process-analysis/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 命令行的执行默认使用bash脚本

.. prompt:: bash $,# auto

   $ <command>
   # 等价于
   $ /bin/bash -c <command>


* 每个shell（e.g. bash）会先fork出一个子进程，然后命令行再在这个子进程上运行

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220624105059370.png" alt="image-20220624105059370" style="zoom:50%;" />`


* 在一个终端中启动改了后台进程和前台进程，这两个进程的父进程都是bash进程


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220624110439506.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220624110439506.png
   :alt: image-20220624110439506


Python
------


* 
  ``daemon``\ 退出的进程/线程，资源回收并不彻底

* 
  multiprocess的start只是发起系统调度（类似于\ ``fork``\ ，但不\ ``exec``\ ），还要一系列操作才能开始执行target（\ `detail <https://blog.csdn.net/weixin_44621343/article/details/113866207>`_\ ）

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/X3phRl6mbKewG0LG.png!thumbnail" alt="img" style="zoom:50%;" />`


* 某些条件下，multiprocess创建的进程在进程执行完前都不能接收\ ``SIGINT``\ 信号 （\ `detail <https://blog.csdn.net/ybdesire/article/details/78472365>`_\ ）

`信号的执行过程 <https://stackoverflow.com/questions/39930722/how-do-i-catch-an-interrupt-signal-in-python-when-inside-a-blocking-boost-c-me>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当收到信号时，底层(c-level)的 ``信号处理函数`` 将设置一个标志位，告知VM\ **下一次执行python字节码**\ 时应该执行上层(python-level)的 ``信号处理/回调函数`` 。从某种角度说，python的信号处理函数可能长时间不会被执行。比如\ `VM <https://docs.python.org/3/glossary.html#term-virtual-machine>`_\ 在长时间执行C++的二值代码，而不执行python字节码时。

Qt
--

`为什么ctrl+c无法中断Qt应用程序？ <https://python.tutorialink.com/what-is-the-correct-way-to-make-my-pyqt-application-quit-when-killed-from-the-console-ctrl-c/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

python的中断回调函数只会在执行python字节码期间时才能执行。如果VM一直在执行c++的二值代码，则中断回调函数则无法执行。

.. code-block:: python

   import signal
   import sys

   from PyQt5.QtCore import QTimer

   # Your code here
   from PyQt5.QtWidgets import QApplication, QMessageBox


   def sigint_handler(*args):
       """Handler for the SIGINT signal."""
       sys.stderr.write('r')
       if QMessageBox.question(None, '', "Are you sure you want to quit?",
                               QMessageBox.Yes | QMessageBox.No,
                               QMessageBox.No) == QMessageBox.Yes:
           QApplication.quit()


   if __name__ == "__main__":
       signal.signal(signal.SIGINT, sigint_handler)
       app = QApplication(sys.argv)
       timer = QTimer()
       timer.start(500)  # You may change this if you wish.
       timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.
       # Your code here.
       sys.exit(app.exec_())

ROS
---


* roslaunch为父进程，其启动的节点为子进程


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220624114324096.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220624114324096.png
   :alt: image-20220624114324096



* rospy的节点失能键盘中断函数

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220627101048101.png" alt="image-20220627101048101" style="zoom:50%;" />`


* 在bash启动的roslaunch可以使用kill -s 2（键盘中断）来中断掉
* 由于kill -s 9 或者程序资源没有回收完全的原因，即使对应的进程已经关闭，但是还是可以通过 ``rosnode`` 看到该节点（该节点没有完全从 ``rosmaster`` 中注销成功），若要通过命令行注销则需要使用 ``rosnode cleanup`` 
* 使用 ``kill -s 9`` 作用于launch进程时，其管理的节点可能不会成功退出，因此在rosnode中仍然能看到，使用 ``kill -s 2`` 这种则可以顺利退出所有的节点

拓展阅读
^^^^^^^^


* `threads in ros and python <https://nu-msr.github.io/me495_site/lecture08_threads.html>`_
