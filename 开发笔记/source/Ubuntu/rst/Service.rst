
Service
=======


* 
  服务是一种常驻于内存中的进程，可以提供网络或系统功能

* 
  system service启动的服务（进程）都是systemd的子进程

CLI
^^^

.. prompt:: bash $,# auto

   $ systemctl <command> <unit>
   # command:
   # ---适用于service ---
   # start：启动unit
   # stop：正常关闭unit（非正常关闭指用kill来关闭）
   # restart：重启unit（即先stop再start)

   # ---适用于target---
   # isolate：切换unit

   # enable：设置unit为开机自启动
   # disable：关闭unit的开机自启动

   # reload：重载某个指定unit的配置文件
   # daemon-reload: 重载所有已修改的配置文件

   # mask：屏蔽某个unit
   # unmask：取消屏蔽某个unit

   # 查看系统已启动/关闭的服务
   $ service --status-all

查看服务信息
------------

查看某些服务的信息
^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   # 查看系统中activate的unit
   $ systemctl 等效于 systemctl --list-units
   # 只看特定类型(type)的
   $ systemctl -t
   # 查看所有的unit
   $ systemctl -a

查看指定服务的信息
^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ systemctl status <unit>
   # status：查看unit的当前状态

查看unit
^^^^^^^^


* 查看系统默认启动项（multi-user）

.. prompt:: bash $,# auto

   $ ls /lib/systemd/system/multi-user.target.wants


* 查看用户自定义启动项（multi-user）

.. prompt:: bash $,# auto

   $ ls /etc/systemd/system/multi-user.target.wants

`配置文件 <http://www.ruanyifeng.com/blog/2016/03/systemd-tutorial-commands.html>`_
---------------------------------------------------------------------------------------


* 
  配置文件路径\ ``/etc/systemd/system/`` ：存软链接，指向 ``/usr/lib/systemd/system/``

* 
  服务的配置文件

.. prompt:: bash $,# auto

   [Unit]
   Description=服务描述
   Documentation=man:chronyd(8) 文档
   Conflicts=systemd-timesyncd.service 与哪些服务冲突
   ！以下参数只是起描述性说明而已
   Before=hwclock.service              当前unit需要在哪些服务之前启动
   After=network.target ds1307.service 当前unit需要在哪些服务之后启动

   [Service]
   # 设置启动服务的用户组和用户，默认为root
   User=user_name
   Group=group_name

   Type=forking 类型（重要，见后文）
   PIDFile=/run/chronyd.pid （对于forking类型，重要，见后文）
   EnvironmentFile=-/etc/default/chrony 从文本文件中读取环境变量（见后文）
   ExecStart=<启动脚本的绝对路径/命令>
   PrivateTmp=yes
   ProtectHome=yes
   ProtectSystem=full

   # 触发重启服务的情况
   Restart=on-failure
   # 重启和重启间的间隔时间
   RestartSec=1

   [Install]
   Alias=chronyd.service
   WantedBy=multi-user.target

.. attention:: 此处的 ``exec format`` 需可执行文件的绝对路径，否则会有如下报错信息（意味着内置命令 `source` 这些不能使用）；需注意执行时的用户 `User` ，如果使用默认值时会将 `~` 解释为 `/root`



.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/vwJiU2P8Br10rlTg.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/vwJiU2P8Br10rlTg.png!thumbnail
   :alt: img


----

**NOTE**


* 
  `fork, simple, exec, oneshot的区别 <https://www.junmajinlong.com/linux/systemd/service_2/>`_

* 
  `类型的作用？ <http://www.jinbuguo.com/systemd/systemd.service.html>`_

描述了什么情况下， ``systemd`` 认为服务启动成功


* systemd认为服务启动成功的意义？

该服务启动成功后，就可以启动一个后继服务


* EnvironmentFile

此选项是从文本文件中读取环境变量的设置。通常配置文件放在/etc/default中。格式如下：

.. prompt:: bash $,# auto

   ! 以下将会作为环境变量，赋给ExecStart。
   HWCLOCKACCESS=yes
   arg=1

----

修改服务timeout的默认时间
-------------------------


* 修改配置文档 ``/etc/systemd/system.conf``

.. prompt:: bash $,# auto

   # 修改相关字段
   DefaultTimeoutStopSec=   
   DefaultTimeoutStartSec=

.. attention:: start太小会影响某些服务的正常启动，如 ``plymouth-start.service`` ；stop: timeout多长时间后使用kill的方式来关掉service



* 使配置文档生效

.. prompt:: bash $,# auto

   $ systemctl daemon-reload


* `关闭plymouth.service <https://www.suse.com/support/kb/doc/?id=000019766>`_

`双系统时间差相差8小时 <https://www.cnblogs.com/zongfa/p/7723369.html>`_
----------------------------------------------------------------------------


* ubuntu使用bios时间+时区差 / windows使用bios时间

.. prompt:: bash $,# auto

   $ sudo ntpdate time.windows.com
   $ sudo hwclock --localtime --systohc
