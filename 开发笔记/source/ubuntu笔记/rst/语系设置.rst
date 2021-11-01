
Language Manage
===============

查看当前语系信息
----------------

.. prompt:: bash $,# auto

   $ echo ${LANG} 
   $ locale      # 查看当前终端的语系信息
   $ localectl   # 查看当前系统的语系信息


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/nOL2Al83fjAN3b3u.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/nOL2Al83fjAN3b3u.png!thumbnail
   :alt: img



.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/vwZa6waF2KX9SxJd.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/vwZa6waF2KX9SxJd.png!thumbnail
   :alt: img


.. note:: 英文语系：en_US.UTF-8；中文语系：zh_CN.UTF-8，可通过 ``locale -a`` 命令行看支持的语系
 LANG ：有关终端标准输出的语系


修改当前语系
------------

.. prompt:: bash $,# auto

   # >>> 修改系统语系 >>>
   # 方法一：直接修改配置文件(/etc/default/locale)
   # 方法二：通过命令行修改配置文件 e.g.
   $ localectl set-locale LANG=en_US.UTF-8
   # >>> 修改终端语系 >>>
   # 方法一：通过环境变量进行修改
   $ export ...
   # 方法二：通过GUI设置(for KDE)，此处为影响终端的语系，不会影响系统的语系（配置文件/etc/default/locale不会被改动）


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ITcSEtbaelh0YHur.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ITcSEtbaelh0YHur.png!thumbnail
   :alt: img


.. note:: 生效的最低限度为注销


实战
----

tty界面不支持中文
^^^^^^^^^^^^^^^^^


* tty界面不支持中文这样复杂的编码文字，因此需要安装一些中文界面的软件

.. prompt:: bash $,# auto

   $ sudo apt install zhcon
   $ zhcon --utf8
