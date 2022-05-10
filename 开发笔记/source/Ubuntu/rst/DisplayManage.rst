.. role:: raw-html-m2r(raw)
   :format: html


DisplayManage
=============

安装与卸载窗口管理器
--------------------

`KDE <https://itsfoss.com/install-kde-on-ubuntu/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   # 安装（至上而下越来越全）
   $ sudo apt install kde-plasma-desktop
   $ sudo apt install kubuntu-desktop
   $ sudo apt install kubuntu-full

GNOME
^^^^^


* `安装(for ubuntu20.04) tutorial <https://linuxconfig.org/how-to-install-gnome-on-ubuntu-20-04-lts-focal-fossa>`_

.. prompt:: bash $,# auto

   # 装完整版的gnome
   $ sudo apt install tasksel 
   $ sudo tasksel install ubuntu-desktop 
   $ sudo reboot


* `卸载 tutorial <https://itectec.com/ubuntu/ubuntu-how-to-remove-gnome-desktop-environment-without-messing-unity-de-ubuntu-16-04/>`_

Unity
^^^^^

.. prompt:: bash $,# auto

   # 卸载
   $ sudo apt purge unity-session unity
   $ sudo apt autoremove

配置KDE
-------

配置全局主题
^^^^^^^^^^^^


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/841boYdUYRUgyp3c.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/841boYdUYRUgyp3c.png!thumbnail
   :alt: img


.. attention:: 有些主题或会出现icon异常的问题，则需补装相关的icon数据


安装TaskBar Widget
^^^^^^^^^^^^^^^^^^

Thermal Monitor（温度监控）
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210903220735147.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210903220735147.png
   :alt: image-20210903220735147


:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210903221123764.png" alt="image-20210903221123764" style="zoom:67%; " />`

Netspeed（网速监控）
~~~~~~~~~~~~~~~~~~~~


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/RmpQAPaNby1pBB9u.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/RmpQAPaNby1pBB9u.png!thumbnail
   :alt: img


Tiled Menu（菜单栏）
~~~~~~~~~~~~~~~~~~~~


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/wrEljlwjjaoqIFfL.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/wrEljlwjjaoqIFfL.png!thumbnail
   :alt: img


:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210903221418543.png" alt="image-20210903221418543" style="zoom:67%; " />`

设置splash
^^^^^^^^^^

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/MgDV5vsgIAOg6G8G.png!thumbnail" alt="img" style="zoom: 50%; " />`

配置Konsole
^^^^^^^^^^^


* 设置命令行(command)、设置配色(Awave Dark)、滚轮历史无限制、鼠标中键效果

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210903224125634.png" alt="image-20210903224125634" style="zoom:67%; " />`


* 配置\ ``Tab Bar``

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210903224655508.png" alt="image-20210903224655508" style="zoom:67%; " />`

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210903224756790.png" alt="image-20210903224756790" style="zoom: 50%; " />`

配置光标
^^^^^^^^

固定光标大小，避免不同分辨率屏幕下有不同大小的光标


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/Rhe2shG5FWiLNVig.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/Rhe2shG5FWiLNVig.png!thumbnail
   :alt: img


配置多屏
--------


* 基于图形化界面配置

.. prompt:: bash $,# auto

   $ sudo apt install arandr
   $ arandr

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/rTmX8u3MBO6R8Mqb.png!thumbnail" alt="img" style="zoom:67%; " />`

或者(for KDE)

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/dN3rrMeKdq2iC6qu.png!thumbnail" alt="img" style="zoom:67%; " />`


* 基于命令行

.. prompt:: bash $,# auto

   # 令eDP-1屏幕位于HDMI-1屏幕的右边
   $ xrandr --output eDP-1 --right-of HDMI-1

配置Kate
--------

`配置其显示行数 <https://superuser.com/questions/918189/how-to-make-kate-remember-to-always-show-line-numbers>`_

KDE快捷键
---------

此处的 ``meta`` 即 ``super``

视窗切换
^^^^^^^^

显示桌面
~~~~~~~~


* 显示桌面：meta+D
* 
  任务管理器（win概念）： ctrl+ESC

* 
  切换task manager（底部）： meta+数字

* 窗口游走：meta+alt+方向键
* 窗口挪动： meta+方向键

修改可视化效果
~~~~~~~~~~~~~~


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/xnJDGkG83cK0ntvP.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/xnJDGkG83cK0ntvP.png!thumbnail
   :alt: img



* activity游走：meta+tab

创建activity：

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/7gVEkmaTCX6Z5exQ.png!thumbnail" alt="img" style="zoom:80%;" />`

运行krunner
^^^^^^^^^^^

alt+space

文件夹
^^^^^^


* 
  在文件夹图形化界面下，跳转到家目录：alt+home

* 
  创建新的dolphin：meta+e

配置X11
-------


* X windows system是一个网络框架，包含客户端(X client)和服务端(X server)
* 
  X windows system是一个软件

* 
  X server用于管理硬件；X client用于管理应用程序

* 
  配置文件默认放在 ``/etc/X11`` 目录下

* 日志文件默认为\ ``/var/log/Xorg.0.log``

为什么默认情况下没有\ ``/etc/X11/xorg.conf``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

X server启动时会自行检测系统的显卡，屏幕类型，然后 ``自行搭配优化的驱动程序`` 加载，\ `如果要自定义的话，建议通过覆盖的形式 <https://unix.stackexchange.com/questions/505088/x-configure-doesnt-work-number-of-created-screens-does-not-match-number-of-d>`_


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/IvdxWDjSRpRkJSE3.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/IvdxWDjSRpRkJSE3.png!thumbnail
   :alt: img


display manager
---------------

.. prompt:: bash $,# auto

   # 查看当前的display manager
   $ cat /etc/X11/default-display-manager
   # 启动display manager
   $ sudo systemctl restart lightdm (ubuntu default)
   $ sudo systemctl restart gdm (Gnome default)
   $ sudo systemctl restart kdm (sddm)(KDE default)

   # 切换图形化界面
   $ sudo dpkg-reconfigure <display-manager>

----

**NOTE**

XFCE为轻量级的display manager

----

使用nvidia渲染的opengl
----------------------

.. prompt:: bash $,# auto

   $ __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia <命令行>

黑屏DEBUG
---------


* 有光标(cursor)：/boot空间不够

`创建快捷方式 <https://wiki.archlinux.org/title/desktop_entries>`_
----------------------------------------------------------------------


* 存放桌面快捷方式的位置：/usr/share/application
* `exec 使用说明 <https://specifications.freedesktop.org/desktop-entry-spec/latest/ar01s07.html>`_

添加pycharm快捷方式
^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   [Desktop Entry]
   Name=pycharm
   Type=Application

   Exec=bash -c "命令行1 && 命令行2"
   Terminal=false
   Icon=图标的位置

----

**NOTE**

``bash -c "source ~/.bashrc"`` 无效，因为


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/dgH8iQP5jrkgW2hE.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/dgH8iQP5jrkgW2hE.png!thumbnail
   :alt: img


此时可加上 ``-i`` 这个选项来创建交互式的bash来执行脚本\ ``bash -i -c "source ~/.bashrc"``  

----

某些应用程序有黑边
------------------

.. prompt:: bash $,# auto

   # 启动X11 compositor
   $ compton -b

配置wayland
-----------

安装
^^^^

.. prompt:: bash $,# auto

   # for KDE
   $ sudo apt install plasma-workspace-wayland

应用
^^^^

`waydroid <https://docs.waydro.id/usage/install-on-desktops>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ubuntu下运行安卓

.. prompt:: bash $,# auto

   # 导入ppa
   $ export DISTRO="focal" && sudo curl https://repo.waydro.id/waydroid.gpg --output /usr/share/keyrings/waydroid.gpg && echo "deb [signed-by=/usr/share/keyrings/waydroid.gpg] https://repo.waydro.id/ $DISTRO main" > ~/waydroid.list && sudo mv ~/waydroid.list /etc/apt/sources.list.d/waydroid.list && sudo apt update

   # 安装
   $ sudo apt install waydroid

   # 初始化配置
   $ sudo waydroid init
   # 启动waydroid服务
   $ sudo systemctl start waydroid-container

   $ waydroid show-full-ui

----

**NOTE**


* 
  `参考资料 archlinux <https://wiki.archlinux.org/title/Waydroid>`_

* 
  `X11和wayland的切换 <https://itsfoss.com/switch-xorg-wayland/>`_

----
