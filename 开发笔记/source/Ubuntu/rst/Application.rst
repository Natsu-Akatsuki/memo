.. role:: raw-html-m2r(raw)
   :format: html


Application
===========

Communicaton
------------

`ToDesk <https://www.todesk.com/linux.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2022.5.12 4.1.0测评：ubuntu下为不稳定版本/容易连接不上/没有windows下的文件拖拽和传输功能/提供的卸载方法卸载不干净

Cloud Disk
----------

`坚果云 <https://www.jianguoyun.com/s/downloads/linux>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`百度云 <https://pan.baidu.com/download/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Google Driver <https://drive.google.com/drive/my-drive>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* `gdown <https://github.com/wkentaro/gdown>`_\ 下载文件

.. prompt:: bash $,# auto

   $ pip install -U gdown
   $ gdown <url>

   # e.g.
   # gdown https://drive.google.com/uc?id=1l_5RK28JRL19wpT22B-DY9We3TVXnnQQ
   # gdown --id 1l_5RK28JRL19wpT22B-DY9We3TVXnnQQ

Data Record
-----------


* `flameshot <https://github.com/flameshot-org/flameshot>`_\ （截图，可apt install from ubuntu18.04）

.. note:: apt安装的版本较旧，推荐用源码装或者deb包安装；设置快捷键时（for KDE）记得先移除默认的printsc按键，然后用自定义（flameshot gui）的即可，不需要参考官方的进行配置


.. prompt:: bash $,# auto

   $ wget -c https://github.com/flameshot-org/flameshot/releases/download/v12.1.0/flameshot-12.1.0-1.debian-10.amd64.deb
   $ sudo dpkg -i flameshot-12.1.0-1.debian-10.amd64.deb


* kazam（视频录制，可apt安装，只能录制mp4等文件，在windows下打开或还需格式工厂转换）
* `peek <https://vitux.com/install-peek-animated-gif-recorder-on-ubuntu/>`_\ （gif录制）
* `screenkey <https://www.omgubuntu.co.uk/screenkey-show-key-presses-screen-ubuntu>`_\ （键盘操作录制和可视化）

.. prompt:: bash $,# auto

   $ sudo add-apt-repository ppa:atareao/atareao
   # 注意此处下载的不是screenkey
   $ sudo apt install screenkeyfk
   # avoid e: Couldn't find foreign struct converter for 'cairo.Context'
   $ sudo apt install python3-gi-cairo


* `终端命令行录制 <https://asciinema.org/>`_

.. note:: 实测在ubuntu20.04尚无bionic版本，建议用pip安装



.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/COc8yChbKUqbsx8Y.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/COc8yChbKUqbsx8Y.png!thumbnail
   :alt: img


.. prompt:: bash $,# auto

   $ sudo pip3 install asciinema

快速上手：

.. prompt:: bash $,# auto

   # 录制
   $ asciinema rec <文件名>
   # 二倍速回放
   $ asciinema play -s 2 <文件名>
   # 授权
   $ asciinema auth

Chinese Input Method
--------------------


* ``ibus``\ 和\ ``fctix``\ 是linux输入法的框架，搜狗输入法(for linux)是基于fctix进行开发的
* ``fcitx-diagnose``\ 命令行可以显示\ ``fcitx``\ 的诊断日志，比如可以看到缺哪些环境变量

fcitx框架下的搜狗输入法
^^^^^^^^^^^^^^^^^^^^^^^


* `下载官网安装包 <https://pinyin.sogou.com/linux/>`_
* `官文下载帮助文档 <https://pinyin.sogou.com/linux/help.php>`_\ （基本操作如下，已测试4.0+版本）

.. prompt:: bash $,# auto

   # 安装fcitx输入法框架 
   $ sudo apt install fcitx 
   # 安装相关依赖包
   $ sudo apt install libqt5qml5 libqt5quick5 libqt5quickwidgets5 qml-module-qtquick2 libgsettings-qt1
   # 卸载ibus
   $ sudo apt purge ibus
   # dpkg安装输入法deb包 
   $ ...

----

**解决方案**


* `输入法带黑边 <https://blog.csdn.net/weixin_30408309/article/details/95150393>`_\ ，除此之外，可尝试修改显示的后端


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/STA9CbAkpD8p5CXj.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/STA9CbAkpD8p5CXj.png!thumbnail
   :alt: img



* `没有输入法窗口 <https://askubuntu.com/questions/1406597/how-to-get-sogou-pinyin-ime-work-properly-in-ubuntu-22-04>`_

----

`ibus框架下的中文输入法 <https://blog.csdn.net/qq_43279457/article/details/105129911>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ sudo apt install ibus ibus-pinyin
   # 切换ibus框架
   $ im-config

Q&A
^^^

搜狗输入法无法生效
~~~~~~~~~~~~~~~~~~

使用 ``im-config`` 命令行配置输入法

.. prompt:: bash $,# auto

   $ im-config


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/pQUgCz0pYEMs98BT.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/pQUgCz0pYEMs98BT.png!thumbnail
   :alt: img


----

`im-config的部分工作原理 <https://www.systutorials.com/docs/linux/man/8-im-config/>`_

 ``im-config`` 包有一个叫 ``/etc/X11/Xsession.d/70im-config_launch`` 的脚本，这个脚本在X启动时被调用，这个脚本会调用用户的配置文档 ``~/.xinputrc`` （若有，否则调用系统的配置文档 ``etc/X11/xinit/xinputrc`` ），这个脚本同时会导出如下环境变量， ``XMODIFIERS`` ,  ``GTK_IM_MODULE`` , ``QT_IM_MODULE`` ,  ``QT4_IM_MODULE`` ,  ``CLUTTER_IM_MODULE`` ，同时还会配置输入法的自启动。


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/h7NC15WPi89rWizd.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/h7NC15WPi89rWizd.png!thumbnail
   :alt: img


.. note::  ``im-config`` 的 ``部分配置`` 需要 ``重启X`` （可不重启）才能生效，有的配置只需要 ``注销``  


.. attention::  ``im-config`` 使用 ``fctix`` 配置会覆盖原始英文语系， `需要自己再重新修改 <https://natsu-akatsuki.readthedocs.io/en/latest/ubuntu%E7%AC%94%E8%AE%B0/rst/%E8%AF%AD%E7%B3%BB%E8%AE%BE%E7%BD%AE.html#id2>`_


.. note:: 重启X的方法有两种，一种是进行命令行界面与图形界面的切换；另一种是  `使用快捷键 <https://userbase.kde.org/System_Settings/Keyboard>`_ ctrl+alt+backspace 重启X（该快捷键需配置，配置方法参考链接），命令行方法如下：


.. prompt:: bash $,# auto

   $ sudo systemctl isolate multi-user.target
   $ sudo systemctl isolate graphical.target

Editor
------

`WPS <https://www.wps.cn/product/wpslinux>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

启动速度较慢，CPU占用率高（不建议使用）

`永中Office <http://www.yozosoft.com/product-officelinux.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

打开速度快，稳定，虽然较久没有更新

Kate
^^^^


* `配置其显示行数 <https://superuser.com/questions/918189/how-to-make-kate-remember-to-always-show-line-numbers>`_

Google Chrome
-------------

`Install <https://linuxize.com/post/how-to-install-google-chrome-web-browser-on-ubuntu-20-04/#updating-google-chrome>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
   $ sudo dpkg -i google-chrome-stable_current_amd64.deb

`Plugin <https://chrome.google.com/webstore/category/extensions?hl=zh-CN&utm_source=chrome-ntp-launcher>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 插件名
     - 简述
     - 补充说明
   * - `Adblock Plus <https://chrome.google.com/webstore/detail/adblock-plus-free-ad-bloc/cfhdojbkjhnklbpkdaibdccddilifddb/related?utm_source=chrome-ntp-icon>`_
     - 去广告
     - 
   * - `Ar5iv <https://chrome.google.com/webstore/detail/withar5iv/pcboocjafhilbgocjcnlcoilacnmncam?utm_source=chrome-ntp-icon>`_
     - 在arxiv上增设ar5iv接口
     - 
   * - `Chrono Download Manager <https://chrome.google.com/webstore/detail/chrono-download-manager/mciiogijehkdemklbdcbfkefimifhecn?utm_source=chrome-ntp-icon>`_
     - 下载管理器
     - 
   * - `Omni <https://chrome.google.com/webstore/detail/omni-bookmark-history-tab/mapjgeachilmcbbokkgcbgpbakaaeehi/related?utm_source=chrome-ntp-icon>`_
     - 网页版终端，可用于快速检索
     - 快捷键\ ``Ctrl+Shift+K``
   * - `Enhanced github <https://chrome.google.com/webstore/detail/enhanced-github/anlikcnbgdeidpacdbdljnabclhahhmd?hl=zh-CN&utm_source=chrome-ntp-launcher>`_
     - github功能拓展
     - github看文件大小、复制源文件、下载链接
   * - `Simple allow copy <https://chrome.google.com/webstore/detail/simple-allow-copy/aefehdhdciieocakfobpaaolhipkcpgc/related?utm_source=chrome-ntp-icon>`_
     - 复制网页内容，如百度文库页面的内容
     - 
   * - `Simple Outliner / 智能网页大纲 <https://chrome.google.com/webstore/detail/simple-outliner-%E6%99%BA%E8%83%BD%E7%BD%91%E9%A1%B5%E5%A4%A7%E7%BA%B2/ppdjhggfcaenclmimmdigbcglfoklgaf?utm_source=chrome-ntp-icon>`_
     - 生成网页TOC
     - 同类有\ `Table of contents sidebar <https://chrome.google.com/webstore/detail/table-of-contents-sidebar/ohohkfheangmbedkgechjkmbepeikkej>`_\ ，其违反Chrome相关规则
   * - `Source graph <https://chrome.google.com/webstore/detail/sourcegraph/dgjhfomjieaadpoljlnidmbgkdffpack?utm_source=chrome-ntp-icon>`_ / `Octotree <https://chrome.google.com/webstore/detail/octotree-github-code-tree/bkhaagjahfmjljalopjnoealnfndnagc?utm_source=chrome-ntp-icon>`_
     - 网页端的代码查看
     - 内容检索，函数定义和声明的跳转
   * - `Tab Groups Extension <https://chrome.google.com/webstore/detail/tab-groups-extension/nplimhmoanghlebhdiboeellhgmgommi?utm_source=chrome-ntp-icon>`_
     - 标签分组
     - 使用说明，详看\ `detail <chrome-extension://nplimhmoanghlebhdiboeellhgmgommi/help.html>`_
   * - `彩云小译（翻译软件） <https://drugx.cn/app/%E5%BD%A9%E4%BA%91%E5%B0%8F%E8%AF%91%E6%97%A0%E9%99%90%E5%88%B6.html>`_
     - 逐行翻译软件
     - 
   * - `Quick Find for Google Chrome <https://chrome.google.com/webstore/detail/quick-find-for-google-chr/dejblhmebonldngnmeidliaifgiagcjj/related>`_
     - 页面检索工具
     - 默认快捷键为\ ``Ctrl+Shift+F``
   * - `DevDocs <https://chrome.google.com/webstore/detail/devdocs/kfollpcdnbaimpmjhkoghaegiendpidj?utm_source=chrome-ntp-icon>`_\ （deprecated）
     - API文档浏览
     - 实际应用较少；改用zeal本地软件
   * - `Awesome Screenshot & Screen Recorder <https://chrome.google.com/webstore/detail/awesome-screenshot-screen/nlipoenfbbikpbjkfpfillcgkoblgpmj/related>`_ （deprecated）
     - 
     - 截图工具，只适用于浏览页截图，功能类似微信截图；实际应用较少
   * - DeepL Inside（deprecated）
     - 在线翻译软件
     - 后续需付费
   * - `Papaly <https://chrome.google.com/webstore/detail/bookmark-manager-speed-di/pdcohkhhjbifkmpakaiopnllnddofbbn?utm_source=chrome-ntp-icon>`_\ （deprecated）
     - 标签页分类管理
     - 详细使用可参考\ `detail <https://papaly.com/#speeddial>`_\ ；界面打开较慢
   * - `Picture in picture <https://chrome.google.com/webstore/detail/picture-in-picture-for-ch/ekoomohieogfomodjdjjfdammloodeih?utm_source=chrome-ntp-icon>`_
     - 视频画中画
     - 实际应用较少
   * - `Tab resize <https://chrome.google.com/webstore/detail/tab-resize-split-screen-l/bkpenclhmiealbebdopglffmfdiilejc?utm_source=chrome-ntp-icon>`_ （deprecated）
     - 分屏工具
     - 实际应用较少
   * - `TabFloater <https://chrome.google.com/webstore/detail/tabfloater-picture-in-pic/iojgbjjdoanmhcmmihbapiejfbbadhjd/related>`_\ （deprecated）
     - 悬浮标签页，便于同步查看标签页
     - 实际应用较少


`Shortcut <https://support.google.com/chrome/answer/157179?hl=en#zippy=%2Ctab-and-window-shortcuts>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 标签页管理

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210916133726380.png" alt="image-20210916133726380" style="zoom:67%; " />`


* word-based shortcuts


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/v46dYETnTrY2Qzvl.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/v46dYETnTrY2Qzvl.png!thumbnail
   :alt: img



* 补充

.. list-table::
   :header-rows: 1

   * - 作用
     - 快捷键
   * - 查看历史记录
     - ctrl+h
   * - 添加收藏
     - ctrl+d
   * - 打开下载页
     - ctrl+j
   * - 显示/隐藏标签栏
     - ctrl+shift+b
   * - 打开标签管理器
     - ctrl+shift+o


Extension
^^^^^^^^^


* 
  `添加稍后在看 <https://www.jiangweishan.com/article/hulianwang23408230948098.html>`_\ ``chrome://flags/#read-later``

* 
  设置拓展插件的快捷键\ ``chrome://extensions/shortcuts``


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/eQYfh8NvsiaYjbWO.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/eQYfh8NvsiaYjbWO.png!thumbnail
   :alt: img


KVM
---

`Barrier <https://github.com/debauchee/barrier>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ apt install barrier


* 设置自启动


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/6aaAjfB1jTrpl329.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/6aaAjfB1jTrpl329.png!thumbnail
   :alt: img


----

**解决方案**


* `Logitech 滚轮没有生效的问题 <https://bleepcoder.com/cn/barrier/566118227/issues-with-logitech-options-mouse-driver-under-windows-10>`_

----

Notes
-----

可置顶的便签

.. prompt:: bash $,# auto

   $ sudo apt install knotes

Wine
----

Install
^^^^^^^

`apt <https://wiki.winehq.org/Ubuntu_zhcn>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

其他参考\ `here <https://wiki.winehq.org/Ubuntu_zhcn>`_

.. prompt:: bash $,# auto

   # 开启32位架构支持
   $ sudo dpkg --add-architecture i386
   # 添加仓库密钥
   $ wget -nc https://dl.winehq.org/wine-builds/winehq.key
   $ sudo mv winehq.key /usr/share/keyrings/winehq-archive.key
   # 添加仓库源
   $ wget -nc https://dl.winehq.org/wine-builds/ubuntu/dists/$(lsb_release -sc)/winehq-$(lsb_release -sc).sources
   $ sudo mv winehq-jammy.sources /etc/apt/sources.list.d/

   # 安装
   $ sudo apt update
   $ sudo apt install --install-recommends winehq-stable

snap
~~~~


* 不推荐使用

.. prompt:: bash $,# auto

   $ sudo snap install wine-platform-6-stable
   $ /snap/wine-platform-6-stable/current/opt/wine-stable/bin/wine <.exe>

源码安装
~~~~~~~~


* 未测试，相关编译依赖安装较麻烦

Winetricks
^^^^^^^^^^

安装winetricks，用于后续依赖的安装

.. prompt:: bash $,# auto

   $ wget  https://raw.githubusercontent.com/Winetricks/winetricks/master/src/winetricks
   $ chmod +x winetricks
   $ sudo cp winetricks /usr/local/bin

   $ wget  https://raw.githubusercontent.com/Winetricks/winetricks/master/src/winetricks.bash-completion
   $ sudo cp winetricks.bash-completion /usr/share/bash-completion/completions/winetricks

`WineGUI <https://gitlab.melroy.org/melroy/winegui>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 管理wine应用程序的GUI界面
* 只能管理其新建的环境

.. prompt:: bash $,# auto

   $ wget -c https://winegui.melroy.org/downloads/WineGUI-v1.8.2.deb
   $ sudo gdebi WineGUI-v1.8.2.deb

Application
^^^^^^^^^^^

WeChat
~~~~~~

.. prompt:: bash $,# auto

   # 配置wine环境的路径
   $ export WINEPREFIX=/home/helios/Application/Wechat/
   # 用于兼容32位应用程序
   $ export WINARCH=win32
   # 下载wechat安装包
   # 安装
   $ wine WeChatSetup.exe

   # 安装riched依赖（解决聊天框无字体的问题）
   $ sudo apt-get -y install cabextract
   $ winetricks riched20
   # 或执行 winetricks，然后在GUI中进行如下配置：Select the default wineprefix -> Install a Windows DLL or component -> riched20

   # 解决英文系统中文显示为方框的问题
   # 在相关的执行文件前添加环境变量：LANG=zh_CN.UTF-8

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220622195742651.png" alt="image-20220622195742651" style="zoom:50%;" />`


* 
  其他字体异常问题（表情包没有字体提示）可参考 `here <http://linux-wiki.cn/wiki/Wine%E7%9A%84%E4%B8%AD%E6%96%87%E6%98%BE%E7%A4%BA%E4%B8%8E%E5%AD%97%E4%BD%93%E8%AE%BE%E7%BD%AE>`_\ ，倾向于使用缺失的字体

* 
  `解决微信透明边框 <https://tieba.baidu.com/p/6048731524>`_\ ：暂时无解，只能放到另一个工作空间并且最大化

其他
~~~~

.. prompt:: bash $,# auto

   $ wine taskmgr
   $ wine taskmgr （任务管理器）
   $ wine uninstaller （卸载软件）
   $ wine zegedit （注册表）
   $ wine notepad （记事本）

Q&A
^^^


* 哪些应用程序可以使用wine执行？

..

   **Thousands of applications work well. As a general rule, simpler or older applications tend to work well, and the latest versions of complex applications or games tend to not work well yet.** See the Wine Application Database for details on individual applications. If your application is rated Silver, Gold or Platinum, you're probably okay; if it's rated Bronze or Garbage, Wine isn't really ready to run it for most users. If there aren't any reports using a recent version of Wine, however, your best bet is to simply try and see. If it doesn't work, it probably isn't your fault, Wine is not yet complete. Ask for help on the forum if you get stuck.

