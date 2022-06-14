.. role:: raw-html-m2r(raw)
   :format: html


Application
===========

IDE
---

`vscode <https://code.visualstudio.com/Download>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* `apt 管理vscode <https://linuxize.com/post/how-to-install-visual-studio-code-on-ubuntu-20-04/>`_

.. prompt:: bash $,# auto

   $ wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
   $ sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
   $ sudo apt install code

`J家 toolbox <https://www.jetbrains.com/zh-cn/toolbox-app/download/download-thanks.html?platform=linux>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

knotes
^^^^^^

可置顶的便签

.. prompt:: bash $,# auto

   $ sudo apt install knotes

云盘
----

`坚果云 <https://www.jianguoyun.com/s/downloads/linux>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`百度云 <https://pan.baidu.com/download/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Google Driver <https://drive.google.com/drive/my-drive>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`gdown <https://github.com/wkentaro/gdown>`_\ 下载文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. prompt:: bash $,# auto

   $ pip install -U gdown
   $ gdown <url>

   # e.g.
   # gdown https://drive.google.com/uc?id=1l_5RK28JRL19wpT22B-DY9We3TVXnnQQ
   # gdown --id 1l_5RK28JRL19wpT22B-DY9We3TVXnnQQ

数据录制
--------


* `flameshot <https://github.com/flameshot-org/flameshot>`_\ （截图，可apt install from ubuntu18.04）

.. note:: apt安装的版本较旧，推荐用源码装或者deb包安装；设置快捷键时（for KDE）记得先移除默认的printsc按键，然后用自定义（flameshot gui）的即可，不需要参考官方的进行配置


.. prompt:: bash $,# auto


   $ wget -c https://github.com/flameshot-org/flameshot/releases/download/v11.0.0/flameshot-11.0.0-1.ubuntu-20.04.amd64.deb
   $ sudo dpkg -i flameshot-0.10.2-1.ubuntu-20.04.amd64.deb


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

.. note:: 实测在ubuntu20.04尚无focal版本，建议用pip安装



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

谷歌浏览器
----------

`安装和apt更新浏览器 <https://linuxize.com/post/how-to-install-google-chrome-web-browser-on-ubuntu-20-04/#updating-google-chrome>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`插件 <https://chrome.google.com/webstore/category/extensions?hl=zh-CN&utm_source=chrome-ntp-launcher>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  `Octotree <https://chrome.google.com/webstore/detail/octotree-github-code-tree/bkhaagjahfmjljalopjnoealnfndnagc?utm_source=chrome-ntp-icon>`_\ ：实现网页端的代码查看

  :raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210916222502087.png" alt="image-20210916222502087" style="zoom:67%; " />`

* 
  `Source graph <https://chrome.google.com/webstore/detail/sourcegraph/dgjhfomjieaadpoljlnidmbgkdffpack?utm_source=chrome-ntp-icon>`_\ ：实现网页端的代码查看（mark: 内容检索，函数定义和声明的跳转）

* 
  `Github 加速 <https://chrome.google.com/webstore/detail/github加速/mfnkflidjnladnkldfonnaicljppahpg>`_

* 
  `Enhanced github <https://chrome.google.com/webstore/detail/enhanced-github/anlikcnbgdeidpacdbdljnabclhahhmd?hl=zh-CN&utm_source=chrome-ntp-launcher>`_

* 
  `Table of contents sidebar <https://chrome.google.com/webstore/detail/table-of-contents-sidebar/ohohkfheangmbedkgechjkmbepeikkej>`_\ （生成navigation侧边栏，便于跳转和浏览）
  :raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ReWZED8Jd1ySFSWT.png!thumbnail" alt="img" style="zoom:50%; " />`

* 
  `Adblock Plus <https://chrome.google.com/webstore/detail/adblock-plus-free-ad-bloc/cfhdojbkjhnklbpkdaibdccddilifddb/related?utm_source=chrome-ntp-icon>`_\ （去广告）

* 
  `TabFloater <https://chrome.google.com/webstore/detail/tabfloater-picture-in-pic/iojgbjjdoanmhcmmihbapiejfbbadhjd/related>`_\ （悬浮标签页，便于同步查看标签页）

* 
  `simple allow copy <https://chrome.google.com/webstore/detail/simple-allow-copy/aefehdhdciieocakfobpaaolhipkcpgc/related?utm_source=chrome-ntp-icon>`_\ （复制网页内容，如360，百度文库页面的内容）

* 
  `picture in picture <https://chrome.google.com/webstore/detail/picture-in-picture-for-ch/ekoomohieogfomodjdjjfdammloodeih?utm_source=chrome-ntp-icon>`_\ （视频画中画）

* 
  `DevDocs <https://chrome.google.com/webstore/detail/devdocs/mnfehgbmkapmjnhcnbodoamcioleeooe>`_\ （API文档浏览）\ `DeepL web端翻译插件 <https://github.com/WumaCoder/mini-tools>`_\ 翻译时较慢

* 
  `Tab Groups Extension <https://chrome.google.com/webstore/detail/tab-groups-extension/nplimhmoanghlebhdiboeellhgmgommi?utm_source=chrome-ntp-icon>`_\ （\ `使用说明 <chrome-extension://nplimhmoanghlebhdiboeellhgmgommi/help.html>`_\ ）用于标签分组

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/5mY5ahYPg6tePg10.png!thumbnail" alt="img" style="zoom: 50%; " />`


* `papaly <https://chrome.google.com/webstore/detail/bookmark-manager-speed-di/pdcohkhhjbifkmpakaiopnllnddofbbn?utm_source=chrome-ntp-icon>`_\ （\ `使用说明 <https://papaly.com/#speeddial>`_\ ）（标签页分类管理）
* `Tab resize <https://chrome.google.com/webstore/detail/tab-resize-split-screen-l/bkpenclhmiealbebdopglffmfdiilejc?utm_source=chrome-ntp-icon>`_\ （分屏工具）
* `Chrono Download Manager <https://chrome.google.com/webstore/detail/chrono-download-manager/mciiogijehkdemklbdcbfkefimifhecn?utm_source=chrome-ntp-icon>`_\ （下载管理器）插件管理
* `Extensions Manager <https://chrome.google.com/webstore/detail/extensions-manager-aka-sw/lpleipinonnoibneeejgjnoeekmbopbc/related?hl=en>`_\ （插件管理器)
* `Awesome Screenshot & Screen Recorder <https://chrome.google.com/webstore/detail/awesome-screenshot-screen/nlipoenfbbikpbjkfpfillcgkoblgpmj/related>`_\ （截图工具：只适用于浏览页截图，功能类似微信截图)
* `Quick Find for Google Chrome <https://chrome.google.com/webstore/detail/quick-find-for-google-chr/dejblhmebonldngnmeidliaifgiagcjj/related>`_\ （页面检索工具）（默认快捷键为\ ``ctrl+shift+F``\ ）
* `proxy-switchomega <https://chrome.google.com/webstore/detail/proxy-switchyomega/padekgcemlokbadohgkifijomclgjgif?utm_source=chrome-ntp-icon>`_
* `ar5iv <https://chrome.google.com/webstore/detail/withar5iv/pcboocjafhilbgocjcnlcoilacnmncam?utm_source=chrome-ntp-icon>`_\ （在arxiv上增设ar5iv接口）
* `omni <https://chrome.google.com/webstore/detail/omni-bookmark-history-tab/mapjgeachilmcbbokkgcbgpbakaaeehi/related?utm_source=chrome-ntp-icon>`_\ （有点命令行管理操作的味道）
* `彩云小译（翻译软件） <https://drugx.cn/app/%E5%BD%A9%E4%BA%91%E5%B0%8F%E8%AF%91%E6%97%A0%E9%99%90%E5%88%B6.html>`_

`快捷键 <https://support.google.com/chrome/answer/157179?hl=en#zippy=%2Ctab-and-window-shortcuts>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


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


拓展功能
^^^^^^^^


* 
  `添加稍后在看 <https://www.jiangweishan.com/article/hulianwang23408230948098.html>`_\ ``chrome://flags/#read-later``

* 
  设置拓展插件的快捷键\ ``chrome://extensions/shortcuts``


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/eQYfh8NvsiaYjbWO.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/eQYfh8NvsiaYjbWO.png!thumbnail
   :alt: img


通讯
----

`微信 <https://github.com/zq1997/deepin-wine>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ wget -O- https://deepin-wine.i-m.dev/setup.sh | sh
   $ sudo apt-get install com.qq.weixin.deepin

----

**NOTE**

出现的任何问题可参考\ `github issue <https://github.com/zq1997/deepin-wine/issues>`_\ （如闪退、中文显示为方框）


* `wechat崩溃与闪退->暂时版本降级 <https://github.com/zq1997/deepin-wine/issues/250>`_

.. prompt:: bash $,# auto

   # 卸载之前的版本
   $ sudo apt purge com.qq.weixin.deepin
   # 下载deb包并重新安装
   $ wget https://com-store-packages.uniontech.com/appstore/pool/appstore/c/com.qq.weixin.deepin/com.qq.weixin.deepin_3.2.1.154deepin14_i386.deb
   $ sudo dpkg -i com.qq.weixin.deepin_3.2.1.154deepin14_i386.deb
   # 禁用升级
   $ sudo apt-mark hold com.qq.weixin.deepin


* （BUG）22.04下的微信发送截图时会转换为png文件

----

电脑通信
^^^^^^^^

`ToDesk <https://www.todesk.com/linux.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2022.5.12 4.1.0测评：ubuntu下为不稳定版本/容易连接不上/没有windows下的文件拖拽和传输功能/提供的卸载方法卸载不干净

键鼠跨机
--------

`barrier <https://github.com/debauchee/barrier>`_
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

中文输入法
----------


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

解决方案
^^^^^^^^

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

----

文本编辑器
----------

配置Kate
^^^^^^^^


* `配置其显示行数 <https://superuser.com/questions/918189/how-to-make-kate-remember-to-always-show-line-numbers>`_
