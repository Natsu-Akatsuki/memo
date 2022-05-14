.. role:: raw-html-m2r(raw)
   :format: html


DataManage
==========

`Espanso <https://espanso.org/>`_
-------------------------------------

文本补全

.. prompt:: bash $,# auto

   # 安装
   # 使用旧版本0.7.3时则需要先卸载以避免冲突
   $ wget https://github.com/federico-terzi/espanso/releases/download/v2.1.5-beta/espanso-debian-x11-amd64.deb
   # ... (gdebi install recommended)

   # Register espanso as a systemd service (required only once)
   $ espanso service register
   # Start espanso
   $ espanso start

   # 查看配置文档路径
   $ espanso path

Gist
----

管理代码块，配合Vscode GistPAD使用有奇效

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220514000206988.png" alt="image-20220514000206988" style="zoom: 50%;" />`

Typora
------


* 现已付费，无官方显式提供提供的deb包
* 无代码块功能

Recoll
------

全文检索工具

`安装 <https://www.lesbonscomptes.com/recoll/pages/download.html#ubuntu>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ sudo add-apt-repository ppa:recoll-backports/recoll-1.15-on
   $ sudo apt-get update
   $ sudo apt-get install recoll


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220513145807199.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220513145807199.png
   :alt: image-20220513145807199


`WPS <https://www.wps.cn/product/wpslinux>`_
------------------------------------------------

`Zeal <https://zealdocs.org/download.html>`_
------------------------------------------------

API管理工具

安装
^^^^

.. prompt:: bash $,# auto

   $ sudo apt install zeal


* `非官方文档CheatSheet <https://zealusercontributions.vercel.app/>`_

实战
^^^^

生成PCL docset
~~~~~~~~~~~~~~


* 生成doxygen文档时启动\ ``GENERATE_DOCSET``


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220201205322976.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220201205322976.png
   :alt: image-20220201205322976



* `生成docset <https://github.com/chinmaygarde/doxygen2docset>`_

.. prompt:: bash $,# auto

   $ doxygen2docset --doxygen ${HOME}/pcl-pcl-1.12.0/build/doc/doxygen/html/ --docset ${HOME}/pcl-pcl-1.12.0/doc/docset/


* 将生成的docset拷贝到zeal保存docset的位置

生成TensorRT docs
~~~~~~~~~~~~~~~~~

TensorRT的文档是直接提供了doxygen文档，而不像pcl docs一样可以编译生成docset。因此需要自己从html文件\ `生成docset文件 <https://kapeli.com/docsets#dashDocset>`_\ （步骤一：根据该教程构建相应的文件结构）。在已有html文件的基础上生成docset(\ `html->docset <https://github.com/selfboot/html2Dash>`_\ ).

.. prompt:: bash $,# auto

   # e.g. python html2dash.py -n <docsset_name> <src_dir>
   $ python html2dash.py -n tensorrt_docset tensorrt

----

**NOTE**


* 
  根据这种方法生成的docset虽然能够直接导入，但是没有classes, funcitons, types等，如下图。可自行添加(\ `Populate the SQLite Index <https://kapeli.com/docsets#dashDocset>`_\ )

  :raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220202215416986.png" alt="image-20220202215416986" style="zoom:67%;" />`

* 
  查看sqlite文件（如\ ``docSet.dsidx``\ ）

.. prompt:: bash $,# auto

   $ sudo apt install sqlitebrowser

生成rclcpp docset
~~~~~~~~~~~~~~~~~

.. prompt:: bash $,# auto

   # 步驟一：导入仓库
   $ git clone https://github.com/ros2/rclcpp
   # 步驟二：添加GENERATE_DOCSET = YES 到Doxyfile
   # 步骤三：生成doxygen docs
   $ doxygen Doxyfile
   # 步骤四：生成docset
   $ doxygen2docset --doxygen <src> --docset <dst>
   # 步骤五：将生成的docset拷贝到zeal保存docset的位置

常用可导入的docset
^^^^^^^^^^^^^^^^^^


* `ROS1 <https://github.com/beckerpascal/ros.org.docset>`_\ : 需下载和导入
* `pytorch cpp docs <https://github.com/pytorch/cppdocs>`_\ ：需下载、转换和导入

`Zotero <https://www.zotero.org/download/>`_
------------------------------------------------

文献管理工具

安装
^^^^


* 脚本安装

.. prompt:: bash $,# auto

   # 步骤一：解压后执行脚本 set_launcher_icon
   # 步骤二：添加软链接: e.g:
   $ ln -s /opt/zotero/zotero.desktop ~/.local/share/applications/zotero.desktop


* `apt安装 <https://github.com/retorquere/zotero-deb>`_

.. prompt:: bash $,# auto

   $ wget -qO- https://raw.githubusercontent.com/retorquere/zotero-deb/master/install.sh | sudo bash
   $ sudo apt update
   $ sudo apt install zotero # zotero-beta

   # 卸载
   $ wget -qO- https://apt.retorque.re/file/zotero-apt/uninstall.sh | sudo bash
   $ sudo apt-get purge zotero

插件
^^^^


* `Zotero Connector <https://chrome.google.com/webstore/detail/zotero-connector/ekhagklcjbdpajgpjgmbionohlpdbjgc/related>`_

安装方式：

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/go5s7zX8R3Aa0VEG.png!thumbnail" alt="img" style="zoom:67%;" />`


* `zotfile <http://zotfile.com>`_\ ：挪动文件夹
* `zotero-folder-import <https://github.com/retorquere/zotero-folder-import>`_ (暂不适用于0.6)
* `坚果云与zotera <https://help.jianguoyun.com/?p=3168>`_
* `Zotero PDF Translate <https://github.com/windingwind/zotero-pdf-translate>`_\ ：内置翻译
* `从SCI-HUB直接获取文献 <https://zhuanlan.zhihu.com/p/268375930>`_
* `Zotero Storage Scanner <https://github.com/retorquere/zotero-storage-scanner>`_\ ：移除无效或者重复的attachments (暂不适用于0.6)

实战
^^^^

导出中文引用
~~~~~~~~~~~~


* 添加中文引用

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/facUg6IFrhhiYcSW.png!thumbnail" alt="img" style="zoom:67%;" />`


* 插入文献引用到word文档

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/facUg6IFrhhiYcSW.png!thumbnail" alt="img" style="zoom:67%;" />`

知网导出国标引用
~~~~~~~~~~~~~~~~

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/fRrnPl2ntRl0cgIh.png!thumbnail" alt="img" style="zoom:80%;" />`
