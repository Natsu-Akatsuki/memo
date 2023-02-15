.. role:: raw-html-m2r(raw)
   :format: html


Package
=======

Apt
---


* 从使用的角度来说，封装程度：aptitude > apt > apt-get

常用命令行
^^^^^^^^^^


* apt

.. prompt:: bash $,# auto

   $ apt clean              # 清除安装包
   $ apt remove <pkg_name>  # 卸载软件，保留配置文件
   $ apt purge <pkg_name>   # 卸载软件和相关的配置文件
   $ apt autoremove            # 卸载已无用和自动安装的软件
   $ apt-mark hold <pkg_name> # 将某些包设置为手动更新

   # 通过升级安装包和删除部分受影响的依赖解决：The following packages have been kept back
   $ apt dist-upgrade


* dpkg

.. prompt:: bash $,# auto

   $ dpkg -i <deb_package>     # 安装包
   # -r: remove
   # -P: purge（此处为大写）
   # 也可以使用gdebi（需安装），其能更好的解决依赖问题
   $ gdebi <deb_package>  # 安装


* apt的加强版

.. prompt:: bash $,# auto

   # 安装
   $ sudo apt install aptitude
   # 专门解决A依赖于B，但B已有其他版本的问题（会触发降级）
   $ sudo aptitude install package-name
   # 查看可用的安装包版本
   $ aptitude versions <package>

查看apt包的相关信息
^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   # 除了显示依赖信息还会显示package的其他信息（如maintainer,recommends packages）
   $ apt show <package_name>
   # 仅显示依赖信息
   $ apt-cache depends <package_name>

显示deb包的相关信息
^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   # Show information about a package
   $ dpkg -I <archive/deb>

显示apt包的依赖链(apt dependency chain)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ sudo apt install apt-rdepends
   $ apt-rdepends -p <package_name>
   # -p: 在依赖包后面追加状态信息
   # -r：显示哪些package依赖当前的package

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210906141516268.png" alt="image-20210906141516268" style="zoom:67%; " />`

----

**NOTE**

要显示哪些包未安装，虽然该命令行有option可配置，但是使用上感觉如下命令行更方便

.. prompt:: bash $,# auto

   $ apt-rdepends -p <package_name> | grep NotInstalled

----

显示已安装的包
^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ apt list --install
   $ dpkg -l

删除无用的配置文档
^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ dpkg -l | grep "^rc" | awk '{print $2}' | sudo xargs apt -y purge

`增删PPA仓库 <https://linuxconfig.org/how-to-list-and-remove-ppa-repository-on-ubuntu-18-04-bionic-beaver-linux>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

实战
^^^^

`apt update失败 <https://askubuntu.com/questions/1095266/apt-get-update-failed-because-certificate-verification-failed-because-handshake>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* updates for this repository will not be applied：使用apt更新源时会出现如上报错，或同步下系统时间即可

PIP
---

常用命令行
^^^^^^^^^^

.. prompt:: bash $,# auto

   # >>> 安装pip >>>
   # 推荐将pip安装到用户目录
   $ python3 -m pip install -U pip
   # 以下为安装到系统目录
   $ sudo apt install python3-pip

   # ---下载--- #
   $ pip install --upgrade / -U <pkg_name>  # 升级给定package
   $ pip install -r <requirements.txt>      # 下载文档中给定的依赖
   $ pip install -i <某源>                  # 通过给定源进行下载
   $ pip install --no-cache-dir             # 不保留缓存地安装
   # ---查看包信息--- #
   $ pip show <pkg_name>
   $ pip list --outdate     # 查看可升级的包
   # ---pip安装到当前用户--- #
   $ pip install --user <pkg_name>
   # ---清除pip缓存--- #
   $ rm -r ~/.cache/pip
   # ---卸载包及其依赖--- #
   # pip install pip-autoremove
   $ pip-autoremove <pkg_name>

.. attention:: pip没有一键升级所有安装包的命令行，感觉是因为他不能够解决python包的依赖问题


.. note:: pip的配置文件存放于 ``~/.config/pip``


Pkg-config
----------


* .pc文件存储了包的元数据（包的库/头文件安装位置等信息）

.. prompt:: bash $,# auto

   # 查看系统的安装包
   $ pkg-config --list-all | grep opencv
   # 查看安装包的版本
   $ pkg-config --modversion opencv4
   $ more /usr/lib/x86_64-linux-gnu/pkgconfig/opencv4.pc

Wget
----

.. prompt:: bash $,# auto

   $ wget -c <链接> -O <file_name>
   # -c: 断点下载
   # -O：重命名
   # -P：下载对应的指定文件夹

.. hint:: aria2据说为增强版wget


Curl
----

.. prompt:: bash $,# auto

   $ curl
   # -k, --insecure      Allow insecure server connections when using SSL
   # -i, --include       Include protocol response headers in the output
   # -s, --silent        Silent mode
   # -L, --location      Follow redirects (配合tee重定向输出数据到文件)
   # --output <file>     Write to file instead of stdout


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211101171306726.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211101171306726.png
   :alt: image-20211101171306726


Snap
----

unix-like自带，安装的应用程序有点像docker容器，整体体积会较大

常用命令行
^^^^^^^^^^

.. prompt:: bash $,# auto

   $ snap list                           # 列出已安装的snap包
   $ sudo snap remove <pkg>              # 卸载snap中安装的包
   $ sudo apt autoremove --purge snapd   # 卸载snap-core

Conda
-----

安装和升级
^^^^^^^^^^

步骤一：\ `下载安装包(anaconda3) <https://www.anaconda.com/products/individual>`_\ ，\ `miniconda3 <https://conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_

.. prompt:: bash $,# auto

   # 完整版anaconda3
   $ https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ./anaconda.sh
   # 执行脚本
   $ conda update conda

   # miniconda3
   $ wget -c https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh
   # 执行脚本
   (base) $ conda update conda

步骤二：交互模式执行安装包（此方法可顺带初始化conda）

----

**NOTE**\ ：无交互式的安装

.. prompt:: bash $,# auto

   $ /bin/bash anaconda.sh -b -p /opt/conda 
   $ 'export PATH=/opt/conda/bin:$PATH' >> ~/.bashrc 
   $ conda init 
   $ conda config --set auto_activate_base false $
   $ conda update conda

   # -b run install in batch mode (without manual intervention), it is expected the license terms are agreed upon
   # -p PREFIX  install prefix, defaults to $PREFIX, must not contain spaces.

----

`卸载 <https://docs.anaconda.com/anaconda/install/uninstall/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   (base) $ conda install anaconda-clean
   (base) $ anaconda-clean
   $ rm -rf ~/anaconda3

配置文档
^^^^^^^^


* 默认不启动conda环境

.. prompt:: bash $,# auto

   $ conda config --set auto_activate_base false


* channel的解读：

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/7VIzKuXudONhw3oP.png!thumbnail" alt="img" style="zoom:50%;" />`


* conda install 时不指定channel (-c url/channel_name)时，则默认用defaults中的源
* 要重设defaults中的源，可利用字段 default_channels进行替换
* 在安装指定channel(即加上了 -c )，且在custom_channels中定义了channel_name这个key的value时，则channel_name会被替换为value

.. prompt:: bash $,# auto

   $ conda install -c pytorch <package_name>
   # 例如如上命令行将转换为：
   $ conda install -c https://mirrors.gdut.edu.cn/anaconda/cloud/pytorch <package_name>


* 如果channel_name不在 ``custom_channels`` 字段的 ``key`` 时，则channel_name被替换为channel_alias/channel_name  

查询信息
^^^^^^^^


* 查询当前环境的所有packages的相关信息

.. prompt:: bash $,# auto

   $ conda list
   # -n <env>: 指定环境


* 查询当前已安装的conda环境

.. prompt:: bash $,# auto

   $ conda env list


* 查询安装历史

.. prompt:: bash $,# auto

   $ conda list --revisions

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/I1JHF95b6IDEWj7M.png!thumbnail" alt="img" style="zoom:67%; " />`


* 查询conda应用程序的相关信息

.. prompt:: bash $,# auto

   $ conda info

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210906223711162.png" alt="image-20210906223711162" style="zoom: 50%; " />`

安装和更新包
^^^^^^^^^^^^

.. prompt:: bash $,# auto

   # 根据文件更新当前环境
   $ conda env update -f <文件名>
   # 跳过interaction进行安装
   $ conda install -y
   # 包的导出和导入
   $ conda env export -n 环境名 > 文件名.yml
   $ conda env create -f 文件名.yml

----

**NOTE**

文件解析：

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/XAWWBAeAbYBXrJRM.png!thumbnail" alt="img" style="zoom:67%; " />`

----

清理
^^^^

.. prompt:: bash $,# auto

   # 删除缓存、索引等
   $ conda clean -a
   # 删除环境
   $ conda env remove -n <env_name>
   # 删除包
   $ conda remove -n <env_name> <pkg>

.. note:: 注意conda使用的是remove而不是install（该命令能够根据依赖关系删包）


触发命令行补全
^^^^^^^^^^^^^^

conda并不提供内部补全的插件，需要\ `安装第三方插件 <https://github.com/tartansandal/conda-bash-completion>`_

步骤一：安装

.. prompt:: bash $,# auto

   $ conda install -n base -c conda-forge conda-bash-completion

步骤二：添加到~/.bashrc

.. prompt:: bash $,# auto

   # 配置conda代码补全
   CONDA_ROOT="${HOME}/anaconda3"
   if [[ -r $CONDA_ROOT/etc/profile.d/bash_completion.sh ]]; then
       source $CONDA_ROOT/etc/profile.d/bash_completion.sh
   fi

.. attention:: 记得修改对应的目录


通道设置
^^^^^^^^

.. prompt:: bash $,# auto

   # 查看通道
   $ conda config --show channels
   # 添加conda-forge作为通道
   $ conda config --add channels conda-forge
   # 安装时指定特定通道
   $ conda install -n base --override-channels -c conda-forge mamba=0.23.1

环境复制
^^^^^^^^


* 本地环境的复制

.. prompt:: bash $,# auto

   $ conda create --clone <被复制的环境> -n <粘贴的环境名>

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/jOxAQgSIQCmervG3.png!thumbnail" alt="img" style="zoom:67%; " />`


* `同操作环境下环境的迁移或部署 <https://conda.github.io/conda-pack/>`_\ （\ `中文翻译 <https://zhuanlan.zhihu.com/p/87344422>`_\ ）

.. prompt:: bash $,# auto

   # base环境下安装 
   $ conda install conda-pack 
   # src机上打包指定环境 
   $ conda pack -n <环境名> 
   # dst机上解压缩（tar...），解压缩到env目录下 
   $ ... 
   # 修复python package前缀项(conda-unpack在bin目录下) 
   $ conda activate <环境名>  && conda-unpack

.. hint:: 虽然conda pack最终的效果是生成一个压缩包，但跟自己用tar生成的压缩包不同，其还在压缩时添加了一些用于解决导出的python包路径错误问 的脚本，如conda-unpack。


版本回退
^^^^^^^^

.. prompt:: bash $,# auto

   # 查看已有的版本
   $ conda list --revision
   # 回退
   $ conda install --rev <revision number>

mamba
^^^^^

多线程提高下载速度
~~~~~~~~~~~~~~~~~~

用\ `mamba <https://github.com/mamba-org/mamba>`_\ 来安装包，建议基础环境使用miniconda，否则安装时要花很长的时间检查的依赖

.. prompt:: bash $,# auto

   $ conda install -n base --override-channels -c conda-forge mamba=0.24.0
   $ mamba install <package_name>

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/CP0aVRAsWIAQWpl3.png!thumbnail" alt="img" style="zoom:50%; " />`

mamba退出码异常无显示
~~~~~~~~~~~~~~~~~~~~~

尝试安装更高级的版本或者重新安装

实战
^^^^

`多用户下conda的配置 <https://docs.anaconda.com/anaconda/install/multi-user/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

包冲突问题
~~~~~~~~~~

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220410110813587.png" alt="image-20220410110813587" style="zoom:67%;" />`

卸载有冲突的包

.. prompt:: bash $,# auto

   $ conda uninstall liblapack liblapacke libcblas libblas

conda / pip install的区别？
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/Sg1aq9YrmHbnGorp.png!thumbnail" alt="img" style="zoom:50%;" />`


* 
  不同的存放位置

  pip 存放在 anaconda/env/相应的目录中，不可被其他虚拟环境的复用；

  conda 的包则存放在 /pkgs中可被其他conda环境复用，避免再进行一次下载

base环境下没有pip
^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   # 查看base环境的pip，发现其使用是系统的
   (base) $ which pip
   # /usr/bin/pip
   # 安装pip到conda base环境
   (base) $ conda install pip

拓展资料
^^^^^^^^


* `conda 说明文档 <https://docs.conda.io/projects/conda/en/latest/user-guide/>`_
* `参数配置文档1 <https://conda.io/projects/conda/en/latest/user-guide/configuration/index.html>`_\ 、\ `参数配置文档2 <https://conda.io/projects/conda/en/latest/configuration.html?highlight=custom_channels%3A>`_
* `任务导向说明 <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/index.html>`_
* `conda-vs-pip-vs-virtualenv-commands <https://docs.conda.io/projects/conda/en/latest/commands.html#conda-vs-pip-vs-virtualenv-commands>`_

`PPA <https://launchpad.net/ubuntu/+ppas>`_
-----------------------------------------------

`添加PPA到PC <https://help.launchpad.net/Packaging/PPA/InstallingSoftware>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   # sudo add-apt-repository ppa:user/ppa-name
   $ sudo add-apt-repository ppa:natsu-akatsuki/sleipnir

.. note:: 本质上是往 ``/etc/apt/sources.list.d`` 中添加source.list


`创建PPA <https://help.launchpad.net/Packaging/PPA>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Activating a PPA

打包一个文件到PPA
^^^^^^^^^^^^^^^^^

步骤一：\ `上传GPG到ubuntu server <https://help.ubuntu.com/community/GnuPrivacyGuardHowto>`_\ ，以让所有客户端可获取

.. prompt:: bash $,# auto

   # gpg --keyserver keyserver.ubuntu.com --send-keys <yourkeyID>
   $ gpg --keyserver keyserver.ubuntu.com --send-keys 96037357E6D61138
   # 查看是否上传成功
   $ gpg --keyserver hkp://keyserver.ubuntu.com --search-key <yourkeyID>

步骤二：\ `launchpad中添加GPG密钥 <https://launchpad.net/+help-registry/import-pgp-key.html>`_


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220125003446149.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220125003446149.png
   :alt: image-20220125003446149


步骤三：生成template

.. prompt:: bash $,# auto

   # cd到待打包的文件中
   $ dh_make --createorig -s -y
   $ dh_make -p tutorial_0.0.1 --single --native --copyright mit --email hong877381@gmail.com
   # optioin:
   # -y, --yes             automatic yes to prompts and run non-interactively
   # -s, --single          set package class to single
   # -i, --indep           set package class to arch-independent
   # -l, --library         set package class to library
   # --python              set package class to python
   # --createorig
   $ rm debian/*.ex debian/*.EX   # 删除不需要的文件

.. note:: For dh_make to find the package name and version, the current directory needs to be in the format of <package>-<version>. Alternatively use the_-p flag using the format <name>_<version> to override it. The directory name you have specified is invalid!



* 其中主要是要完善changelog、copyright、control文件

----

**ATTENTION**


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220125105404202.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220125105404202.png
   :alt: image-20220125105404202


----

.. prompt:: bash $,# auto

   $ perl -i -0777 -pe "s/(Copyright: ).+\n +.+/\${1}$(date +%Y) natsu-akatsiku Foo <hong877381@gmail.com>/" copyright

步骤四：构建deb包


* 填写完成后即进行打包和sign

.. prompt:: bash $,# auto

   $ sudo apt-get install devscripts build-essential lintian
   # 等价于：cd到待打包的目录，构建deb包
   $ dpkg-buildpackage -us -uc
   # option:
   # -us, --unsigned-source      unsigned source package
   # -uc, --unsigned-changes     unsigned .buildinfo and .changes file.

   # sign .changes file（会同时把dsc, buildinfo也sign了）
   $ debsign -k <keyID> <filename>.changes

   # 要一体化dpkg-buildpackage和debsign命令则可以使用debuild命令
   # 打包和sign文件，注意k后无空格
   $ debuild -k<keyID> -S

步骤五：\ `上传文件到PPA <https://help.launchpad.net/Packaging/PPA/Uploading>`_

.. prompt:: bash $,# auto

   $ sudo apt install dput
   # dput ppa:your-lp-id/ppa <source.changes>
   $ dput ppa:natsu-akatsuki/sleipnir <source.changes>

.. note:: 可查看绑定邮件看上传情况


Q&A
~~~


* （已设置GPG的情况下）package上传成功后，不会很快生效，需要等一段时间。

.. code-block::

    Failed to add key. helios@helios:**~**$ sudo add-apt-repository ppa:natsu-akatsuki/sleipnir. More info: <https://launchpad.net/~natsu-akatsuki/+archive/ubuntu/sleipnir>. Press [ENTER] to continue or Ctrl-c to cancel adding it. Error: signing key fingerprint does not exist. Failed to add key.


* `上传失败 <https://help.launchpad.net/Packaging/UploadErrors>`_

参考资料
^^^^^^^^


* `ppa-guide之十万个为什么 <https://itsfoss.com/ppa-guide/>`_
* `利用debuild整合版工具来构建deb包 <https://blog.packagecloud.io/buildling-debian-packages-with-debuild/>`_
* `debian目录的相关描述 <https://packaging.ubuntu.com/html/debian-dir-overview.html>`_

Auto Upgrade
------------

关闭gnome的软件更新自启动
^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ sudo rm /etc/xdg/autostart/update-notifier.desktop

unattended-upgrade
^^^^^^^^^^^^^^^^^^


* 配置文档：/etc/apt/apt.conf.d/20auto-upgrades

.. prompt:: bash $,# auto

   // Enable the update/upgrade script (0=disable)
   APT::Periodic::Enable "1";

   // Do "apt-get update" automatically every n-days (0=disable)
   APT::Periodic::Update-Package-Lists "1";

   // Do "apt-get upgrade --download-only" every n-days (0=disable)
   APT::Periodic::Download-Upgradeable-Packages "1";

   // Run the "unattended-upgrade" security upgrade script
   // every n-days (0=disabled)
   // Requires the package "unattended-upgrades" and will write
   // a log in /var/log/unattended-upgrades
   APT::Periodic::Unattended-Upgrade "1";
