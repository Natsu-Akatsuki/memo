.. role:: raw-html-m2r(raw)
   :format: html


Terminal
========

Application
-----------

`Konsole <https://ambook.readthedocs.io/zh/latest/Ubuntu/rst/AppearanceManage.html#konsole>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

自从用了konsole的分屏后就很少用tmux了

.. prompt:: bash $,# auto

   $ sudo apt install konsole

`Tmux <https://manpages.ubuntu.com/manpages/focal/en/man1/tmux.1.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

安装
~~~~

.. prompt:: bash $,# auto

   $ sudo apt install -y tmux

配置文档
~~~~~~~~

触发配置文档生效

.. prompt:: bash $,# auto

   $ tmux source ~/.tmux.conf`

.. note:: 配置文档所在位置为 `~/.tmux.conf`


分屏
~~~~


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210902091648903.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210902091648903.png
   :alt: image-20210902091648903


.. prompt:: bash $,# auto

   # 分屏得四宫格(split-window alias:split)
   $ tmux new -s ros
   $ tmux split -v
   $ tmux split -h
   $ tmux select-pane -t 1
   $ tmux split -h

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210902094739706.png" alt="image-20210902094739706" style="zoom:67%; " />`

pane操作
~~~~~~~~


* 显示panes

.. prompt:: bash $,# auto

   $ tmux list-panes

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210902092500394.png" alt="image-20210902092500394" style="zoom:67%; " />`


* panes切换

.. prompt:: bash $,# auto

   $ tmux select-pane <-t pane_id>
   # pane id可通过display-panes来知悉

session操作
~~~~~~~~~~~


* 创建session


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210902093923093.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210902093923093.png
   :alt: image-20210902093923093


常用快捷键
~~~~~~~~~~

该部分快捷键包含自定义的快捷键

.. list-table::
   :header-rows: 1

   * - 作用
     - 快捷键
   * - 分屏：vertical split
     - 前导符+-
   * - 分屏：horizon split
     - 前导符+|
   * - 分屏：panel switch
     - 前导符+o
   * - 游走(navigation)：panel/window 选择性地切换
     - 前导符+w
   * - 粘贴版：显示粘贴板
     - 前导符+w
   * - 粘贴板：粘贴
     - 前导符+p
   * - 粘贴板：选择性粘贴
     - 前导符+P


实战
~~~~


* 
  `自定义配置 <https://github.com/Natsu-Akatsuki/MyTmux>`_

* 
  `复制pane的文字 <https://blog.csdn.net/RobertFlame/article/details/92794332>`_

需要在使用前使用 ``shift``


* 安装tpm

安装后，后续可用前导符+I（大写）进行插件安装

.. prompt:: bash $,# auto

   $ git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm


* 面板缩放(zoom)： ``前导符+z``

Terminator
^^^^^^^^^^


* 安装与使用

.. prompt:: bash $,# auto

   $ sudo apt install terminator
   $ terminator


* `常用快捷键 <https://blog.csdn.net/zhangkzz/article/details/90524066>`_

Screen
^^^^^^

略

`Yakuake <https://github.com/KDE/yakuake>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

顶部终端，暂感觉用处不大（2022.06.04）

`NNN <https://github.com/jarun/nnn>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

基于终端的文件管理

显示icon
~~~~~~~~

需要源码编译才能支持该功能

步骤一：安装\ `icons-in-terminal <https://github.com/sebastiencs/icons-in-terminal#bash-integration>`_

.. prompt:: bash $,# auto

   $ git clone https://github.com/sebastiencs/icons-in-terminal.git
   $ ./install.sh  
   $ # Follow the instructions to edit ~/.config/fontconfig/conf.d/30-icons.conf

步骤二：\ `源码编译nnn <https://github.com/jarun/nnn/wiki/Advanced-use-cases#file-icons>`_

.. prompt:: bash $,# auto

   # 安装相关依赖
   $ sudo apt install pkg-config libncursesw5-dev libreadline-dev
   $ git clone https://github.com/jarun/nnn
   $ cd nnn
   $ sudo make O_ICONS=1

----

**NOTE**


* nnn不是所有版本都有-S(du)的功能
* 其效果有点像\ ``spacevim``\ ，学习曲线较长，暂没从中提高过什么效率


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/oCtqAxAiA9SZmIAd.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/oCtqAxAiA9SZmIAd.png!thumbnail
   :alt: img


----

Appearance
----------

cowsay
^^^^^^

.. prompt:: bash $,# auto

   $ apt install cowsay
   $ cowsay <...文本>

echo
^^^^


* echo颜色

.. prompt:: bash $,# auto

   $ echo -e "\e[32mComplete \e[0m"
   $ \e 等价于 \033
   $ echo -e "\033[32mComplete \033[0m"

figlet
^^^^^^


* 字体符号化

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/QutNVaj257Fg5yrN.png!thumbnail" alt="img" style="zoom:67%;" />`

Shell
-----

首行配置
^^^^^^^^

.. prompt:: bash $,# auto

   # e.g. 用于指明执行当前脚本的执行器
   #!/bin/bash

特殊参数
^^^^^^^^

.. prompt:: bash $,# auto

   $$：查看当前终端的pid 
   $1：取命令行的第1个参数（序号从0开始） 
   ${@:2} ：取所有的参数，取从第2个开始的所有参数 
   $? ：获取上一个命令行返回的exit code
   `

`配置特殊的终端效果 <https://www.cnblogs.com/robinunix/p/11635560.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

一般用在bash脚本中，该选项对应于 ``bash [option]``

.. prompt:: bash $,# auto

   # 启动调试模式，输出详细的日志（会标准输出当前执行的命令）
   $ set -x
   # 若脚本执行有问题，则直接退出脚本
   $ set -e

输入输出流重定向
^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   # 1>     标准输出重定向 (dafault)
   # 2>     标准输出错误重定向 
   # 1>&2   标准输出转换为标准输出错误（放置在命令行末尾） 
   # 2>&1   标准输出错误转换为标准输出   （放置在命令行末尾）

   $ echo "hello" 2> /dev/null

`read函数 <https://linuxcommand.org/lc3_man_pages/readh.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ read -r -p "Are You Sure? [Y/n] " input 
   $ read -p "Remove all RealSense cameras attached. Hit any key when ready"
   # -p：输入时显示提示信息
   # -r: 不支持字符串转义 do not allow backslashes to escape any characters（支持直接接收回车键）
   `

`自定义函数 <https://blog.csdn.net/bornfree5511/article/details/109091233>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

bash option
^^^^^^^^^^^


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/O3qeGIlZbro6Cifs.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/O3qeGIlZbro6Cifs.png!thumbnail
   :alt: img


.. prompt:: bash $,# auto

   # -i：启动交互式的脚本（若没显式制定-i，bash会根据代码是否有IO交互，隐式加上 -i ）
   # -v：执行脚本前，先显示脚本内容
   # -x：显示正在执行的命令行(commands)和其参数(arguments)
   # -e：若有一个命令行返回值为非0则退出(end)脚本

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/vc2ZAhmLzlmLH17y.png!thumbnail" alt="img" style="zoom:67%; " />`

`判断语法 <https://www.cnblogs.com/mlfz/p/11427760.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 使用方括号作为shell的判断式

.. prompt:: bash $,# auto

   # 判断变量是否非空
   temp="..."
   [ -z "$temp" ] 单对中括号变量必须要加双引号
   [[ -z $temp ]] 双对括号，变量不用加双引号

   # 一般配合if语法使用
   # if [...]
   # then
   # fi  

   # 常用：
   # -d: 文件夹存在

   # Get the linux kernel and change into source tree
   if [ ! -d ${kernel_name} ]; then
    mkdir ${kernel_name}
    cd ${kernel_name}
    git init
    git remote add origin git://kernel.ubuntu.com/ubuntu/ubuntu-${ubuntu_codename}.git
    cd ..
   fi

.. attention:: 注意空格，[空格... 空格]



* 拓展资料：


#. 
   `方括号的等价含义 <https://unix.stackexchange.com/questions/99185/what-do-square-brackets-mean-without-the-if-on-the-left>`_

#. 
   `test command <https://linuxhint.com/bash-test-command/>`_ (or man test)

`for语法 <https://blog.csdn.net/guodongxiaren/article/details/41911437>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   #!/bin/bash
   ans=0
   for i in {1..100}; do
       let ans+=$i
   done
   echo $ans

Shortcut
--------


* 快捷键：

.. list-table::
   :header-rows: 1

   * - 快捷键
     - 作用
   * - ctrl+w
     - 删除一个单词
   * - ctrl+7 / ctrl+8
     - 撤销操作(cancel) / 取消撤销
   * - ctrl+u
     - 剪切至开头
   * - ctrl+k
     - 剪切至末尾
   * - ctrl+y
     - 粘贴
   * - ctrl+←/ctrl+→
     - 以单词为单位进行左右跳转
   * - ctrl+#
     - 注释当前命令行


.. attention:: 此处快捷键的剪切板并不是系统的剪切板



* 拓展资料：\ `终端的艺术 <https://github.com/jlevy/the-art-of-command-line/blob/master/README-zh.md>`_

SpecialInfo
-----------


* 想要在输入密码时，有提示信息，可修改\ ``/etc/ssh/sshd_config``\ 的\ ``Banner``\ 字段
* 想要在登录界面中，添加提示信息，可

.. prompt:: bash $,# auto

   $ sudo apt install landscape-common
   # 添加bash文件到/etc/update-motd.d/，其中文件顺序从小到大进行执行

TTY
---


* 
  界面分为 ``命令行界面`` 和 ``图像化界面``

* 
  ``命令行界面`` ，又称为 ``终端界面``\ ，对应的tty为 ``ttyX``

* 
  ubuntu下默认提供6个 ``终端界面`` 给用户登录，每个终端界面下启动的 ``terminal`` 对应的tty为 ``pts/X``


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/e2wbM5698Gcp7CcW.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/e2wbM5698Gcp7CcW.png!thumbnail
   :alt: img


查看某些按键的特殊效果
^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ stty -a

切换界面
^^^^^^^^

.. prompt:: bash $,# auto

   # 查询当前默认的界面（命令行界面or终端界面）
   $ systemctl get-default
   # 切换界面(依次为命令行界面和终端界面)
   $ systemctl isolate multi-user.target
   $ systemctl isolate graphical.target
   # 设置默认界面
   $ systemctl set-default graphical.target

注销
^^^^


* `for KDE <https://fostips.com/log-out-command-linux-desktops/>`_

.. prompt:: bash $,# auto

   $ qdbus org.kde.ksmserver /KSMServer logout 1 0 3
   # 重定向
   $ alias logout="qdbus org.kde.ksmserver /KSMServer logout 1 0 3"

TUI
---

`Textual <https://github.com/Textualize/textual>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

python模块，暂时没感觉到适用的地方（2022.6.5）

Dialog
^^^^^^


* `Cody的探索日记 <https://codychen.me/2020/29/linux-shell-%E7%9A%84%E5%9C%96%E5%BD%A2%E4%BA%92%E5%8B%95%E5%BC%8F%E4%BB%8B%E9%9D%A2-dialog/>`_
* `Sleipnir.Setup的工程 <https://github.com/GDUT-IIDCC/Sleipnir.setup/blob/ubuntu20/Setup.sh>`_

Extension
---------

`hstr <https://github.com/dvorka/hstr>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

命令行补全工具

.. prompt:: bash $,# auto

   $ sudo add-apt-repository ppa:ultradvorka/ppa && sudo apt-get update && sudo apt-get install hstr && hstr --show-configuration >> ~/.bashrc && . ~/.bashrc

script
^^^^^^

命令行录制工具

.. prompt:: bash $,# auto

   $ script <output_file_name>
   # 命令行操作
   # 结束操作
   $ exit

`history <https://zhuanlan.zhihu.com/p/248520994>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 默认存放数据的文件为 ``~/.bash_history``


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/w3AkpBGZgJwA4SJZ.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/w3AkpBGZgJwA4SJZ.png
   :alt: img



* 
  使用history时，显示的是命令行 ``历史列表`` 的内容。此处的 ``历史列表`` 即 ``当前终端执行过的命令`` +读取 ``~/.bash_history`` 得到的历史记录（默认打开终端时读取一次）

* 
  只有终端 ``logout`` 后才会将终端输入过的命令行加入到 ``~/.bash_history`` 中

* 
  如果不需要等终端 ``logout(ctrl+d/exit)`` 后才将命令行写入文件中，使得新开一个终端按history就能看到所有终端执行过的命令行），可添加该行到 ``~/.bashrc`` ；安装了 :ref:`hstr`. 的话，该部分会自动添加。

.. prompt:: bash $,# auto

   # 设置每执行完一个指令后的操作，以下的作用为即时刷新文件内容和更新历史列表
   export PROMPT_COMMAND="history -a; history -n; ${PROMPT_COMMAND}"
   # option:
   a：（写）将历史列表中相对于文件增加的命令行 追加到文件中
   n：（读）将文件中相对于历史列表增加的命令行 追加到终端的历史列表中

Q&A
---

`隐藏Qt警告 <https://www.reddit.com/r/kde/comments/asseoc/how_to_hide_qfilesystemwatcherremovepaths_list_is/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ export QT_LOGGING_RULES='*=false'
