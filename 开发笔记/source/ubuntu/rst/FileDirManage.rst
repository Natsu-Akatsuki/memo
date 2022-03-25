.. role:: raw-html-m2r(raw)
   :format: html


FileDirManage
=============


.. raw:: html

   <p align="right">Author: kuzen, Natsu-Akatsuki</p>


`文件夹跳转 <https://github.com/gsamokovarov/jump>`_
--------------------------------------------------------

jump
^^^^

一般都是跳转到使用次数最多的文件夹


* 安装(for ubuntu)

.. prompt:: bash $,# auto

   $ wget https://github.com/gsamokovarov/jump/releases/download/v0.40.0/jump_0.40.0_amd64.deb && sudo dpkg -i jump_0.40.0_amd64.deb
   $ echo 'eval "$(jump shell)"' >> ~/.bashrc

打开文件夹
----------

.. prompt:: bash $,# auto

   $ xdg-open

查看文件
--------

more
^^^^

.. prompt:: bash $,# auto

   # 运行时可用按键
   # space：向下翻页
   # enter：向下翻行
   # /    ：向下查找
   # :f   ：显示文件名和目前显示的行数

batcat
^^^^^^

相当于带语法高亮的cat


* 安装

.. prompt:: bash $,# auto

   $ sudo apt install catbat


* 配置

.. prompt:: bash $,# auto

   $ echo 'alias bat="batcat --paging=auto"' >> ~/.bashrc
   $ echo "export MANPAGER=\"/bin/bash -c 'col -bx | batcat -l man -p'\"" >> ~/.bashrc

----

**REMARK**

给man手册上色的功能很赞，但不建议使用（其不像man一样独享一个窗口，会跟前面的终端输出混在一起），可以使用\ `most <https://www.cyberciti.biz/faq/unix-linux-color-man-pages-configuration/>`_

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210905011156100.png" alt="image-20210905011156100" style="zoom:67%; " />`

查找文件
--------

locate
^^^^^^

.. prompt:: bash $,# auto

   $ sudo apt install mlocate
   # 更新数据库
   $ sudo updatedb
   $ locate <文件名>

find
^^^^

.. prompt:: bash $,# auto

   # find [-H] [-L] [-P] [-Olevel] [-D debugopts] [path...] [expression]
   # -name: file name


* 文件类型：


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220323140153693.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220323140153693.png
   :alt: image-20220323140153693


du
^^

统计当前目录的文件夹大小

.. prompt:: bash $,# auto

   $ du -h --max-depth=1

ncdu
^^^^

效果同du，但有更人性化的可视化效果和计算速度

.. prompt:: bash $,# auto

   $ sudo apt install ncdu
   $ ncdu

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/fHcvhrY9Y6y3set8.png!thumbnail" alt="img" style="zoom:67%; " />`

查看说明文档
------------

help
^^^^

多用于查看可用选项和参数、子命令

.. prompt:: bash $,# auto

   $ <命令行> --help

man
^^^

多用于查看详细的说明文档

.. prompt:: bash $,# auto

   $ man <命令行>
   # man面板页的可用操作
   # ---跳转---
   # 空格：向下翻一页
   # page down：向下翻一页
   # page up：向上翻一页
   # home：跳转到首页
   # end：跳转后尾页

   # ---查找---
   # /string (enter)：向下查找    n/N：向下查找/向上查找
   # ?string (enter)：向上查找    n/N：向上查找/向下查找

解压缩
------

tar
^^^

.. prompt:: bash $,# auto

   # 压缩
   $ tar -czvf <生成的压缩文件名> <待压缩的文件/文件夹>
   # 解压缩
   $ tar -xzvf <解压的文件名> -C <待存放的文件夹路径>

.. list-table::
   :header-rows: 1

   * - option
     - 效果
   * - -C
     - 指定解压缩的目的文件夹
   * - -c
     - create a archive
   * - -f
     - 指定生成的压缩文件名/待解压的文件名
   * - -v
     - list verbose information
   * - -x
     - 解压
   * - -z
     - 使用gzip的支持进行压缩


`pigz <https://zlib.net/pigz/pigz.pdf>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

多线程解压缩

.. prompt:: bash $,# auto

   # 安装
   $ sudo apt install pigz
   # 压缩
   $ tar -cvf - <待压缩的文件> | pigz -p 8 > <输出的压缩文件名> 
   # -p：设置线程数 

   # 解压缩（分两步）p
   $ unpigz <待解压的文件名>
   $ tar -xf <待第二次解压的文件名.tar>

.. attention:: 注意命令行中有一个 ``>``


pv
^^

显示解或压缩时的进度条

.. prompt:: bash $,# auto

   # 压缩文件
   $ tar -cf - <待压缩文件或目录> | pv | gzip > <file.tar.gz
   # 解压缩
   $ pv <file.tar.gz> | tar -xzf -

软链接
------


* 文件软链接

.. prompt:: bash $,# auto

   $ ln -s <源地址> <目的地>
   # 可以不指定目的地，然后生成软链接到当前目录
   $ ln -s <源地址>

.. attention:: 所有地址都需要绝对路径（实际使用时常跟 ``$(pwd)`` 配合使用）



* 管理文件软链接

.. prompt:: bash $,# auto

   # e.g. manage gcc/g++
   $ sudo apt install gcc-9 g++-9 gcc-10 g++-10
   $ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10
   $ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9

   # 其他常用选项
   # --remove-all name: Remove all alternatives and all of their associated slave links. name # is a name in the alternatives directory.

   # 修改默认版本
   $ sudo update-alternatives --config gcc


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220115010416693.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220115010416693.png
   :alt: image-20220115010416693


.. note:: 指定slave和master的作用在于，master进行变动时，slave也会进行变动。比如gcc(master)从9.0切换到10.0时，g++(slave)也会从9.0切换到10.0


文件编辑
--------

vim
^^^

`常用快捷键 <https://vim.rtorr.com/lang/zh_cn>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* 命令行模式（Normal Mode）

.. list-table::
   :header-rows: 1

   * - 作用
     - 快捷键
   * - (navigation) 向右移动一格
     - space
   * - (navigation) 向某个方向挪动n格
     - [n]\ :raw-html-m2r:`<上向左右快捷键>`
   * - (navigation word-based) 向右/向左跳转
     - w/b | ctrl+left arrow/ctrl+right arrow
   * - (cut and paste) 复制当前行
     - yy(yank)
   * - (cut and paste) 复制光标所在处到行首/行尾
     - y0/y$
   * - (edit) 缩进
     - >> / <<
   * - (edit) 撤销
     - u (undo)
   * - (edit) 反撤销
     - ctrl+r
   * - （区块选择）字符选择，给光标经过的地方高亮显色
     - v
   * - （区块选择）以矩形的方式选择数据
     - ctrl+v
   * - （区块选择）复制/删除/粘贴反白的地方
     - y/d/p



* 末行模式

.. list-table::
   :header-rows: 1

   * - 作用
     - 命令行
   * - 删除指定行
     - :num, [num]d
   * - 执行bash命令
     - :!\ :raw-html-m2r:`<command>`
   * - 向上查找
     - ?
   * - 向下查找
     - /
   * - 替换某些行[n,m]的某些词
     - :n,ms/待替换的词/被替换的内容/g
   * - 替换前还要有交互（认证是否进行替换，末尾加c）
     - :n,ms/待替换的词/被替换的内容/gc


`配置文档 <https://blog.csdn.net/silence1772/article/details/81057702>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* `共享剪切板 <https://www.zhihu.com/question/19863631>`_

共享系统的剪切板与vim的剪切板（鼠标中键、p、 ``c+s+v`` ）

.. prompt:: bash $,# auto

   # 安装gvim插件，并在配置文档中添加：
   set clipboard=unnamedplus:s


* vim的全局参数配置文件放在 ``/etc/vim/vimrc`` 但不建议修改，而是建议修改 ``~/.vimrc``

高级封装
~~~~~~~~

`spacevim <https://spacevim.org/cn/quick-start-guide/>`_ 使用文档涉及，基本的配置文档设置、插件的更新

spacevim下的常用快捷键：

.. list-table::
   :header-rows: 1

   * - 作用
     - 快捷键
   * - 创建文件树面板
     - F3



* 拓展插件

  * `vim-plug <https://github.com/junegunn/vim-plug>`_

.. note:: vim可打开pcd文件和rosbag等文件，以快速查看文件头


:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210916223113843.png" alt="image-20210916223113843" style="zoom:50%; " />`

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210916223302201.png" alt="image-20210916223302201"  />`

去重行
^^^^^^

.. prompt:: bash $,# auto

   $ uniq
   # -i：忽略大小写
   # -c：进行计数

   # 案例：统计账号登录次数
   $ last | cut -d ' ' -f1 | sort | uniq -c

排序
^^^^

.. prompt:: bash $,# auto

   $ sort
   # -f: 忽略大小写
   # -b：忽略前导空格字符，ignore leading blanks
   # -t：分隔符号
   # -n：利用数字排序
   # -k：用第几个字段进行排序

   # 案例：对/etc/passwd的内容以账号id进行排序
   $ cat /etc/passwd | sort -t ':' -k 3 -n

diff
^^^^

记录文件改变的内容/不同的地方

.. prompt:: bash $,# auto

   $ diff A B

patch
^^^^^


* 
  `相关案例 <https://www.runoob.com/linux/linux-comm-patch.html>`_

* 
  根据文件的不同还原文件

.. prompt:: bash $,# auto

   # patch - apply a diff file to an original
   $ patch [options] [originalfile [patchfile]]
   e.g. patch -pnum patchfile

文本替换
--------

sed
^^^


* 语法

.. prompt:: bash $,# auto

   $ sed [OPTION] {script-only-if-no-other-script} [input-file]...

   {script-only-if-no-other-script}
   s/<正则表达式（待替换的内容）>/<替换的内容>/：使用正则表达式进行替换

.. note:: {script-only-if-no-other-script}这部分需要加引号 ``''``



* 实例

.. prompt:: bash $,# auto

   # 替换code-block为prompt-block
   $ m2r ${file} && sed -i -e 's/.. prompt:: bash $,# auto/.. prompt:: bash $,# auto/' ${fileDirname}/${fileBasenameNoExtension}.rst

   # 替换code-block为prompt-block
   # 去掉行首的第一个$ prompt
   $ m2r ${file}&& sed -i -e 's/.. prompt:: bash $,# auto/.. prompt:: bash $/' -e 's/$ //' ${fileDirname}/${fileBasenameNoExtension}.rst

更多可参考\ `此网站(linux hint) <https://linuxhint.com/50_sed_command_examples/>`_

----

**NOTE**


* s加g和不加g的区别：

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210906081613293.png" alt="image-20210906081613293" style="zoom:50%; " />`

----


* 常用选项

.. list-table::
   :header-rows: 1

   * - option
     - 效果
   * - -i
     - 原地替换 (in-place replacement)
   * - -e
     - command的拼接


perf
^^^^

.. prompt:: bash $,# auto

   $ perf 
   # option:
   -i:   edit files in place
   -e:   后接command
   # perl -i -ep "s/unstable/$(lsb_release -cs)/" changelog

内容提取
--------

awk
^^^

**逐行**\ 的\ **字段分割**\ （类型于python的str.split操作，默认分隔符为空格或Tab键）

.. prompt:: bash $,# auto

   # awk '条件类型1{操作1} 条件类型2{操作2} ...' 文件名
   # 查看/etc/passwd中第三字段小于10的数据，并且仅列出第一和第三字段
   $ cat /etc/passwd | awk 'BEGIN {FS=":"} $3<10 {print $1 "\t" $3}'

   # BEGIN    表示条件类型的效果从首行开始生效（否则是从第二行才开始生效）
   # FS=":"   指定分割符（FS：内置变量）

.. attention:: 注意使用的是单引号


cut
^^^

以字符为单位，截取字符

.. prompt:: bash $,# auto

   $ cut -c 2-3 截取第二到第三个字符
   $ cut -c 1-  截取第一个之后的字符（含第一个）

grep
^^^^

.. prompt:: bash $,# auto

   # grep [OPTION]... PATTERNS [FILE/DIR]...
   $ <...> | grep -A 3 "..." # 附带输出结果的后三行数据
   $ <...> | grep -B 3 "..." # 附带输出结果的前三行数据
   $ <...> | grep -C 3 "..." # 附带输出结果的前后三行数据
   # 递归查找某文件夹下含有该字符串的文件
   $ grep -rn <pattern> <file>
   # 去注释行和空格行
   $ cat <file_name> | grep -v ^# | grep -v ^$

   # 选项：
   # -o, --only-matching：只显示匹配的部分（一个匹配一行），不显示匹配行
   # -i：忽略大小写(ignore case distinctions)
   # -n：显示行数
   # -r：递归查找
   # -v：反选

xargs
^^^^^

读取标准输入作为某个命令的参数

.. prompt:: bash $,# auto

   $ xarg [option] cmd
   # 案例一：
   $ echo '--help' | xargs cat
   # 等价于 echo cat --help

   # 案例二：
   $ xargs rm < install_manifest.txt
   # -a <file_name> 以文件的内容为标准输入
   # -i cmd {}      用{}指代标准输入

获取文件信息
------------

.. prompt:: bash $,# auto

   $ dirname <absolute_file_name>
   $ basename <absolute_file_name>
   # -s <extension> 去后缀

   # 统计new line个数≈行数
   $ wc -

文件比对
--------

.. prompt:: bash $,# auto

   $ sudo apt install meld

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220123235004476.png" alt="image-20220123235004476" style="zoom:67%;" />`

拓展
----

正则表达式
^^^^^^^^^^

正则表达式是一个字符串匹配模板(pattern)，可进行字符查找和信息提取；支持正则表达式的命令行包括，vim, locate, find, sed, awk, grep

语法
~~~~


* postion match

.. list-table::
   :header-rows: 1

   * - pattern
     - function
   * - ^char
     - 匹配处在行首的char字符
   * - char$
     - 匹配处在行末为char字符



* word-level match

.. list-table::
   :header-rows: 1

   * - pattern
     - function
   * - .
     - 匹配任意一个字符（除换行符\n外），可匹配中文
   * - [...]
     - 匹配在字符集中的字符
   * - [^...]
     - 匹配不在字符集中的字符
   * - \d
     - 匹配数字
   * - \D
     - 匹配非数字
   * - \s
     - 匹配空白字符
   * - \S
     - 匹配非空白字符
   * - \w
     - 匹配英文、数字、下划线、中文字符



* string-level match

.. list-table::
   :header-rows: 1

   * - pattern
     - function
   * - (str)
     - 匹配字符串str
   * - (str1|str2)
     - 匹配字符串str1或str2



* continuous match

.. list-table::
   :header-rows: 1

   * - pattern
     - function
   * - char*
     - 重复零个或多个的前导符
   * - char+
     - 重复1个或多个的前导符
   * - char?
     - 重复0个或1个的前导符
   * - char{m, n}
     - 重复m-n个的前导符


.. note:: *, +, ? 称为限定符或数量符


拓展资料
~~~~~~~~


* `练习教程 <https://regexone.com/lesson/introduction_abcs>`_
* `说明文档 <https://deerchao.cn/tutorials/regex/regex.htm>`_
