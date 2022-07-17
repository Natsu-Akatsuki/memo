.. role:: raw-html-m2r(raw)
   :format: html


VSCode
======


.. raw:: html

   <p align="right">Author: kuzen, Natsu-Akatsuki</p>


配置文档
--------


* 配置文档\ ``setting.json``

.. prompt:: bash $,# auto

   # 触发保存时格式化
   "editor.formatOnSave": true

插件
----

格式化
^^^^^^

一般采用\ ``Ctrl+Shift+I``\ 进行触发


* shell-format
* cmake-format

.. prompt:: bash $,# auto

   # 使用时需要下载cmake_
   $ pip3 install cmake_format


* xml tools

bash, dockerfile, ignore
~~~~~~~~~~~~~~~~~~~~~~~~

c/c++
~~~~~

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/rwMxwrqpHdPxgLFX.png!thumbnail" alt="img" style="zoom: 67%; " />`


* `自定义配置 <https://blog.csdn.net/star871016/article/details/109526408>`_

括号高亮
^^^^^^^^

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210815144939462.png" alt="image-20210815144939462" style="zoom:50%; " />`

.. note::  `vscode 1.6版已经内置了括号高亮功能 <https://code.visualstudio.com/updates/v1_60#_editor>`_ ，在配置文档添加   ``"editor.bracketPairColorization.enabled": true`` 即可


代码块
^^^^^^


* 
  `snippetsmanager <https://github.com/zjffun/vscode-snippets-manager/tree/e2c865caee86944f2b0df7c70b18110c3c850621>`_\ ：代码块管理，更优化的可视化界面查看代码块

* 
  c/c++ snippets

代码运行快捷键
^^^^^^^^^^^^^^


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210822092237158.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210822092237158.png
   :alt: image-20210822092237158


代码补全
^^^^^^^^

如果用了gihub copilot的话，则可以不用下面的补全


* Tabnine
* `Kite AutoComplete <https://www.kite.com/linux/>`_

.. prompt:: bash $,# auto

   # 使用该插件前需要安装kite engine
   $ bash -c "$(wget -q -O - https://linux.kite.com/dls/linux/current)"
   $ systemctl --user start kite-autostart

查看API文档
^^^^^^^^^^^

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210919093438088.png" alt="image-20210919093438088" style="zoom:67%; " />`

对应的默认快捷键： ``ctrl+h`` , ``ctrl+alt+h``

生成doxygen文档
^^^^^^^^^^^^^^^


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211002021724564.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211002021724564.png
   :alt: image-20211002021724564


使用 ``/**`` 和回车键进行触发

使用文件模板
^^^^^^^^^^^^


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210927231828413.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210927231828413.png
   :alt: image-20210927231828413


快速编译和运行代码
^^^^^^^^^^^^^^^^^^

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210928115134786.png" alt="image-20210928115134786" style="zoom:67%; " />`


* 设置编译项

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210928115123099.png" alt="image-20210928115123099" style="zoom:67%; " />`

DEBUG
^^^^^

`cppcheck <https://cppcheck.sourceforge.io/>`_\ , `flawfinder <https://github.com/david-a-wheeler/flawfinder>`_\ , clang, flexelint...

.. prompt:: bash $,# auto

   # cppcheck
   $ sudo apt install cppcheck
   # flawfinder
   $ sudo pip3 install flawfinder

`markdown插件 <https://code.visualstudio.com/docs/languages/markdown>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint:: 官方教程包括了： `preview` , `outline` , `code snippet`


:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210901141059733.png" alt="image-20210901141059733" style="zoom: 67%; " />`


* 表格美化

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210901141231319.png" alt="image-20210901141231319" style="zoom:67%; " />`


* 格式化

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210901141415436.png" alt="image-20210901141415436" style="zoom:67%; " />`


* `链接粘贴（自动生成rst和markdown格式的超链接） <https://marketplace.visualstudio.com/items?itemName=kukushi.pasteurl>`_

等价于typora的超链接功能

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210907093539625.png" alt="image-20210907093539625" style="zoom:67%; " />`


* markdown lint(markdown文档规范化)

.. code-block:: json

   // 配置文档示例
   "markdownlint.config": {
       "MD013": {
           "code_blocks": false
       },
       "MD014": false,
       "MD033": false,
       "line-length": false,
       "no-inline-html": {
           "allowed_elements": [

           ]
       }
   }

其他
^^^^


* markdown math：给vscode中markdown添加数学支持（latex）
* remote ssh：远程连接
* ros：添加ros支持
* printcode：代码打印
* Code Spell Checker：拼写检查与修正
* Live Share: 实时协作写代码（微软官方插件）
* Live Share Audio：为Live Share开启语音交流（微软官方插件）
* meld diff：文本比对（支持粘贴版）

实战
----

配置文档
^^^^^^^^

本部分介绍vscode涉及的配置文档


* ``tasks.json``\ ：告诉编译器怎么构建程序
* ``launch.json``\ ：告诉GDB怎么启动Debug

`全局配置 <https://code.visualstudio.com/docs/getstarted/settings#_default-settings>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210828002103557.png" alt="image-20210828002103557" style="zoom:67%; " />`

`同步配置信息（配置文档、插件） <https://code.visualstudio.com/docs/editor/settings-sync>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`构建插件组(expansion pack) <https://code.visualstudio.com/blogs/2017/03/07/extension-pack-roundup>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

步骤一：\ `安装高版本的node.js <https://github.com/nodejs/help/wiki/Installation>`_\ （用apt安装的有如下警告和报错）

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/LjKQ3d57TAU133aE.png!thumbnail" alt="img" style="zoom:67%; " />`

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/uQkyGFoF25MOUDCs.png!thumbnail" alt="img" style="zoom:67%; " />`

.. hint:: 安装时不用像官网一样导入到系统路径


步骤二：安装 ``Yeoman generator``

.. prompt:: bash $,# auto

   $ npm install -g yo generator-code

步骤三：构建一个 ``expansion pack`` 文件夹

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210901234442008.png" alt="image-20210901234442008" style="zoom:67%; " />`

步骤四：安装 ``vsce`` 和在该文件夹下构建拓展插件

.. prompt:: bash $,# auto

   $ npm install -g vsce
   # vsce报错缺什么，package.json就加哪个字段的信息
   $ vsce package

.. hint:: 上述命令行执行时并非一步到位，此处省略了根据提示而进行操作的步骤；其中要vsce package构建成功需要合适的package.json文件；更多可参考 [github实例](https://github.com/robertoachar/vscode-extension-pack)


自定义Button来触发pandoc
^^^^^^^^^^^^^^^^^^^^^^^^


* `插件 <https://marketplace.visualstudio.com/items?itemName=seunlanlege.action-buttons>`_

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210905204620163.png" alt="image-20210905204620163" style="zoom:67%; " />`

pandoc
~~~~~~

.. hint:: 正如[此处](https://github.com/miyakogi/m2r)所述，在实际的测试中，pandoc对markdown->rst的转换效果一般，如容易丢失图片，不建议用pandoc进行转换



* 安装

.. prompt:: bash $,# auto

   $ wget -c https://github.com/jgm/pandoc/releases/download/2.14.2/pandoc-2.14.2-linux-amd64.tar.gz -O ~/application/pandoc-2.14.2-linux-amd64.tar.gz
   $ cd application && tar -xzvf pandoc-2.14.2-linux-amd64.tar.gz
   $ cd pandoc-2.14.2
   $ sudo ln -s $(pwd)/bin/pandoc /usr/local/bin
   $ sudo ln -s $(pwd)/share/man/man1/pandoc.1.gz /usr/share/man/man1
   # 添加自动补全
   $ echo 'eval "$(pandoc --bash-completion)"' >> ~/.bashrc

----

**NOTE**

pandoc常用选项：

.. list-table::
   :header-rows: 1

   * - option
     - 作用
   * - -f/--from
     - 指定输入格式
     - 
   * - -t/--to
     - 指定输出格式（若无指定格式则会根据文件名进行推导）


----


* 配置配置文件

.. code-block:: json

   "actionButtons": {
       "defaultColor": "#ff0034", // Can also use string color names.
       "loadNpmCommands": false, // Disables automatic generation of actions for npm commands.
       "reloadButton": "♻️", // 触发配置生效
       "commands": [
           {
               "cwd": "${workspaceFolder}", // cd workspace
               "name": "pandoc",            // terminal name/ tip name
               "color": "green",
               "singleInstance": true,
               "command": "pandoc -s -f markdown -t rst ${file} >> ${fileDirname}/${fileBasenameNoExtension}.rst", // This is executed in the terminal.
           }
       ]
   },


* 拓展资料

  * `支持的转换格式 <https://docs.onap.org/en/dublin/guides/onap-developer/how-to-use-docs/converting-formats.html#fixing-the-converted-document>`_
  * `用例 <https://pandoc.org/demos.html>`_

web端查看github代码
^^^^^^^^^^^^^^^^^^^

将 ``.com`` 改为 ``.dev``

设置滚轮速度
^^^^^^^^^^^^


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/CuLMsVboZB2jId0c.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/CuLMsVboZB2jId0c.png!thumbnail
   :alt: img


设置自动格式化
^^^^^^^^^^^^^^


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210928102245843.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210928102245843.png
   :alt: image-20210928102245843


`取消标签页的重用（取消preview模式） <https://code.visualstudio.com/docs/getstarted/userinterface#_preview-mode>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211002022616570.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211002022616570.png
   :alt: image-20211002022616570


根据代码长度调整显示
^^^^^^^^^^^^^^^^^^^^

View->Word Wrap (Alt+Z)

`代码块设置 <https://code.visualstudio.com/docs/editor/userdefinedsnippets#_creating-your-own-snippets>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  由ctrl+space触发

* 
  设置"editor.tabCompletion": "on"后可以按Tab触发snippet的插入

* 
  语法：\ `占位符 <https://code.visualstudio.com/docs/editor/userdefinedsnippets#_placeholders>`_

  :raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220509000034490.png" alt="image-20220509000034490" style="zoom: 67%;" />`

.. list-table::
   :header-rows: 1

   * - 作用
     - 语法
     - 案例
   * - 设置光标在snippet的最终位置
     - $0
     - 
   * - 选择项
     - \${num|optionA,optionB|} |   ${1|one,two,three|}
     - 
   * - 设置正则以修改变量
     - ${变量/正则表达式/替换的内容/option}


.. note:: 其中的option如g



* latex中给选定的文本添加颜色

.. code-block::

   ${color|red,green,blue}
   \textcolor{blue}{...} -> \\textcolor\{blue\}{...}



   \textcolor{.*}{(.*)} -> \\textcolor\{.*\}\{(.*)\}

   ${TM_SELECTED_TEXT/\\textcolor\{.*\}\{(.*)\}/{$1}}
   |one,two,three|}


   转义分为两部分：body内的，一个是正则内部的

DEBUG
^^^^^


* `org.freedesktop.secrets was not provided by any service <https://github.com/Foundry376/Mailspring/issues/681>`_

.. prompt:: bash $,# auto

   $ sudo apt install gnome-keyring

常用快捷键
----------

代码/文件间反复横跳(code navigation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 作用
     - 快捷键
   * - 括号跳转
     - ctrl+shift+ (i.e. ctrl+|)
   * - 打开最近工作空间
     - ctrl+r

