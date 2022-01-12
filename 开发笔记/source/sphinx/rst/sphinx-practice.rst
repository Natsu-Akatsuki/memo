.. role:: raw-html-m2r(raw)
   :format: html


sphinx-practice
===============

quick tutorial
--------------


* 
  ``sphinx`` , ``MKdocs`` , ``Hexo`` 都可以用来生成html文档

* 
  本次将使用 ``ReadTheDocs`` 部署 ``sphinx`` 生成的文档

步骤一：安装sphinx相关依赖

.. prompt:: bash $,# auto

   # test in ubuntu20.04
   $ sudo apt install python3-sphinx

步骤二：创建基础模板

.. prompt:: bash $,# auto

   $ mkdir sphinx_tutorial && cd sphinx_tutorial
   # 有相关内容需要进行填写，如输入作者、版号等信息
   $ sphinx-quickstart

步骤三：(optional) 生成html并查看

.. prompt:: bash $,# auto

   $ make html
   # 使用其他浏览器打开亦可
   $ google-chrome index.html

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210731000407495.png" style="zoom: 50%; " />`

步骤四：源文件提交至github

.. attention:: 没必要提交编译结果；同时如要免费使用 `ReadTheDocs` 的功能，只能使用公有仓


步骤五：ReadTheDocs关联github

基本在\ `网页 <https://readthedocs.org/>`_\ 上直接 ``Import a project`` 一波就能部署完毕，详细教程可参考\ `link <https://docs.readthedocs.io/en/stable/intro/import-guide.html>`_\ ；\ **连workflow/action这些都不用配置就能部署**

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210731001321545.png" style="zoom: 50%; " />`

Reference
^^^^^^^^^


#. `reStructure文件语法 for sphinx <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#>`_
#. `资料2 <https://sublime-and-sphinx-guide.readthedocs.io/en/latest/images.html>`_
#. `资料3 <https://docs.typo3.org/m/typo3/docs-how-to-document/master/en-us/WritingReST/Admonitions.html>`_
#. `资料4 <https://bashtage.github.io/sphinx-material/rst-cheatsheet/rst-cheatsheet.html>`_

`代码块 <https://sublime-and-sphinx-guide.readthedocs.io/en/latest/code_blocks.html>`_
------------------------------------------------------------------------------------------

上述链接包含如下解决方案


* 使用代码块
* 导入(include)文件作为代码块
* 高亮某一行文本

`显示warnings等 <https://sublime-and-sphinx-guide.readthedocs.io/en/latest/notes_warnings.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning:: This is warning text. Use a warning for information the user must understand to avoid negative consequences. Warnings are formatted in the same way as notes. In the same way, lines must be broken and indented under the warning tag.


`显示emoji <https://sphinxemojicodes.readthedocs.io/en/stable/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. attention:: emoji需要加竖线和空格，例如 |:smile:|


内嵌html/xml
^^^^^^^^^^^^


* `simple usecase <https://stackoverflow.com/questions/50565770/how-to-embed-html-or-xml-in-restructuredtext-sphinx-so-the-browser-cna-render>`_
* `嵌入asciinema脚本 <https://raw.githubusercontent.com/catkin/catkin_tools/master/docs/verbs/catkin_build.rst>`_\ ; `官方教程 <https://asciinema.org/docs/embedding>`_


.. raw:: html

   <center><script type="text/javascript" src="https://asciinema.org/a/GdUfcN6YnWz6GSwNAF9Mqhm25.js" id="asciicast-GdUfcN6YnWz6GSwNAF9Mqhm25"></script></center>


`download role <https://stackoverflow.com/questions/3615142/how-to-include-pdf-in-sphinx-documentation>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For an in-depth explanation, please see :download: ``A Detailed Example <https://appletree.or.kr/quick_reference_cards/Python/Python%20Debugger%20Cheatsheet.pdf>`` .

拓展工具
--------

构建rst文档技巧
^^^^^^^^^^^^^^^

先构建Markdown文档再使用转换工具（如：\ `m2r <https://github.com/miyakogi/m2r>`_\ ）再转其为rst文件

.. prompt:: bash $,# auto

   $ m2r your_document.md [your_document2.md ...]

typora markdown结合阿里云图床
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

文件的本地链接管理比较麻烦，因此可以使用网络链接

.. hint:: [图床工具的使用：PicGo](https://www.jianshu.com/p/9d91355e8418)


vscode拓展插件：\ `vscode-restructuredtext <https://github.com/vscode-restructuredtext/vscode-restructuredtext>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

支持如下功能：


* 
  自定义代码块(live template or `code snippets <https://docs.restructuredtext.net/articles/snippets.html>`_\ )

* 
  live previous（触发方式默认为前导符和 ``ctrl+r``\ ）

* 
  `IntelliSense <https://docs.restructuredtext.net/articles/intellisense.html>`_\ ：包含代码补全（i.e. 文件路径）

* 
  reStructuredText Syntax highlighting（语法高亮）

sphinx拓展插件
^^^^^^^^^^^^^^

第三方插件
~~~~~~~~~~


* 
  `sphinx-prompt <https://sphinx-extensions.readthedocs.io/en/latest/sphinx-prompt.html>`_ ：能够给命令行添加\ **不可选**\ 的前导符(prompt)（\ `实例 <http://sbrunner.github.io/sphinx-prompt/>`_\ ）

  :raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210907091138001.png" alt="image-20210907091138001" style="zoom:50%; " />`

.. hint:: 该插件会丢失code-block的语法高亮功能



* `sphinx-copybutton <https://github.com/executablebooks/sphinx-copybutton>`_ ：给代码块添加复制按钮
* `sphinx-toggleprompt <https://sphinx-toggleprompt.readthedocs.io/en/master/>`_ ：隐藏python代码块的prompt
* `sphinx_last_updated_by_git <https://github.com/mgeier/sphinx-last-updated-by-git>`_\ ：添加更新时间提示

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210912170245248.png" alt="image-20210912170245248" style="zoom:67%; " />`


* `使用Markdown和reStructuredText生成html文件 <https://www.sphinx-doc.org/en/master/usage/markdown.html>`_

.. attention:: 高版本的 `sphinx` 推荐使用 `myst-parser` ，而非 `recommonmark`



* `readthedocs-sphinx-search <https://readthedocs-sphinx-search.readthedocs.io/en/latest/index.html>`_\ ：使用快捷键"/"触发搜索框（常规需要部署到readthedocs才能生效）


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211215115030153.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211215115030153.png
   :alt: image-20211215115030153



* `sphinx-nofound-page <https://github.com/readthedocs/sphinx-notfound-page>`_

内置插件
~~~~~~~~


* `sphinx.ext.todo <https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#confval-todo_include_todos>`_\ ：效果如下，使用方法见\ `此 <https://stackoverflow.com/questions/22290548/sphinx-todo-box-not-showing/22290786>`_

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210907084217088.png" alt="image-20210907084217088" style="zoom:67%; " />`


* `sphinx.ext.autosectionlabel <https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html#module-sphinx.ext.autosectionlabel>`_\ ：实现当前页的标签跳转

参考资料
~~~~~~~~


* https://sphinx-extensions.readthedocs.io/en/latest/

配置文件
--------

切换主题
^^^^^^^^

主题的修改可参考\ `link <https://www.sphinx-doc.org/en/master/usage/theming.html>`_\ ，主要是修改 ``cong.py`` 配置文件中的 ``html_theme`` 字段；常用主题为 ``sphinx_rtd_theme`` ，具体效果如下所示：

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/SwKXV7YrO9MAwnQG.png!thumbnail" alt="img" style="zoom:67%; " />`

SEO
---

`网页审查 website-checker <https://tranngocthuy.com/websitechecker/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`谷歌审查 google search control <https://search.google.com/search-console>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`sitemap生成 <https://www.xml-sitemaps.com/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

sitemap
^^^^^^^

readthedocs生成的网站自带sitemap


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220109165744879.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220109165744879.png
   :alt: image-20220109165744879

