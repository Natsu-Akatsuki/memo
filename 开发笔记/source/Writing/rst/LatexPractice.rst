.. role:: raw-html-m2r(raw)
   :format: html


LatexPractice
=============

`Tex Live <https://www.tug.org/texlive>`_
---------------------------------------------

通过图形化界面安装Tex Live
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ wget -c https://mirror.ctan.org/systems/texlive/tlnet/install-tl.zip
   $ unzip install-tl.zip
   # cd到解压目录
   # note: 该图形化界面可以选择镜像源
   $ sudo ./install-tl -gui

添加环境变量
~~~~~~~~~~~~

.. prompt:: bash $,# auto

   # 配置latex环境
   $ export MANPATH=${MAT_PATH}:"/usr/local/texlive/2022/texmf-dist/doc/man" 
   $ export INFOPATH=${INFOPATH}:"/usr/local/texlive/2022/texmf-dist/doc/info" 
   $ export PATH=${PATH}:"/usr/local/texlive/2022/bin/x86_64-linux"

Semantic
--------

公式对齐
^^^^^^^^

.. code-block:: latex

   \begin{aligned}
    a + b + c &= d \
    e + f &= g  
   \end{aligned}

公式编号
^^^^^^^^

.. code-block:: latex

   % 注意加*在equation后不生成公式编号
   \begin{equation}
     a+b=\gamma\label{eq}
   \end{equation}

图片插入
^^^^^^^^

.. code-block:: latex

   % 需要特定格式的图片，不是所有图片格式都能用
   % width项用于调整图片大小
   \begin{figure}[htbp]
     \includegraphics[width=7cm]{elbow_robot_arm.png}
     \caption{肘型机械臂}
   \end{figure}

`文本颜色 <https://tex.stackexchange.com/questions/17104/how-to-change-color-for-a-block-of-texts>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: latex

   \usepackage{xcolor}
   \begin{document}

   This is a sample text in black.
   \textcolor{blue}{This is a sample text in blue.}

   \end{document}

文本居中
^^^^^^^^

.. code-block:: latex

   \centerline{$r=x_4^2+y_4^2$}

`字体大小 <https://blog.csdn.net/zou_albert/article/details/110532165>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

字体类型
^^^^^^^^


* ``\cal``\ `花体 <https://www.cnblogs.com/xiaofeisnote/p/13423726.html>`_  ；\ ``\mathbb`` `空体 <https://www.overleaf.com/learn/latex/Mathematical_fonts>`_

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/uBiXd1DVMqM5e3o5.png!thumbnail" alt="img" style="zoom:50%;" />`

构建引用
^^^^^^^^

.. code-block:: latex

   \bibliographystyle{IEEEtran} 
   \bibliography{<.bst文件名>}

IDE
---

`Texstudio <http://texstudio.sourceforge.net/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Vscode
^^^^^^

`LaTeX Workshop <https://github.com/James-Yu/LaTeX-Workshop/wiki/Install#usage>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* 启动Chktex：\ `语法检查工具 <https://www.nongnu.org/chktex/>`_\ ；安装tex live后自带


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220508214254785.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220508214254785.png
   :alt: image-20220508214254785



* 
  `indent格式化 <https://github.com/James-Yu/LaTeX-Workshop/wiki/Format#LaTeX-files>`_\ ：安装tex live后自带，ctrl+shirt+I触发

* 
  位置跳转：ctrl+点击pdf的对应位置，实现编辑位置的跳转

Code Spell Checker
~~~~~~~~~~~~~~~~~~


* 词汇补全和正确性校验

LTeX
~~~~

latex/ markdown的文本语法检查

同步pdf和latex文本的位置
^^^^^^^^^^^^^^^^^^^^^^^^

根据pdf定位到latex的位置：ctrl+点击pdf某个位置

根据latex位置定位到pdf的位置：命令行SyncTeX

实战
----

`IEEE中文模板 <https://blog.csdn.net/qq_34447388/article/details/86488686>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

IEEE模板
--------

关键词
^^^^^^

.. code-block:: latex

   \begin{IEEEkeywords}
     Dynamic trajectory planning, MPC, obstacle avoidance.
   \end{IEEEkeywords}

贡献分段
^^^^^^^^

.. code-block:: latex

   \begin{enumerate}
     \item ...
     \item ...
   \end{enumerate}

拓展插件
--------

`CTEX <http://www.ctex.org/HomePage>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

支持中文的拓展插件

格式化
^^^^^^


* latexindent

.. prompt:: bash $,# auto

   $ latexindent a.tex -o b.tex

拓展资料
--------


* `awesome latex <https://asmcn.icopy.site/awesome/awesome-LaTeX/>`_
