.. role:: raw-html-m2r(raw)
   :format: html


DotPractice
===========

Dot语言是开源工具包\ `Graphviz <http://graphviz.org/>`_\ 的脚本语言，用于画图

IDE
---

Jetbrain
^^^^^^^^

Dot Language
~~~~~~~~~~~~

支持语法高亮、在线渲染和预览

`External Tool <https://stackoverflow.com/questions/52352836/visualise-dot-files-in-pycharm>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   # Tools Setting
   dot
   -Tpng $FileName$ -o $FileNameWithoutExtension$.png
   $FileDir$

Install
-------

.. prompt:: bash $,# auto

   $ sudo apt install graphviz

Practice
--------

`可视化TensorRT引擎 <https://zhuanlan.zhihu.com/p/422108110>`_

QuickUsage
----------


* 编写demo.dot脚本

.. code-block:: dot

   digraph demo {
       A -> B [dir = both];
       B -> C [dir = none];
       C -> A [dir = back];
       D -> C [dir = forward];
   }


* 生成图片（需先下载graphviz）

.. prompt:: bash $,# auto

   $ dot demo.dot –Tpng –o demo.png

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/fD240mKwDQ5y4vDU.png!thumbnail" alt="img" style="zoom:50%;" />`

Reference
---------


* `csdn 基础用法 <https://blog.csdn.net/jy692405180/article/details/52077979>`_
