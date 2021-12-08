.. role:: raw-html-m2r(raw)
   :format: html


机械狗20.04
===========

相关依赖安装
------------

.. prompt:: bash $,# auto

   $ sudo apt install mesa-common-dev freeglut3-dev coinor-libipopt-dev libblas-dev liblapack-dev gfortran liblapack-dev coinor-libipopt-dev cmake gcc build-essential libglib2.0-dev
   $ sudo apt-get install -y libqt5gamepad5-dev

`安装lcm <https://lcm-proj.github.io/build_instructions.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

需要进行官网上所述的后处理，否则会有如下Error：\ `Error while loading shared libraries: liblcm.so.1: cannot open shared object file: No such file or directory <https://github.com/CogChameleon/ChromaTag/issues/2>`_ 

googletest：'master' tag missing from github breaking cmake builds
------------------------------------------------------------------

将master git tag改为main


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/rk620BCpYkbfqsYI.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/rk620BCpYkbfqsYI.png!thumbnail
   :alt: img


eigen路径问题
-------------


* 方法一(recommend)：

.. prompt:: bash $,# auto

   $ sudo ln -s /usr/include/eigen3 /usr/local/include/eigen3


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ow4h2nnkNcUDeLpj.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ow4h2nnkNcUDeLpj.png!thumbnail
   :alt: img



* 方法二：修改CMakeLists.txt


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/eKG2H7COousqe9h7.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/eKG2H7COousqe9h7.png!thumbnail
   :alt: img
`no stropts <https://blog.csdn.net/jongden/article/details/24995415>`_
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

.. prompt:: bash $,# auto

   $ touch /usr/include/stropts.h

`ioctl was not declared in this scope <https://www.codeleading.com/article/93875966263/>`_
----------------------------------------------------------------------------------------------


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/QjGKxU2zfT8NebHD.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/QjGKxU2zfT8NebHD.png
   :alt: img


`cc1plus: all warnings being treated as errors <https://stackoverflow.com/questions/11561261/how-to-compile-without-warnings-being-treated-as-errors>`_
-----------------------------------------------------------------------------------------------------------------------------------------------------------

**usr/include/x86_64-linux-gnu/bits/string_fortified.h:106:34:** **error:** ‘\ **char* __builtin_strncpy(char*, const char*, long unsigned int)**\ ’ specified bound 2056 equals destination size [\ **-Werror=stringop-truncation**\ ]   106 |  return **\ **builtin_**\ strncpy_chk (\ **dest, **\ src, **len, **\ bos (__dest))**\ ;     |      **\ :raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~~^</del>`\ :raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~</del>`\ ** cc1plus: all warnings being treated as errors
