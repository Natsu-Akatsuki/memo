
Hook应用：防止用不合法的邮箱进行commit
======================================

01. 知识点
----------


#. git ``hook``\ 是一个脚本（可执行文件bash shell或者python均可），是执行一些git的操作前或者操作后需要运行的脚本
#. ``hook``\ 可以根据触发的时机分为两类：客户端(clien-side)或者服务端(server-side)，前者如git commit/merge，后者如服务端接收到推送的commit
#. git init后会有一系列的hook模板在\ ``.git/hooks``\ 下来提供参考，可以在此基础上进行修改

.. code-block:: bash

   ~/.git/hooks$ tree
   .
   ├── applypatch-msg.sample
   ├── commit-msg.sample
   ├── fsmonitor-watchman.sample
   ├── post-update.sample
   ├── pre-applypatch.sample
   ├── pre-commit.sample
   ├── pre-merge-commit.sample
   ├── prepare-commit-msg.sample
   ├── pre-push.sample
   ├── pre-rebase.sample
   ├── pre-receive.sample
   └── update.sample

.. attention::
   使用前面提到的 ``git/hook`` 中的脚本，并不能同步到远程仓


1.  参考资料
------------


#. `怎样防止同事用 QQ 邮箱提交公司代码 <https://mp.weixin.qq.com/s/nTujGu1tbde--X3KEO22WA>`_
