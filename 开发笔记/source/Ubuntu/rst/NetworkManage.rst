.. role:: raw-html-m2r(raw)
   :format: html


NetworkManage
=============
Hareware
---------
查看网卡设备
~~~~~~~~~~

.. prompt:: bash $,# auto

   $ sudo lshw -c network


启动网卡
~~~~~~~~~~~~~~~~~~~~

.. prompt:: bash $,# auto

   # 查看网卡是否启动（看是down还是up）
   $ ip link


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210827010043385.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210827010043385.png
   :alt: image-20210827010043385


.. prompt:: bash $,# auto

   # link：network device
   # interface：网卡名（ref: man）
   $ ip link set <网卡名interface> up/down

.. hint:: `网卡和其对应属性 <https://blog.csdn.net/dxt16888/article/details/80741175>`_\ ： ``eth/eno`` 一般对应有线网卡； ``elan/wlo`` 一般对应无线网卡；br一般与桥接有关

.. note:: ``DOWN``\ 的情况有两种，一种是硬件上没联网（没插网线、没连wifi），二是软件上DOWN了（这种可以命令行UP回去）

Backend
--------

- Ubuntu服务器版和桌面版使用不同的后端（backend）管理网络，分别是 `systemd-networkd` （也称 `networkd` ）和 `network-manager` （也称 `NetworkManger` ）
- `netplan` 用于生成不同后端（ `networkd` 或 `NetworkManger` ）的配置文档

.. tabs::

   .. code-tab:: bash nmcli (NetworkManger)

         # 查看网卡状态
         $ nmcli device status

         # nmcli的connections即配置文档（ref: man）可在以下目录查看
         $ ls /etc/NetworkManager/system-connections
         # 查看配置信息
         $ nmcli connection show
         # 添加配置文档
         # nmcli connection add type <ethernet/wifi> con-name <connection-name> ifname <interface-name>

         # >>> 设置有线连接 >>>
         # 静态ip配置
         $ nmcli connection modify <connection_name>
         ipv4.method manual \
         ipv4.addresses 192.168.1.100/16 \
         ipv4.gateway 192.168.1.1

         # 动态ip配置
         $ nmcli connection modify <connection_name> ipv4.method auto

         # 使配置文档的修改生效
         $ sudo systemctl restart NetworkManager

         # >>> 设置无线连接 >>>
         # 显示可连接的wifi信息/信号强弱
         $ nmcli dev wifi

         # 查看当前的wifi信息（如密码）
         $ nmcli dev wifi show

         # 连接wifi
         # sudo nmcli dev wifi connect <wifi_ssid> password <password>
         $ sudo nmcli dev wifi connect 877381 password 96899968

         # >>> 图形化界面配置 >>>
         $ nm-connection-editor

   .. code-tab:: bash networkd (networkd)

      # 查看网卡状态
      $ networkctl

   .. code-tab:: bash netplan

      # --debug项为可选（需放中间），用于查看日志
      # 使配置文档生效
      $ sudo netplan --debug apply

----

**NOTE**

#. 写在\ ``/etc/netplan``\ 的配置文档的文件名需要以数字为前缀，如\ ``00-netplan.yaml``
#. 实测，若在netplan中设置静态ip时一定要添加 ``nameserver`` ， ``gateway`` (default为0.0.0.0/0)
#. `netplan template <https://gist.github.com/Natsu-Akatsuki/81f5ff6cc53e01950e30d2be9901e269#file-00-netplan-template-yaml>`_
#. `netplan：绑定多张有线网卡以实现网络冗余 <https://gist.github.com/Natsu-Akatsuki/81f5ff6cc53e01950e30d2be9901e269#file-00-netplan-practice-yaml>`_  
#. 服务切换：\ `Network Manager切换到systemd-networkd <https://www.xmodulo.com/switch-from-networkmanager-to-systemd-networkd.html>`_\ ，\ `译文 <https://m.linuxidc.com/Linux/2015-11/125430.htm>`_
#. `bonding的若干种模式介绍 <https://askubuntu.com/questions/464747/channel-bonding-modes>`_
#. `LACP配置实战 <https://www.snel.com/support/how-to-set-up-lacp-bonding-on-ubuntu-18-04-with-netplan/>`_

查看DNS server
--------------

.. prompt:: bash $,# auto

   $ systemd-resolve --status


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210826231213916.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210826231213916.png
   :alt: image-20210826231213916


----

**NOTE**


#. 常用DNS servers

.. code-block:: plain

   # 阿里云
   nameserver 223.5.5.5
   nameserver 223.6.6.6

   # 百度
   nameserver 180.76.76.76

   # 腾讯
   nameserver 119.29.29.29

   # google
   nameserver 8.8.8.8


#.
   dns的配置可以使用nmcli, netplan, 在\ ``/etc/resolv.conf``\ 增加nameserver，或图形化界面上进行修改均可，不赘述

#.
   ``/etc/resolv.conf``\ 的配置只起临时修改作用，重启后会恢复回原来的状态；使其生效需要

.. prompt:: bash $,# auto

   $ sudo service resolvconf restart


#. 配置文档总的其余配置参数（e.g. domain和search）可参考\ `link <https://blog.csdn.net/u010472499/article/details/95216015>`_

----

`查看是否正常解析域名 <https://www.geeksforgeeks.org/nslookup-command-in-linux-with-examples/>`_
----------------------------------------------------------------------------------------------------

.. prompt:: bash $,# auto

   # nslookup www.baidu.com
   $ nslookup <domain_name>


* `在线解析DNS <https://www.ipaddress.com/>`_

`使用arp查看是否ip冲突 <https://www.unixmen.com/find-ip-conflicts-linux/>`_
-------------------------------------------------------------------------------

.. prompt:: bash $,# auto

   $ sudo apt install arp-scan
   $ sudo arp-scan -l <-I device_name>
   # -I 指定网卡设备
   # -l Generate  addresses from network interface configuration

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/6kl0A3112mKoYEFw.png!thumbnail" alt="img" style="zoom:67%; " />`

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/JdGUZH5wPVkEQhnp.png!thumbnail" alt="img" style="zoom:50%; " />`

查看是否正常地分配到ip
----------------------


* 网卡已正确获取IP地址：

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/pb4XovJl3q1QlGQ1.png!thumbnail" style="zoom: 80%; " />`


* 网卡未正确获得IP地址：

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/pb4XovJl3q1QlGQ1.png!thumbnail" alt="img" style="zoom: 80%; " />`

监听端口
--------

.. prompt:: bash $,# auto

   $ netstat
   # -a: all
   # -n：(numerical)显示数值型地址
   # -p：显示socket对应的pid和程序
   # -l：(listen)仅显示正在监听的sockets
   # -t: 列出tcp封包信息（一般与浏览器有关）
   # -u：列出utp封包信息
   $ sudo netstat -anp | grep 32345

修改Hosts与IP的映射
-------------------

.. prompt:: bash $,# auto

   # 永久修改
   $ sudo vim /etc/hosts
   # 等价于
   $ sudo hostnamectl set-hostname <new-hostname>

   # 临时修改
   $ sudo hostname <new-hostname>


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/MZ5nrO4JuUYqaqYf.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/MZ5nrO4JuUYqaqYf.png!thumbnail
   :alt: img


.. note:: 实测可直接生效


路由
----

`显示静态路由表 <https://devconnected.com/how-to-add-route-on-linux/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   # 以下给出三种方案
   $ route -n
   # -n：不将ip解析为域名，能提高route命令行的速度
   $ ip route
   $ netstat -nr


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/LJBYTOBkPD33qnoS.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/LJBYTOBkPD33qnoS.png!thumbnail
   :alt: img


----

**NOTE**

Flags  Possible flags include
              **U (route is up)**
              H (target is a host)  目标ip指向一台主机
              **G (use gateway)**
              R (reinstate route for dynamic routing)
              D (dynamically installed by daemon or redirect)
              M (modified from routing daemon or redirect)
              A (installed by addrconf)
              C (cache entry)
              !  (reject route)

----

屏蔽抵达某个ip的路由
^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ sudo route add -net 10.23.21.110 netmask 255.255.255.255 reject
   # 等价于：
   $ sudo route add -host 10.23.21.110 reject
   # 取消配置
   $ sudo route del -net 10.23.21.110 netmask 255.255.255.255 reject

可由如下效果：


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/8A2OeXYZWVCC63Ok.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/8A2OeXYZWVCC63Ok.png!thumbnail
   :alt: img


增设抵达某个ip的路由
^^^^^^^^^^^^^^^^^^^^

指定抵达\ ``172.16.1.*``\ ip的路由：访问\ ``172.16.1.*``\ 需经过\ ``192.168.43.1``\ 这个网关

.. prompt:: bash $,# auto

   $ sudo route add -net 172.16.1.0 netmask 255.255.255.0 gw 192.168.43.1

路由跟踪
^^^^^^^^

.. prompt:: bash $,# auto

   $ traceroute <ip/domain_name>


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/urGTDLi4UmGEazyP.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/urGTDLi4UmGEazyP.png!thumbnail
   :alt: img


.. note:: 数据先由当前无线网卡192.168.200.123广播到无线路由192.168.200.1，再经过...


`科学上网v2raya <https://v2raya.org/docs/prologue/installation/debian/>`_
-----------------------------------------------------------------------------


* 全局代理开启后，任何tcp流量都会经过代理
* 要主机作为网关，让其他主机或docker也使用代理，则需要开启局域网共享

安装
^^^^

.. prompt:: bash $,# auto

   $ curl -Ls https://mirrors.v2raya.org/go.sh | sudo bash
   $ sudo systemctl disable v2ray --now
   $ wget -qO - https://apt.v2raya.mzz.pub/key/public-key.asc | sudo apt-key add -
   # add V2RayA's repository
   $ echo "deb https://apt.v2raya.mzz.pub/ v2raya main" | sudo tee /etc/apt/sources.list.d/v2raya.list
   $ sudo apt update
   # install V2RayA
   $ sudo apt install v2raya -y
   $ sudo systemctl start v2raya.service
   $ sudo systemctl enable v2raya.service
   # 打开http://127.0.0.1:2017/进行配置（默认网站）

卸载v2ray和v2raya
^^^^^^^^^^^^^^^^^

.. prompt:: bash $,# auto

   # 卸载v2ray(core)
   $ sudo bash go.sh --remove
   # 若设置了自启动，还需删除相关service配置文件
   $ sudo systemctl disable v2raya
   # 删除v2raya cookie

实战
^^^^

`指定代理路由 <https://github.com/v2rayA/v2rayA/issues/376>`_\ （\ `routingA文档 <https://v2raya.org/docs/manual/routinga/>`_\ ）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* ieee设置直连而不进行代理

.. code-block:: go

   # GFWList模式
   default: direct
   # 学术网站
   domain(geosite:google-scholar)->proxy
   domain(geosite:category-scholar-!cn, geosite:category-scholar-cn)->direct
   # domain(ext:"LoyalsoldierSite.dat:gfw", ext:"LoyalsoldierSite.dat:greatfire")->proxy
   domain(geosite:geolocation-!cn)->proxy

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220211142133128.png" alt="image-20220211142133128" style="zoom:67%;" />`

`各种代理的区别 <https://v2raya.org/docs/prologue/quick-start/#%E9%85%8D%E7%BD%AE%E4%BB%A3%E7%90%86>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* 透明代理、系统代理、浏览器代理

实战
----

网络故障排除清单
^^^^^^^^^^^^^^^^


* 是否有网卡，有网卡后，网卡是否启动
* 网卡是否正确的配置，用ifconfig判断是否正确获得了ip
* dns是否正确的配置
* 是否有ip冲突（使用DHCP自动分配或重新静态绑定个未使用的ip）
* 是否启动了代理

同时收发激光雷达数据和上网
^^^^^^^^^^^^^^^^^^^^^^^^^^

使用了激光雷达后无法使用无线上网： ``路由规则`` （i.e. 描述数据传输的路径）配置不妥当。

以下 ``路由规则`` 为：目的地ip为 ``192.168.1.*`` 时使用有线网卡 ``enp89s0`` 进行广播；目的地ip为 ``192.168.43.*`` 时使用无线网卡 ``wlp0s20f3`` 进行广播；同理目的地ip为 ``169.254.*.*`` 时使用有线网卡 ``enp89s0``\ ；其他目的地ip则使用metric最小的默认路由，相关的数据经过有线网卡 ``enp89s0`` ，传输到网关 ``192.168.1.1`` （理论上应该是经过无线网卡 ``wlp0s20f3`` ，传输到网关 ``192.168.43.1`` ）


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/v4fgMRslXtNMbN3b.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/v4fgMRslXtNMbN3b.png!thumbnail
   :alt: img


一种解决方案为，可以删除有线网卡的 ``默认路由`` ，只保留无线网卡的 ``默认路由`` ，让有线网卡处理ip地址为192.168.1.\ *的传感器设备的数据收发，无线网卡访问因特网。换句话说： `192.168.1.*\ ``的ip走有线网卡（收发激光雷达和相机的数据），不用走网关``\ 192.168.2.\ *\ ``的ip走无线网卡（收发互联网的数据），走``\ 192.168.43.*\ ` 的网关

.. prompt:: bash $,# auto

   # 仅生效一次（重启会重置）
   $ route del default enp89s0
   # route -n
   Detstination      Gateway        Flags     Iace     Metric
   0.0.0.0/0        192.168.43.1     UG      wlp0s20f3   600
   169.254.0.0/18     0.0.0.0         U       enp89s0    1000
   192.168.1.0/24     0.0.0.0         U       enp89s0    100
   192.168.43.0/24    0.0.0.0         U      wlp0s20f3   600

子网重复
^^^^^^^^

由于子网重复而无法ping通路由器的\ ``192.168.1.1``\ 可以指定路由进行连接

.. prompt:: bash $,# auto

   # route -n
   Detstination      Gateway        Flags     IFace     Metric
   0.0.0.0/0        192.168.1.1      UG       wlp3s0    20600
   169.254.0.0/18     0.0.0.0         U       enp4s0     1000
   192.168.1.0/24     0.0.0.0         U       enp4s0     100
   192.168.1.0/24     0.0.0.0         U       wlp3s0     600
   $ sudo route add -host 192.168.1.1 wlp3s0
   Detstination      Gateway        Flags     IFace     Metric
   0.0.0.0/0        192.168.1.1      UG       wlp3s0    20600
   169.254.0.0/18     0.0.0.0         U       enp4s0     1000
   192.168.1.0/24     0.0.0.0         U       enp4s0     100
   192.168.1.0/24     0.0.0.0         U       wlp3s0     600
   192.168.1.1/32     0.0.0.0        UH       wlp3s0      0

`cisco交换机指示灯含义一览 <https://community.cisco.com/t5/small-business-switches/sg110-leds/td-p/3208369>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


*
  PWR 电源状态灯
  绿色：系统正常上电且正常工作
  琥珀色：系统正常上电但运行不正常
  关闭：系统未上电

*
  link/act gigabit 端口状态灯：
  绿灯常亮：链路正常
  闪烁：链路正传输数据
  左灯闪，右灯常绿：1G
  左灯闪，右灯不亮：100 Mbps /  10 Mbps
  左灯琥珀色，右灯常绿：check the cables for network loop.
  左灯琥珀色，右灯不亮：most probably you have PoE capable device attached, but there is no link anymore .

服务器不能直接连卡座的网口而需交换机做中介
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/rm22dbmeCnLYNfP4.png!thumbnail" alt="img" style="zoom:67%;" />`

拓展插件
--------

实时查看网速
^^^^^^^^^^^^

.. prompt:: bash $,# auto

   $ sudo apt install ethstatus
   # 监控特定网卡 ethstatus -i <inferface_name>
   $ ethstatus -i eno1

限制网速
^^^^^^^^

----

**ATTENTION**


*
  注意需要sudo，否则配置不生效

*
  此处是 bps ，而不是 Bps

----

.. prompt:: bash $,# auto

   # 设置限速
   # sudo wondershaper 10000 10000
   $ sudo wondershaper <device_name> <下行速度bps> <上行速度bps>
   # 取消限速 sudo wondershaper clear eno1
   $ sudo wondershaper clear <device_name>

测速
^^^^

.. prompt:: bash $,# auto

   $ sudo apt install speedtest-cli
   $ speedtest-cli --bytes

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/bvvQm0BFO9Ber3EB.png!thumbnail" alt="img" style="zoom:67%; " />`

远程登录
^^^^^^^^

.. prompt:: bash $,# auto

   # 安装这个别人才能ssh到本机
   $ sudo apt install openssh-server

在线网站测试工具
^^^^^^^^^^^^^^^^


#. http://tool.chinaz.com/

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210827142503797.png" alt="image-20210827142503797" style="zoom:67%; " />`\ wi
