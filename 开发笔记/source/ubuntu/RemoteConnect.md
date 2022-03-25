# RemoteConnect

## ssh

* 一台电脑对应一个私钥才能成功配置，尚未试一台机上多个私钥
* ssh系统配置文档： `/etc/ssh/sshd_config`

.. attention::  对 `ssh文件权限有严格的要求 <https://docs.digitalocean.com/products/droplets/resources/troubleshooting-ssh/authentication/>`_ bash

### [服务器免密登录](https://wiki.archlinux.org/title/SSH_keys#Copying_the_public_key_to_the_remote_server)

* 使用命令行 `ssh-copy-id` 更方便

```bash
$ ssh-copy-id username@remote-server.org
```

### X11转发

* 使远程X11程序(e.g. firefox, gedit)在本地显示
* 远程服务器的配置文件 `/etc/ssh/sshd_config` 的`X11Forwarding` 需设置为yes

```bash
$ ssh -X user_name@ip_address
```

* 适用于显示一些流量不大的x client或只是短时间可视化的操作（不然受传输速度限制会很卡）

### ssh GUI

在实际的使用中ssh命令行比ssh GUI使用更频繁

* [snowflake](https://github.com/subhra74/snowflake)(bin安装)
* [easyssh(远程登录）](https://github.com/muriloventuroso/easyssh#install-with-flatpak)使用flatpak安装（tested）；[卸载](https://discover.manjaro.org/flatpaks/com.github.muriloventuroso.easyssh)

.. note:: easyssh实测在ubuntu20下有问题，体感不好

### [ConnectBot](https://connectbot.org/)(安卓端ssh)

提供相关IP地址即可，最终效果如下：

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211202105905884.png" alt="image-20211202105904455" style="zoom: 33%;" />

## vnc

### 安装相关依赖和初始化

```bash
# 服务端（受控端）安装
$ sudo apt install tigervnc-common tigervnc-standalone-server
# 安装完后，设置密码和进行一波初始化
$ vncserver

# 关闭某个vncserver
$ vncserver -kill :1
```

### 服务端修改配置文档

添加文件`~/.vnc/xstartup`

* 使用**KDE ssdm display manager**

```bash
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESSt
dbus-launch startplasma-x11
```

* 使用**xfce display manager**

```bash
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
```

### 启动vnc server

```bash
# 尺寸/配置文件/控制端口号
# vncserver [-geometry 1920x1080] [-xstartup /usr/bin/xterm] :1
# 默认根据~/.vnc/xstartup的内容进行启动
$ vncserver -geometry 1920x1080 
# 重新设置vnc密码
$ vncpasswd
```

### 启动vnc client

客户端启动vnc client

```bash
# 安装vncviewer
$ sudo apt install tigervnc-viewer
# ssh helios@192.168.1.112 -L 5901:127.0.0.1:5901
$ ssh <server username>@<server ip> -L 5901:127.0.0.1:5901
# 新开一个终端，账号为localhost:5901，密码为服务端的密码
$ vncviewer
```

---

**NOTE**

```bash
# ssh -L [bind_address:]port:host:hostport
端口绑定，将bind_address:port映射到host:hostport
```

---

### [noVNC(web)](https://github.com/novnc/noVNC)

- 以web端的方式交付vnc

```bash
$ git clone https://github.com/novnc/noVNC.git --depth=1
# install（也可以通过snap安装）和启动
$ ./utils/novnc_proxy --vnc localhost:5901
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/oTge9ryVokLqPaFk.png!thumbnail)

.. note:: 在vnc server端启动

.. note:: 跟vnc viewer一样无法传special key进行操作



