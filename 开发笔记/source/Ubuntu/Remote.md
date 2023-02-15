# Remote

远程连接

## SSH

* 一台电脑对应一个私钥才能成功配置，尚未试一台机上多个私钥
* ssh系统配置文档： `/etc/ssh/sshd_config`

.. attention::  对 `ssh文件权限有严格的要求 <https://docs.digitalocean.com/products/droplets/resources/troubleshooting-ssh/authentication/>`_ bash

### Install

```bash
# 安装这个别人才能ssh到本机
$ sudo apt install openssh-server
```

### [服务器免密登录](https://wiki.archlinux.org/title/SSH_keys#Copying_the_public_key_to_the_remote_server)

* 使用命令行 `ssh-copy-id` 更方便

```bash
$ ssh-copy-id username@remote-server.org
```

### X11 Forward

* 使远程X11程序(e.g. firefox, gedit)在本地显示
* 远程服务器的配置文件 `/etc/ssh/sshd_config` 的`X11Forwarding` 需设置为yes

```bash
$ ssh -X user_name@ip_address
```

* 适用于显示一些流量不大的x client或只是短时间可视化的操作（不然受传输速度限制会很卡）

### GUI

在实际的使用中ssh命令行比ssh GUI使用更频繁，暂时图形化管理还是比较鸡肋

* [snowflake](https://github.com/subhra74/snowflake)（bin安装）
* [easyssh(远程登录）](https://github.com/muriloventuroso/easyssh#install-with-flatpak)使用flatpak安装（tested）；[卸载](https://discover.manjaro.org/flatpaks/com.github.muriloventuroso.easyssh)

.. note:: easyssh实测在ubuntu20下有问题，体感不好

* [asbru-cm](https://github.com/asbru-cm/asbru-cm)

```bash
$ curl 'https://dl.cloudsmith.io/public/asbru-cm/release/cfg/setup/bash.deb.sh' | sudo bash
$ sudo apt install asbru-cm
```

### [ConnectBot](https://connectbot.org/)

安卓端SSH，提供相关IP地址即可，最终效果如下：

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211202105905884.png" alt="image-20211202105904455" style="zoom: 33%;" />

## VNC

### Install

#### TigerVNC

* 能稳定地在docker中使用

```bash
# 安装服务端（受控端）
$ sudo apt install tigervnc-common tigervnc-standalone-server tigervnc-xorg-extension

# 安装服务端（控制端）
$ sudo apt install tigervnc-viewer

# 安装完后，设置密码和进行一波初始化
$ vncserver

# 关闭某个vncserver
$ vncserver -kill :1
```

#### [TurboVNC](https://sourceforge.net/projects/turbovnc/files/)

```bash
# 安装
$ wget -c "https://downloads.sourceforge.net/project/turbovnc/3.0/turbovnc_3.0_amd64.deb?ts=gAAAAABikQPtLcfRHL3VSbB2izA4d1rmaDANhrm7xE00zhL8-q403sxZhfLgXYz13VHS8v0BHCeeEG49ObEjAfFv44hCZnH5hA%3D%3D&use_mirror=udomain&r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fturbovnc%2Ffiles%2F3.0%2F" -O turbovnc_3.0_amd64.deb
$ sudo dpkg -i turbovnc_3.0_amd64.deb
# 设置服务
$ sudo /lib/systemd/systemd-sysv-install enable tvncserver
# vim ~/.bashrc，然后即可等价地使用vncserver和vncviewer...
$ TURBOVNC="/opt/TurboVNC/bin"
$ export PATH="${TURBOVNC}:$PATH"
```

## x11VNC

```bash
# 服务端安装
$ sudo apt install x11vnc
```

### Configure

在服务端修改配置文档，添加文件`~/.vnc/xstartup`，看不同的`Deskop Environment`进行配置

* 使用**KDE**（[最新的KDE已没有startkde而由startplasma-x11替代](https://askubuntu.com/questions/746885/start-kde-5-through-vnc)）

```bash
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
# startkde e.g. ubuntu18.04
dbus-launch startplasma-x11 # startplasma-wayland 
```

* 使用**lxqt display manager**

```bash
#!/bin/bash
startlxqt &
```

* 使用**lxde display manager**（可用）

```bash
#!/bin/bash
startlxde &
```

* 使用**xfce display manager**

```bash
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
```

### Launch

#### 启动服务端

```bash
# 尺寸/配置文件/控制端口号
# vncserver [-geometry 1920x1080] [-xstartup /usr/bin/xterm] :1
# :1对应5901；:2对应5902
# 默认根据~/.vnc/xstartup的内容进行启动
$ vncserver -geometry 1920x1080
# 容器配置（适用于tigerVNC）
$ vncserver :0 -localhost no
# 重新设置vnc密码
$ vncpasswd
```

#### 启动客户端

```bash
# 构建ssh隧道，连接服务端5901和客户端5901端口
# ssh -L [bind_address:]port:host:hostport
# ssh helios@192.168.1.112 -L 5901:127.0.0.1:5901
$ ssh <server username>@<server ip> -L 5901:127.0.0.1:5901
# 新开一个终端，账号为localhost:5901，密码为服务端的密码
$ vncviewer localhost:5901
```

### Auto Start

* vncserver自1.11开始新增了system服务，binary（[heres](https://github.com/TigerVNC/tigervnc/releases)），但实测效果不ok（黑屏）

* TurboVNC/tigerVNC/etc/systemd/system/vnc@.service

```service
[Unit]
Description=TurboVNC remote desktop service
After=syslog.target network.target

[Service]
Type=simple
User=helios
PAMName=login
PIDFile=/home/helios/.vnc/%H%i.pid
ExecStartPre=/bin/bash -c '/opt/TurboVNC/bin/vncserver -kill %i > /dev/null 2>&1 || :'
ExecStart=/opt/TurboVNC/bin/vncserver %i -fg -xstartup /home/helios/.vnc/xstartup
ExecStop=/bin/bash -c '/opt/TurboVNC/bin/vncserver -kill %i > /dev/null 2>&1 || :'

[Install]
WantedBy=multi-user.target
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220418184210024.png" alt="image-20220418184210024" style="zoom:67%;" />

### [noVNC](https://github.com/novnc/noVNC)

* 以web端的方式交付VNC，需在服务端启动

```bash
$ git clone https://github.com/novnc/noVNC.git --depth=1
# install（也可以通过snap安装）和启动
$ ./utils/novnc_proxy --vnc localhost:5901
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/oTge9ryVokLqPaFk.png!thumbnail)

### Reference

* [TigerVNC（含常见的Q&A）](https://wiki.archlinux.org/title/TigerVNC_(%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87)#%E6%B2%A1%E6%9C%89%E7%AA%97%E5%8F%A3%E8%A3%85%E9%A5%B0/%E8%BE%B9%E6%A1%86/%E6%A0%87%E9%A2%98%E6%A0%8F/%E6%97%A0%E6%B3%95%E7%A7%BB%E5%8A%A8%E7%AA%97%E5%8F%A3)
* [各种display manager的配置](https://bytexd.com/how-to-install-configure-vnc-server-on-ubuntu-20-04/)

## TODO

* TigerVNC（container 18.04 / 20.04 -> 22.04）的复制粘贴效果失效
* TurboVNC似乎不支持localhost的链接

* 容器中使用基于KDE/GNOME的VNC（开启了systemd）会使主机端的一部分应用程序无法使用

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220419095839592.png" alt="image-20220419095839592"  />
