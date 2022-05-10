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

- tigerVNC

```bash
# 服务端（受控端）安装
$ sudo apt install tigervnc-common tigervnc-standalone-server tigervnc-xorg-extension
# 安装完后，设置密码和进行一波初始化
$ vncserver

# 关闭某个vncserver
$ vncserver -kill :1
```

- [turboVNC](https://sourceforge.net/projects/turbovnc/files/)

```
$ wget -c https://udomain.dl.sourceforge.net/project/turbovnc/2.2.90%20%283.0%20beta1%29/turbovnc_2.2.90_amd64.deb
$ sudo dpkg -i turbovnc_2.2.90_amd64.deb

$ sudo /lib/systemd/systemd-sysv-install enable tvncserver



# vim ~/.bashrc，然后即可等价地使用vncserver和vncviewer...
TURBOVNC="/opt/TurboVNC/bin"
export PATH="${TURBOVNC}:$PATH"
```





### 服务端修改配置文档

添加文件`~/.vnc/xstartup`

* 使用**KDE ssdm display manager**

```bash
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
# startkde e.g. ubuntu18.04
dbus-launch startplasma-x11 # startplasma-wayland 
```

---

**NOTE**

- [最新的KDE已没有startkde而由startplasma-x11替代](https://askubuntu.com/questions/746885/start-kde-5-through-vnc)了

---

- 使用**lxqt display manager**（可用）

安装：

```bash
$ sudo apt install lxqt
```

启动文档`~/.vnc/xstartup`

```bash
#!/bin/bash
startlxqt &
```

* 使用**lxde display manager**（可用）

安装：

```bash
$ sudo apt install startlxde
```

启动文档`~/.vnc/xstartup`

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

### 启动vnc server

```bash
# 尺寸/配置文件/控制端口号
# vncserver [-geometry 1920x1080] [-xstartup /usr/bin/xterm] :1
# :1对应5901；:2对应5902
# 默认根据~/.vnc/xstartup的内容进行启动
$ vncserver -geometry 1920x1080
# 重新设置vnc密码
$ vncpasswd
```

### 配置文档

- `~/.vnc/config`

```bash
geometry=1920x1080
localhost # 仅localhost能访问
alwaysshared # 其他用户可以同时访问
```

### 启动vnc client

客户端启动vnc client

```bash
# 安装vncviewer
$ sudo apt install tigervnc-viewer

# 构建ssh隧道，连接服务端5901和客户端5901端口
# ssh helios@192.168.1.112 -L 5901:127.0.0.1:5901
$ ssh <server username>@<server ip> -L 5901:127.0.0.1:5901
# 新开一个终端，账号为localhost:5901，密码为服务端的密码
$ vncviewer localhost:5901
```

---

**NOTE**

```bash
# ssh -L [bind_address:]port:host:hostport
端口绑定，将bind_address:port映射到host:hostport
```

---

### VNC自启动

- vncserver自1.11开始新增了system服务，binary([heres](https://github.com/TigerVNC/tigervnc/releases))，但实测效果不ok（黑屏）

- TurboVNC/tigerVNC

  /etc/systemd/system/vnc@.service

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

### BUG

- 使用VNC会使主机端的一部分应用程序无法使用

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220419095839592.png" alt="image-20220419095839592"  />

### 拓展资料

- [TigerVNC（含常见的Q&A）](https://wiki.archlinux.org/title/TigerVNC_(%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87)#%E6%B2%A1%E6%9C%89%E7%AA%97%E5%8F%A3%E8%A3%85%E9%A5%B0/%E8%BE%B9%E6%A1%86/%E6%A0%87%E9%A2%98%E6%A0%8F/%E6%97%A0%E6%B3%95%E7%A7%BB%E5%8A%A8%E7%AA%97%E5%8F%A3)

- [各种display manager的配置](https://bytexd.com/how-to-install-configure-vnc-server-on-ubuntu-20-04/)i
