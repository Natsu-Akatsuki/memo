# README

## Usage

现已测试的公开仓库包括：

[kaho_slam](https://github.com/kahowang/sensor-fusion-for-localization-and-mapping)(cuda:11.1.1-cudnn8-devel-ubuntu18.04)

[shenlan](https://github.com/Natsu-Akatsuki/shenlan)(perception)(cuda:11.1-cudnn8-devel-ubuntu20.04 + gnome)

# BUG

- 基于X11的ros rviz(noetic)无法修改此处的值

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220329145821156.png" alt="image-20220329145821156" style="zoom: 50%;" />

- 暂时无法顺利的调通KDE的display manager

  e.g. kdeinit启动失败，dbus问题
  

## Feature

- 支持 gnome-vnc

```bash
# 如下为比较重要的代码块
# >>> 安装和配置vnc server >>>
RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y \
        tigervnc-common \
        tigervnc-standalone-server \
        tigervnc-xorg-extension \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p $HOME/.vnc \
    && echo "admin" | vncpasswd -f >> $HOME/.vnc/passwd && chmod 600 $HOME/.vnc/passwd
    
COPY tigervnc@.service /etc/systemd/system/tigervnc@.service
RUN systemctl enable tigervnc@:1
COPY xstartup /root/.vnc/xstartup

# >>> 安装gnome display manager >>>
RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y --fix-missing \
        ubuntu-desktop \
        gnome-shell-extensions \
        gnome-tweaks \
        gnome-tweak-tool \
        gnome-panel \
    && rm -rf /var/lib/apt/lists/*

# >>> 入口点函数 >>>
ENTRYPOINT ["/usr/sbin/init"]
```

## 参考案例

- [gnome vnc 实例](https://github.com/RavenKyu/docker-ubuntu-desktop-vnc/blob/main/Dockerfile)
