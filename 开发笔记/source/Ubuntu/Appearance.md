# Apperance

## Deskop Environment

### [KDE](https://itsfoss.com/install-kde-on-ubuntu/)

```bash
# 安装（至上而下越来越全）
$ sudo apt install kde-plasma-desktop
$ sudo apt install kubuntu-desktop
```

### GNOME

* [安装(for ubuntu20.04) tutorial](https://linuxconfig.org/how-to-install-gnome-on-ubuntu-20-04-lts-focal-fossa)

```bash
# 装完整版的gnome
$ sudo apt install tasksel 
$ sudo tasksel install ubuntu-desktop 
$ sudo reboot
```

* [卸载 tutorial](https://itectec.com/ubuntu/ubuntu-how-to-remove-gnome-desktop-environment-without-messing-unity-de-ubuntu-16-04/)

### Unity

```bash
# 卸载
$ sudo apt purge unity-session unity
$ sudo apt autoremove
```

## KDE Apperance

### Cursor

固定光标大小，避免不同分辨率屏幕下有不同大小的光标

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/Rhe2shG5FWiLNVig.png!thumbnail" alt="img" style="zoom:50%;" />

### Dolphin

* 安装拓展插件

```bash
$ sudo apt install dolphin-plugin
```

* 其他

Copy Path(kde for ubuntu 22.04已支持)

Comparing using Meld

Color Folder

Open in VSCode

### Global Theme

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/841boYdUYRUgyp3c.png!thumbnail)

.. attention:: 有些主题或会出现icon异常的问题，则需补装相关的icon数据

### Login Screen

* Ant-Dark

### Konsole

* 设置命令行(command)、设置配色(Awave Dark)、滚轮历史无限制、鼠标中键效果

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210903224125634.png" alt="image-20210903224125634" style="zoom:67%; " />

* 配置`Tab Bar`

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210903224655508.png" alt="image-20210903224655508" style="zoom:67%; " />

* 配色：Breeze
* 快捷键：

| 作用                             | 快捷键                    |
| -------------------------------- | ------------------------- |
| 水平切分窗口                     | ctrl+(                    |
| 垂直切分窗口                     | ctrl+)                    |
| 切换窗口                         | ctrl+tab / ctrl+shirt+tab |
| 切换tab                          | shirt+左/右箭头           |
| 放大窗口（适用于窗口切分的情况） | ctrl+shift+E              |

* 配置功能

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220523010417070.png" alt="image-20220523010417070" style="zoom:50%;" />

### Screen Login

* 配置login界面

### Splash

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/MgDV5vsgIAOg6G8G.png!thumbnail" alt="img" style="zoom: 50%; " />

### TaskBar Widget

#### Thermal Monitor

温度监控

![image-20210903220735147](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210903220735147.png)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210903221123764.png" alt="image-20210903221123764" style="zoom:67%; " />

#### Netspeed

网速监控

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/RmpQAPaNby1pBB9u.png!thumbnail)

#### Tiled Menu

菜单栏

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/wrEljlwjjaoqIFfL.png!thumbnail)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210903221418543.png" alt="image-20210903221418543" style="zoom:67%; " />

### Task Switcher

* 设置compact

### User Manager

* 设置头像

### Window Decoration

* windows decoration: Nordic-r
* 设置window behaviour：鼠标中键（shade）、滚轮（上下层切换）

## KDE Shortcut

* 此处的 `meta` 即 `super`
* 快捷键：

| 作用                               | 快捷键          |
| ---------------------------------- | --------------- |
| 显示桌面                           | meta+D          |
| 窗口挪动                           | meta+方向键     |
| 切换task manager                   | meta+数字       |
| 窗口游走（底部）                   | meta+alt+方向键 |
| 任务管理器（win概念）              | ctrl+ESC        |
| activity游走                       | meta+tab        |
| 运行krunner                        | alt+space       |
| 创建新的dophin                     | meta+e          |
| 在文件夹图形化界面下，跳转到家目录 | alt+home        |

* 创建activity：

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/7gVEkmaTCX6Z5exQ.png!thumbnail" alt="img" style="zoom:80%;" />

## Display Server Protobuf

### X11

* X windows system是一个网络框架，包含客户端(X client)和服务端(X server)
* X windows system是一个软件

* X server用于管理硬件；X client用于管理应用程序

* 配置文件默认放在 `/etc/X11` 目录下
* 日志文件默认为`/var/log/Xorg.0.log`

* 只有$DISPLAY变量有值，才能够使用Xserver服务，如tty1没有该变量， 所以无法顺利执行图形化应用程序，如执行xclock，会返回"can't open display"

#### 为什么默认情况下没有`/etc/X11/xorg.conf`

X server启动时会自行检测系统的显卡，屏幕类型，然后 `自行搭配优化的驱动程序` 加载，[如果要自定义的话，建议通过覆盖的形式](https://unix.stackexchange.com/questions/505088/x-configure-doesnt-work-number-of-created-screens-does-not-match-number-of-d)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/IvdxWDjSRpRkJSE3.png!thumbnail)

### Wayland

#### 安装

```bash
# for KDE
$ sudo apt install plasma-workspace-wayland
```

#### 应用

* [waydroid](https://docs.waydro.id/usage/install-on-desktops)

Ubuntu下运行安卓

```bash
# 导入ppa
$ export DISTRO="focal" && sudo curl https://repo.waydro.id/waydroid.gpg --output /usr/share/keyrings/waydroid.gpg && echo "deb [signed-by=/usr/share/keyrings/waydroid.gpg] https://repo.waydro.id/ $DISTRO main" > ~/waydroid.list && sudo mv ~/waydroid.list /etc/apt/sources.list.d/waydroid.list && sudo apt update

# 安装
$ sudo apt install waydroid

# 初始化配置
$ sudo waydroid init
# 启动waydroid服务
$ sudo systemctl start waydroid-container

$ waydroid show-full-ui
```

---

**NOTE**

* [参考资料 archlinux](https://wiki.archlinux.org/title/Waydroid)

* [X11和wayland的切换](https://itsfoss.com/switch-xorg-wayland/)

---

## Display Manager

```bash
# 查看当前的display manager
$ cat /etc/X11/default-display-manager
# 启动display manager
$ sudo systemctl restart lightdm (ubuntu default)
$ sudo systemctl restart gdm (Gnome default)
$ sudo systemctl restart kdm (sddm)(KDE default)

# 切换图形化界面
$ sudo dpkg-reconfigure <display-manager>
```

---

**NOTE**

XFCE为轻量级的display manager

---

## Fonts

### [解决jupyter乱码问题](https://blog.csdn.net/BigData_Mining/article/details/111656101)

```bash
$ wget -c https://github.com/StellarCN/scp_zh/blob/master/fonts/SimHei.ttf?raw=true
$ sudo cp SimHei.ttf /usr/share/fonts/
# 重新运行
```

## [Windows Manager](https://wiki.archlinux.org/title/Window_manager)

### Stacking Window Managers

#### [Metacity](https://en.wikipedia.org/wiki/Metacity)

### Tilting Window Managers

#### Kwin

* K系统自带

* Tilting extension

```bash
# 触发脚本
current=`kreadconfig5 --file kwinrc --group Plugins --key krohnkiteEnabled`

if [ $current = "true" ]; then
  kwriteconfig5 --file kwinrc --group Plugins --key krohnkiteEnabled false
elif [ $current = "false" ]; then
  kwriteconfig5 --file kwinrc --group Plugins --key krohnkiteEnabled true
fi

qdbus org.kde.KWin /KWin reconfigure
```

#### [Bismuth](https://github.com/Bismuth-Forge/bismuth/tree/master)

[系统依赖较高，至少需要ubuntu21+](https://volian.org/bismuth/)

### [Compositor](https://dev.to/l04db4l4nc3r/compositors-in-linux-1hhb)

用于调整窗口的特效 / 透明度；有时应用程序存在黑边时则可以启动合成器

#### kwin compositor

一般直接用KDE环境默认的合成器即可，使用一些轻量级的桌面环境时才需要下载额外的合成器

#### [compton](https://github.com/chjj/compton)

```bash
# 启动X11 compositor
$ compton -b
```

#### [picom](https://github.com/yshui/picom)

* apt for ubuntu20.10+；其他版本需要源码安装；[ppa安装](https://libredd.it/r/kde/comments/p822c2/perfect_kde_plasma_compositing_combo_kwin_picom/)

### Practice

#### [切换windows manager](https://userbase.kde.org/Tutorials/Using_Other_Window_Managers_with_Plasma)

配置文档一般在/usr/share/xsessions

## [Create Link](https://wiki.archlinux.org/title/desktop_entries)

* 存放桌面快捷方式的位置：/usr/share/application
* [exec 使用说明](https://specifications.freedesktop.org/desktop-entry-spec/latest/ar01s07.html)

* 增加pycharm快捷方式

```bash
[Desktop Entry]
Name=pycharm
Type=Application

Exec=bash -c "命令行1 && 命令行2"
Terminal=false
Icon=图标的绝对位置
```

---

**NOTE**

`bash -c "source ~/.bashrc"` 无效，因为

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/dgH8iQP5jrkgW2hE.png!thumbnail)

此时可加上 `-i` 这个选项来创建交互式的bash来执行脚本`bash -i -c "source ~/.bashrc"`  

---

## Debug

### Black Screen

* 有光标(cursor)：/boot空间不够
