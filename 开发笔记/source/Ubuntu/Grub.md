# Grub

* MBR(master boot record)主引导记录，在第一个可启动设置的第一个扇区上；启动引导(boot loader)安装在上面
* boot loader的作用： `加载内核`
* 按下开机键后的启动流程

1. 计算机硬件主动读取 `BIOS` 来**加载硬件信息**和**进行硬件上的自检**
2. 读取 `第一个可启动设备` （由BIOS设置），并从其中的MBR读入和执行 `启动引导程序`，启动引导程序根据配置，指定加载对应的内核文件到内存进行解压缩和执行
3. 内核在内存中活动，并检测所有硬件信息和加载适当的驱动程序。这个结束后，就搭建完成了一个基本的操作系统。
4. 调用外部程序准备软件执行的环境，加载操作系统运行时的软件程序
5. 等待用户登录和操作
6. 加载内核程序
7. 硬件检查和程序加载（hardware is ready）
8. 内核调用第一个程序 `systemd` ， `systemd` 调用服务集，来配置基本的软件环境（网络环境、语系环境）

## GrubOption

### 修改配置文件

* `/etc/default/grub`

```bash
GRUB_DEFAULT=0
GRUB_HIDDEN_TIMEOUT_QUIET=true
# 启动项永久等待(sec)
GRUB_TIMEOUT=-1       
GRUB_DISTRIBUTOR=`lsb_release -i -s 2> /dev/null || echo Debian`
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
GRUB_CMDLINE_LINUX=""
```

---

[选项说明](https://askubuntu.com/questions/716957/what-do-the-nomodeset-quiet-and-splash-kernel-parameters-mean)

quiet：不输出日志信息

ro：boot时/boot为只读

nomodest：禁用KMS(kernel mode setting)，不在内核导入时配置分辨率和颜色深度

nouveau.modeset=0：启动时禁用 nouveau drivers

此处的splash不等同于kde的splash

---

.. attention:: 修改配置文件后，需执行命令行 `update-grub` 以生成 `/boot/grub/grub.cfg` / 生效

## 显示开机启动(boot)时加载的参数

```bash
$ cat /proc/cmdline
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/OAszWAD2imR7ZbMI.png!thumbnail)

## [开关机时显示boot日志](https://itectec.com/ubuntu/ubuntu-how-to-enable-boot-messages-to-be-printed-on-screen-during-boot-up/)

.. attention:: 有时候黑屏时只保留光标，可以按``F11``等键来查看相关的grub日志

## [修改grub/tty界面的分辨率](https://wiki.archlinux.org/title/GRUB/Tips_and_tricks#Setting_the_framebuffer_resolution)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/QqOPCOHKD7D4af68.png!thumbnail)

```bash
# 在/etc/default/grub中添加如下几行
GRUB_GFXMODE=1920x1080 
GRUB_GFXPAYLOAD_LINUX=keep
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/wP1h8CkXV812by7G.png!thumbnail" alt="img" style="zoom:80%;" />

.. note:: 查看支持的分辨率和深度（实测hwinfo的不准，应该参考grub命令行的videoinfo）

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/CTWAJIEnWOpfT104.jpg!thumbnail)

## 启动时默认进入window没有grub

在bios设置启动引导优先级，优先为ubuntu

## grub界面显示异常

```bash
$ sudo update-grub
```

## [制作多重引导](https://www.linuxbabe.com/apps/create-multiboot-usb-linux-windows-iso)

步骤一：安装[Ventory](https://github.com/ventoy/Ventoy/releases)并用其格式化U盘

步骤二：安装镜像并导入到U盘后即可使用

镜像网站参考：

* ubuntu
* [kubuntu](https://kubuntu.org/getkubuntu/)
