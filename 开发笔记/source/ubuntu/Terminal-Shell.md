# Terminal && Shell

## [tmux](https://manpages.ubuntu.com/manpages/focal/en/man1/tmux.1.html)（终端复用）

### 安装

```bash
$ sudo apt install -y tmux
```

### 配置文档

触发配置文档生效

```bash
$ tmux source ~/.tmux.conf`
```

.. note:: 配置文档所在位置为 `~/.tmux.conf`

### 分屏

![image-20210902091648903](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210902091648903.png)

```bash
# 分屏得四宫格(split-window alias:split)
$ tmux new -s ros
$ tmux split -v
$ tmux split -h
$ tmux select-pane -t 1
$ tmux split -h
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210902094739706.png" alt="image-20210902094739706" style="zoom:67%; " />

### pane操作

* 显示panes

```bash
$ tmux list-panes
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210902092500394.png" alt="image-20210902092500394" style="zoom:67%; " />

* panes切换

```bash
$ tmux select-pane <-t pane_id>
# pane id可通过display-panes来知悉
```

### session操作

* 创建session

![image-20210902093923093](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210902093923093.png)

### 常用快捷键

该部分快捷键包含自定义的快捷键

|                    作用                     |  快捷键   |
| :-----------------------------------------: | :-------: |
|            分屏：vertical split             | 前导符+-  |
|             分屏：horizon split             | 前导符+\| |
|             分屏：panel switch              | 前导符+o  |
| 游走(navigation)：panel/window 选择性地切换 | 前导符+w  |
|             粘贴版：显示粘贴板              | 前导符+w  |
|                粘贴板：粘贴                 | 前导符+p  |
|             粘贴板：选择性粘贴              | 前导符+P  |

### 实战

#### [自定义配置](https://github.com/Natsu-Akatsuki/MyTmux)

#### [复制pane的文字](https://blog.csdn.net/RobertFlame/article/details/92794332)

需要在使用前使用 `shift`

#### 安装tpm

安装后，后续可用前导符+I（大写）进行插件安装

```bash
$ git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
```

#### 面板缩放(zoom)

 `前导符+z`

## terminator（终端复用）

* 安装与使用

```bas
$ sudo apt install terminator
$ terminator
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/HLG3YQFJyk39WIM5.png!thumbnail)

* [常用快捷键](https://blog.csdn.net/zhangkzz/article/details/90524066)

## screen（终端复用）

略

## [yakuake（顶部终端）](https://github.com/KDE/yakuake)

暂时感觉用处不大

## [nnn](https://github.com/jarun/nnn)（基于终端的文件管理）

### 应用

#### 显示icon

需要源码编译才能支持该功能

步骤一：安装[icons-in-terminal](https://github.com/sebastiencs/icons-in-terminal#bash-integration)

```bash
$ git clone https://github.com/sebastiencs/icons-in-terminal.git
$ ./install.sh  
$ # Follow the instructions to edit ~/.config/fontconfig/conf.d/30-icons.conf
```

步骤二：[**源码**编译nnn](https://github.com/jarun/nnn/wiki/Advanced-use-cases#file-icons)

```bash
# 安装相关依赖
$ sudo apt install pkg-config libncursesw5-dev libreadline-dev
$ git clone https://github.com/jarun/nnn
$ cd nnn
$ sudo make O_ICONS=1
```

---

**NOTE**

* nnn不是所有版本都有-S(du)的功能
* 其效果有点像`spacevim`，学习曲线较长，暂没从中提高过什么效率

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/oCtqAxAiA9SZmIAd.png!thumbnail)

---

## 终端常用快捷键

* [终端的艺术](https://github.com/jlevy/the-art-of-command-line/blob/master/README-zh.md)
* ShortCut Table

|    快捷键     |           作用           |
| :-----------: | :----------------------: |
|    ctrl+w     |       删除一个单词       |
|    ctrl+7     |     撤销操作(cancel)     |
|    ctrl+u     |        剪切至开头        |
|    ctrl+k     |        剪切至末尾        |
|    ctrl+y     |           粘贴           |
| ctrl+←/ctrl+→ | 以单词为单位进行左右跳转 |
|    ctrl+#     |      注释当前命令行      |

.. attention:: 此处快捷键的剪切板并不是系统的剪切板

## 界面配置

* 界面分为 `命令行界面` 和 `图像化界面`

* `命令行界面` ，又称为 `终端界面`，对应的tty为 `ttyX`

* ubuntu下默认提供6个 `终端界面` 给用户登录，每个终端界面下启动的 `terminal` 对应的tty为 `pts/X`

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/e2wbM5698Gcp7CcW.png!thumbnail)

### 查看某些按键的特殊效果

```bash
$ stty -a
```

### [配置特殊的终端效果](https://www.cnblogs.com/robinunix/p/11635560.html)

一般用在bash脚本中，该选项对应于 `bash [option]`

```bash
# 启动调试模式，输出详细的日志（会标准输出当前执行的命令）
$ set -x
# 若脚本执行有问题，则直接退出脚本
$ set -e
```

### 切换界面

```bash
# 查询当前默认的界面（命令行界面or终端界面）
$ systemctl get-default
# 切换界面(依次为命令行界面和终端界面)
$ systemctl isolate multi-user.target
$ systemctl isolate graphical.target
# 设置默认界面
$ systemctl set-default graphical.target
```

### 终端显示特殊的信息

* 想要在输入密码时，有提示信息，可修改`/etc/ssh/sshd_config`的`Banner`字段
* 想要在登录界面中，添加提示信息，可

```bash
$ sudo apt install landscape-common
# 添加bash文件到/etc/update-motd.d/，其中文件顺序从小到大进行执行
$ ...
```

## 拓展工具

### 录制按键

```bash
$ script <output_file_name>
# 命令行操作
# 结束操作
$ exit
```

### [解析命令行](https://explainshell.com/)

## shell脚本

### 首行配置

```bash
# e.g. 用于指明执行当前脚本的执行器
#!/bin/bash
```

### 特殊参数

```bash
$$：查看当前终端的pid 
$1：取命令行的第1个参数（序号从0开始） 
${@:2} ：取所有的参数，取从第2个开始的所有参数 
$? ：获取上一个命令行返回的exit code
````

### 输入输出流重定向

```bash
# 1>     标准输出重定向 (dafault)
# 2>     标准输出错误重定向 
# 1>&2   标准输出转换为标准输出错误（放置在命令行末尾） 
# 2>&1   标准输出错误转换为标准输出   （放置在命令行末尾）

$ echo "hello" 2> /dev/null
```

### [read 函数](https://linuxcommand.org/lc3_man_pages/readh.html)

```bash
$ read -r -p "Are You Sure? [Y/n] " input 
# -p：输入时显示提示信息
````

### bash option

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/O3qeGIlZbro6Cifs.png!thumbnail)

```bash
# -i：启动交互式的脚本（若没显式制定-i，bash会根据代码是否有IO交互，隐式加上 -i ）
# -v：执行脚本前，先显示脚本内容
# -x：显示正在执行的命令行(commands)和其参数(arguments)
# -e：若有一个命令行返回值为非0则退出脚本
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/vc2ZAhmLzlmLH17y.png!thumbnail" alt="img" style="zoom:67%; " />

### [判断语法](https://www.cnblogs.com/mlfz/p/11427760.html)

使用中括号作为shell的判断式

.. attention:: 注意空格，[空格... 空格]

## 命令行补全

### [hstr](https://github.com/dvorka/hstr)

```bash
$ sudo add-apt-repository ppa:ultradvorka/ppa && sudo apt-get update && sudo apt-get install hstr && hstr --show-configuration >> ~/.bashrc && . ~/.bashrc
```

## 内置命令行

### 查看历史命令行

#### [history](https://zhuanlan.zhihu.com/p/248520994)

* 默认存放数据的文件为 `~/.bash_history`

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/w3AkpBGZgJwA4SJZ.png)

* 使用history时，显示的是命令行 `历史列表` 的内容。此处的 `历史列表` 即 `当前终端执行过的命令` +读取 `~/.bash_history` 得到的历史记录（默认打开终端时读取一次）

*  只有终端 `logout` 后才会将终端输入过的命令行加入到 `~/.bash_history` 中

* 如果不需要等终端 `logout(ctrl+d/exit)` 后才将命令行写入文件中，使得新开一个终端按history就能看到所有终端执行过的命令行），可添加该行到 `~/.bashrc` ；安装了 :ref:`hstr`. 的话，该部分会自动添加。

```bash
# 设置每执行完一个指令后的操作，以下的作用为即时刷新文件内容和更新历史列表
export PROMPT_COMMAND="history -a; history -n; ${PROMPT_COMMAND}"
# option:
a：（写）将历史列表中相对于文件增加的命令行 追加到文件中
n：（读）将文件中相对于历史列表增加的命令行 追加到终端的历史列表中
```

#### whereis和which的区别？

前者搜索范围（database）更广，后者只在 `PATH` 中寻找
