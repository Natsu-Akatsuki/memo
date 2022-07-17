# JetBrains

## [PyCharm](https://www.jetbrains.com/help/pycharm/installation-guide.html#toolbox)

### Install

### Debug

#### 多线程

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/7u9B4RAD0DKlb2J7.png!thumbnail)

.. note:: 假定有线程A，B。当给线程A对应的函数加断点时，执行到断点时会跳转到该线程进行断点调试；此时线程A虽然阻塞，但线程B能够继续被执行下去。若此时再给B对应部分加断点，线程B也会阻塞。

- 可视化查看当前程序启动的线程数和线程状态(concurrency diagram)
- ![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/js7MR5uwACpReKRc.png!thumbnail)

#### 前台进程

- [对一个正在运行的程序进行debug](https://www.jetbrains.com/help/pycharm/attaching-to-local-process.html)：attach后可像debug模式下在程序对应位置上打断点

### Performance

#### The current inotify watch limit is too low

```bash
$ sudo sysctl fs.inotify.max_user_watches=524288
```

### Tool

#### [Proxy](https://www.jetbrains.com/help/pycharm/settings-http-proxy.html)

#### Matplotlib

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/YCrevUSseWHvSjCM.png!thumbnail" alt="img" style="zoom:67%;" />

#### Profile

![image-20210908211513561](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210908211513561.png)

#### [External tools](https://www.jetbrains.com/help/pycharm/configuring-third-party-tools.html?q=exter)

- [autopep8](https://www.cnblogs.com/aomi/p/6999829.html)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/iP4LwM5UFypOKsrZ.png!thumbnail" alt="img" style="zoom:67%;" />

```bash
# 需要装在系统中，否则要写可执行文件的绝对路径
Programs: autopep8
Arguments: --in-place --aggressive --aggressive $FilePath$
Working directory: $ProjectFileDir$
Output filters: $FILE_PATH$\:$LINE$\:$COLUMN$\:.*
```

- isort

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/mdmrBwjYhSDwtFsB.png!thumbnail)

- black

```bash
Programs: black
Arguments: $FileDir$/$FileName$
Working directory: $ProjectFileDir$
```

#### Deployment

- 屏蔽某些需要同步的文件和文件夹

方法一：

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/5uEicm5ALtL9tkgh.png" alt="img" style="zoom:67%;" />

方法二：

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/qdPFiJjg6S2slAkU.png)

方法三：

`remote host` 界面中对相关文件和文件夹，右键`exclude path`

#### Interpreter

- 添加额外的库搜索路径

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/spqZAYN9kdaQPJOr.png)

#### Smart key

- 自动补加 `self`
- 生成document时是否添加 `type`

### View

#### 滚轮缩放界面字体

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/wpnajyQeSVpUydTf.png!thumbnail)

#### 只显示已打开文件所对应的文件夹

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210929103600049.png" alt="image-20210929103600049" style="zoom:67%;" />

#### [查看异同](https://www.jetbrains.com/help/pycharm/differences-viewer-for-folders.html)

- 可视化界面

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/KKkanOtkhaJ5sBJI.png!thumbnail)

- 命令行

```bash
# where path_1 and path_2 are paths to the folders you want to compare. 
$ <path to PyCharm executable file> diff <path_1> <path_2> 
```

#### 切换缩进方式

- Edit | convert indents / 或使用Action

#### 设置文档类型

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/XDqHzyb0V7RHrda2.png!thumbnail" alt="img" style="zoom:67%;" />

- rst文档格式

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/RdUVs7HBrZHoUxk7.png!thumbnail)

- Epytest文档格式

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/hlQHvIcUNrHfKSJ6.png!thumbnail" alt="img" style="zoom:67%;" />

## [CLion](https://www.jetbrains.com/help/clion/installation-guide.html)

### [Install](https://www.jetbrains.com/help/clion/uninstall.html#standalone)

### Debug

#### [前台进程](https://www.jetbrains.com/help/clion/attaching-to-local-process.html#attach-to-local)

#### [ROS](https://www.jetbrains.com/help/clion/ros-setup-tutorial.html)

#### [Valgrind](https://www.jetbrains.com/help/clion/memory-profiling-with-valgrind.html#start)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211002014651471.png" alt="image-20211002014651471" style="zoom:67%; " />

#### [Perf](https://www.jetbrains.com/help/clion/cpu-profiler.html)

- 安装

``` bash
# 安装 perf
$ suto apt install inux-tools-$(uname -r)
# 调整内核选项（以获取调试信息），以下选项的设置为永久生效
$ sudo sh -c 'echo kernel.perf_event_paranoid=1 >> /etc/sysctl.d/99-perf.conf'
$ sudo sh -c 'echo kernel.kptr_restrict=0 >> /etc/sysctl.d/99-perf.conf'
$ sudo sh -c 'sysctl --system'
```

![image-20220214144648235](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220214144648235.png)

### Performance

#### [提高IDE性能](https://www.jetbrains.com/help/clion/performance-tuning-tips.html)

包括增加可用内存、提高代码解析和索引的速度、关闭不必要的插件

### Edit

#### Template

- 使用[surround template](https://www.jetbrains.com/help/clion/template-variables.html#pdtv)

![image-20220222100921276](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220222100921276.png)

#### [Head Comment](https://www.dyxmq.cn/program/turning-off-created-by-header-when-generating-files-in-clion.html)

#### Comment Position

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220426100000489.png" alt="image-20220426100000489" style="zoom:50%;" />

#### [Naming](https://www.jetbrains.com/help/clion/naming-conventions.html)

![image-20220304101835683](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220304101835683.png)

### Tool

#### Toolchains

- 自定义工具链（用什么generator，compiler，debugger）

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210815202103721.png" alt="image-20210815202103721" style="zoom:50%; " />

- 配置cmake编译参数

  <img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210815202247964.png" alt="image-20210815202247964" style="zoom: 50%; " />

#### [Doxygen](https://www.jetbrains.com/help/clion/creating-and-viewing-doxygen-documentation.html)

- [效果](https://www.jetbrains.com/clion/features/code-documentation.html)

### View

#### ANSI

CLion的终端无法显示ANSI：正常，CLion的console不是终端，暂时不支持ANSI渲染

#### Docker

- 参考资料：[docker配置](https://www.jetbrains.com/help/clion/docker-connection-settings.html)、[视频资料](https://blog.jetbrains.com/clion/2021/12/clion-2021-3-remote-debugger-docker/#docker_and_other_toolchain_updates)、[docker CLion docs](https://www.jetbrains.com/help/clion/clion-toolchains-in-docker.html#build-run-debug-docker)

## Plugin

- [plugin for CLIon](https://www.jetbrains.com/help/clion/managing-plugins.html)

| 插件名                       | 功能                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| Awesome console              | 触发单击文件链接以打开文件                                   |
|                              |                                                              |
| OpenCV Image Viewer          | [根据数组显示图像](https://plugins.jetbrains.com/plugin/14371-opencv-image-viewer) |
| CodeGlance3 / CodeGlance Pro | [代码缩略图](https://github.com/vektah/CodeGlance)（类似vscode右侧浏览栏） |
| Rainbow Brackets             | 高亮括号                                                     |
| Github Copilot               | 代码AI补全                                                   |
| .ignore                      | 导入ignore文件模板                                           |

## Shortcut

### Navigation

|             作用              |           快捷键            |
| :---------------------------: | :-------------------------: |
|         **括号**折叠          |       ctrl+[shift]+-        |
|           括号跳转            |     ctrl+shift+m(match)     |
|        **代码块**跳转         |       ctrl+[ / ctrl+]       |
|         **书签**跳转          |     ctrl+num(F11创标签)     |
|     **ERROR/WARNING**跳转     | F2(next) / shift+F2(before) |
|        **标签页**跳转         |         alt+←/alt+→         |
| last / next **edit location** |  (custom) alt+光标上下滚轮  |
|        show in Dolphin        |      ctrl+shift+alt+2       |
|         打开**文件**          |            c+s+n            |

### Selection

|           作用           |            快捷键             |
| :----------------------: | :---------------------------: |
| expand current selection | ctrl+w / (redo)  ctrl+shift+w |
|     column selection     |       ctrl+shift+insert       |

### Refactor

|    作用    |  快捷键  |
| :--------: | :------: |
|  修改签名  |  ctrl+6  |
| 修改变量名 | shift+F6 |

### Edit

|                作用                 |           快捷键           |
| :---------------------------------: | :------------------------: |
|           replace in path           |            c+r             |
| replace in files（可设置File mask） |           c+s+r            |
|  Code Complete（偏向语法上的补全）  |         c+s+enter          |
|             选择性粘贴              |           c+s+v            |
|          代码块折叠与展开           |    c+"+/-" / c+s+"+/-"     |
|            live template            |            c+j             |
|          surround template          | ctrl+alt+a(custom) / c+a+j |
