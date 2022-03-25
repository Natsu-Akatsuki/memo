# ProcessManage

## 查看进程

### ps

```bash
$ ps          # 查看当前终端的进程（BSD）
$ ps (-l)     # 查看当前终端的进程[l:Long format]
$ ps aux      # 显示当前系统中的进程（以PID升序的顺序）
$ ps -ef      # 查看当前系统的进程（含父进程）
$ ps -o ppid (子进程pid) # 查子进程ppid(-o:指定输出项)
```

---

**进程的状态**

```bash
# p525《鸟叔的LINUX私房菜》
# R: running    进程正在运行
# S: sleep      进程正在睡眠状态（IDLE），但可以被唤醒（signal）
# D:            不可被唤醒的睡眠状态（该进程可能在等待IO）
# T: stop       停止状态
# Z: zombie     进程已停止但无法从内存中被删除
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/cfxMqcDd5UPVsw7e.png!thumbnail)

---

### htop

#### [取消显示线程](https://blog.csdn.net/FengHongSeXiaoXiang/article/details/53515995)

- [htop中将线程视为进程，所以会看到多个分配了同样资源的进程](https://superuser.com/questions/118086/why-are-there-many-processes-listed-under-the-same-title-in-htop)，可通过设置进行取消

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/3SrBiGojwbmLfKQq.png!thumbnail)

#### [常用快捷键](https://www.cnblogs.com/lurenjiashuo/p/htop.html)

```bash
# -u(--user)：显示指定用户
# -p(--pid)：显示指定pid
# -t --tree：树状形式显示进程（实际体验绝对pstree比较清晰）
```

## 结束进程

### 根据进程ID来结束

```bash
$ kill <PID>
```

### 根据启动时的命令名进行结束

```bash
$ killall <command>
# -w: 阻塞，直到该进程成功结束
```

---

**NOTE**

此处的command指该字段的第一列命令（因此要关掉roscore则需要`killall /usr/bin/python3`而不是`python`）；在实测过程中 `killall roscore` 也行

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/nE7rfI0LJCdv47bq.png!thumbnail)

---

### [kilall后面应该输出什么样的command？](https://unix.stackexchange.com/questions/14479/killall-gives-me-no-process-found-but-ps)

```bash
# 方法一：参考出来的第二个字段
$ cat /proc/<pid>/stat 
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/aKPaQo2LCUtmPpGl.png!thumbnail)

### 不同信号触发的关闭

```bash
# 用这种方式强制关闭ros launch进程，不会同时关闭其管理的节点进程
$ kill -s 9 <pid>    # 进程终端立即执行（资源回收会不彻底）
$ kill -s 17 <ppid>  # 让父进程回收僵尸进程 -CHLD
```

## 根据进程查文件 / 根据文件查进程

### 列出系统中正使用的文件(list open file)

```bash
$ lsof
$ lsof -u <user_name> # 查看指定用户在使用的文件
$ lsof -p <pid>       # 查看指定进程所使用的文件资源
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/uPRoNIIO1CN9lkti.png!thumbnail)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/z5f7Ms5G4IeSuzUM.png!thumbnail)

## 根据文件/文件夹查进程

```bash
# 根据文件查进程，该命令行等效于fuser的效果
$ lsof <file_name 绝对 or 相对> 
$ fuser <file_name>
# 根据文件夹查进程（大小写区别暂时未详细理解）
$ lsof +d / +D <dir_name>
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ghQWsd2q2yJRozgJ.png!thumbnail)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/MP0WZ9JfX7xRRlYv.png!thumbnail" alt="img" style="zoom:80%;" />

## 根据port查调用方

```bash
$ lsof -i :22
```

## 查看进程树

```bash
$ pstree
# -s：查看指定pid的父进程和子进程
# -u：显示user
# -p：显示pid号
# -T：只显示进程，不显示线程
# -n：使用pid号进行排序
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/3ET7WfGOPSqsNplH.png!thumbnail)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/RcJ69wSDy1VxZhsp.png!thumbnail)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/5BNu7I1emlKg6t91.png!thumbnail)
