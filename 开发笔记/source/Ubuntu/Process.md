# Process

## Status

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

```bash
$ htop
# -u(--user)：显示指定用户
# -p(--pid)：显示指定pid
# -t --tree：树状形式显示进程（实际体验绝对pstree比较清晰）
```

#### [取消显示线程](https://blog.csdn.net/FengHongSeXiaoXiang/article/details/53515995)

- [htop中将线程视为进程，所以会看到多个分配了同样资源的进程](https://superuser.com/questions/118086/why-are-there-many-processes-listed-under-the-same-title-in-htop)，可通过设置进行取消

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/3SrBiGojwbmLfKQq.png!thumbnail)

- htop界面选项

| 功能键  |                 作用                 |
| :-----: | :----------------------------------: |
| N P M T | 基于PID / CPU% / MEM% / TIME进行排序 |
|    t    |               看目录树               |

![image-20220620100427948](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220620100427948.png)

### 根据进程查文件 / 根据文件查进程

#### 列出系统中正使用的文件(list open file)

```bash
$ lsof
$ lsof -u <user_name> # 查看指定用户在使用的文件
$ lsof -p <pid>       # 查看指定进程所使用的文件资源
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/uPRoNIIO1CN9lkti.png!thumbnail)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/z5f7Ms5G4IeSuzUM.png!thumbnail)

#### 根据文件/文件夹查进程

```bash
# 根据文件查进程，该命令行等效于fuser的效果
$ lsof <file_name 绝对 or 相对> 
$ fuser <file_name>
# 根据文件夹查进程（大小写区别暂时未详细理解）
$ lsof +d / +D <dir_name>
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ghQWsd2q2yJRozgJ.png!thumbnail)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/MP0WZ9JfX7xRRlYv.png!thumbnail" alt="img" style="zoom:80%;" />

### 根据port查调用方

```bash
$ lsof -i :22
```

### 查看进程树

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

## Kill

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

## Terminology

### [僵尸进程](https://en.wikipedia.org/wiki/Zombie_process)

- 僵尸进程是一个调用了 `exit` system call的**子进程**，但仍有一些资源（entry）保留（e.g pid, exit status）在进程表（process table）中。要真正的结束这些进程（即要回收在进程表中的剩下的资源），需要父进程读取子进程的exit status，然后去调用 `wait` system call，来将这些资源从进程表中被移除（这个过程称之为"reaped"）
- 僵尸进程不能够通过发 `kill -s 9/15` 来结束（可以理解为已经被kill了，再kill也没用），只能由父进程对它进行回收处理。可以发 `-17(CHLD)` 信号给僵尸进程的父进程让其回收僵尸进程。（但在实测中不一定能奏效，可能是应用程序没有写好，接收到信号后不会调用wait()）
- 僵尸进程是一个正常进程结束的必经状态。正常进程->exit->僵尸进程->父进程wait->所有资源释放成功

### [孤儿进程](https://en.wikipedia.org/wiki/Orphan_process)

- 孤儿进程指失去原来父进程（父进程已经完成或终止），但仍在运行的子进程。这些进程最后都会被`init`进程管理

### 前后台、守护进程

- 前/后台进程：占用/不占用终端的进程
- 守护进程：一种特殊的后台进程，父进程为systemd（真正脱离了终端（detached），不能放置于前台）

### [SID](https://unix.stackexchange.com/questions/18166/what-are-session-leaders-in-ps)

SID（session id）和GID（group id）都是进程的一个组织单位(unit)，适用于管理进程。比如session leader关掉后，其余的sid一样的进程都会关闭。具体是下发一个`SIGHUP`的信号进行管理。

## Q&A

### 为什么用bash执行含conda命令的脚本时会报错？

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/6CvGagEbvUQqRC9A.png!thumbnail" alt="img" style="zoom:67%;" />

- [自定义变量/函数不会被子进程继承，环境变量才能继承](https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script)

  <img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ca4PGYUdSsJbQJLb.png!thumbnail" alt="img" style="zoom:67%;" />

- 如何判断命令是外部命令还是内部命令？

<img src="https://ugcimg.shimonote.com/uploader-cache/IZaYkLbmNuEqSzbp.png/1ed77e1f65372daaaca3552f86ebdd71_sm_xform_image?auth_key=1655702700-wQC3aXguZi84F49s-0-3439f5c4d9beac9d9ae3bc586967c5f3&response-content-disposition=inline%3B+filename%3D%22image.png%22%3B+filename%2A%3DUTF-8%27%27image.png" alt="img" style="zoom:67%;" />

### 为什么kill/killall没有效果？

默认是发`-15`的信号，但这个信号可以被程序选择性忽略；所以可以使用`-9`来强制结束进程

### [fork twice的作用？](https://stackoverflow.com/questions/10932592/why-fork-twice)

- 让`init`管理子进程，从而让exit()后的子进程（i.e. 僵尸进程）能够及时地被处理

- 假定有两个进程处理任务，一个是父进程，一个是子进程，**父进程处理的时间比子进程的处理时间要长**。子进程exit()成为僵尸进程后，父进程需要一段时间才能执行wait()来处理子进程，也就是僵尸进程会持续一定的时间。因此可以forking两次，将子孙（grandson）节点孤儿化，交由`init`来管理，那就能及时地处理僵尸进程

### 进程和线程的优点和不足

- 需要更多的内存  / 更少的内存使用量
- 父进程先于子进程关闭，则子进程会成为**孤儿进程**(应该是孤儿进程) / 进程关闭后，所有线程将关闭
- 进程的数据交互需要更大的代价 / 共享内存，数据交互开销更小
- 进程间的虚拟内存空间是独立的；线程共享内存，需要解决并发时的内存问题
- 需要进程间通信；可以通过队列和共享内存进行通信
- 创建和关闭相对较慢  / 相对更快
- 更容易debug和写代码 / debug和写代码较难
