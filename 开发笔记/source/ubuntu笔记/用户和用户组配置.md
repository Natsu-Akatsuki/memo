# User && Group

## 用户

### 配置文档

* `/etc/passwd` ：与用户名、ID相关
* `/etc/shadow` ：与密码相关

### 增加用户&&修改用户属性

步骤一：增设用户

```bash
# 增设用户
$ useradd -m <user_name> -G <group_name>
# m：建立家目录
# G：将用户添加到某用户组
```

.. attention:: 添加多个用户组时，需使用多次-G

步骤二：修改密码

```bash
$ sudo passwd <user_name>
```

.. hint:: 一般账户对应的ID≥1000，系统账户则<1000；系统账户主要用来执行系统所需的服务，默认不会建立家目录

.. note:: 用sudo能够设置简单密码

步骤三：[初始化家目录](https://askubuntu.com/questions/152707/how-to-make-user-home-folder-after-account-creation)，[提供默认](https://askubuntu.com/questions/404424/how-do-i-restore-bashrc-to-its-default) `.bashrc` ，修改session启动的默认bash

```bash
$ sudo mkhomedir_helper <username>
$ sudo cp /etc/skel/.bashrc /home/<user_name>
$ sudo usermod -s /bin/bash <user_name>
```

步骤四：[给用户追加管理员权限](https://www.tecmint.com/create-a-sudo-user-on-ubuntu/)（optional）

```bash
$ sudo usermod -aG sudo <user_name>
```

### 删除用户

```bash
$ userdel -r <user_name>
# r: 同时删除家目录
```

### 切换用户

switch user

```bash
# 如果是切换到管理员用户时，可以使用sudo su或者su -
$ su <user_name>
```

### 查看用户登录信息

* 查看系统已登录的一般用户(who或者w)，对应文件`/var/run/utemp`

```bash
$ who
# w会显示更详细的信息，包括cpu占用率，占用session所对应的执行程序
$ w
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210906101901670.png" alt="image-20210906101901670" style="zoom:50%; " />

* 查看系统的一般用户

```bash
$ cat /etc/passwd | awk 'BEGIN {FS=":"} $3>=1000 {print $1 "\t" $3}'
```

* 查看用户最近的登录信息，对应文件`/var/log/lastlog`

```bash
$ lastlog
```

* 查看登录成功的日志，对应文件`/var/log/wtmp`

```bash
$ last
```

* 查看用户访问信息（包括密码错误和ssh）

```bash
$ sudo tail -n 100 /var/log/auth.log
```

### 查看当前用户所在用户组

```bash
# 第一字段为有效用户组
$ groups
```

### 临时修改主机名

```bash
$ hostname <new-name>
```

## 用户组

### 增加用户组

```bash
$ groupadd <group_name>
```

### 删除用户组

```bash
$ groupdef <group_name>
```

### 修改用户组属性

```bash
$ groudmod -n [dst_group_name] [src_group_name] -g [dst_id] [src_id]
# -n：修改用户组名
# -g：修改用户组id
```

### 切换当前的有效用户组

```bash
$ newgrp
```

### 修改文件/文件夹的用户组所有者信息

```bash
$ chgrp -R <group_name> <directory/file>
# -R 递归
```

### [常用用户组名说明](https://wiki.debian.org/SystemGroups)

## 权限设置

用户A创建文件或文件夹时，该文件的文件组所有者属性从属于 `有效用户组`

### 给文件夹增加SGID权限

```bash
$ chmod g+s <directory>
```

.. hint:: SGID权限即用户在该文件夹下的有效用户组为该文件夹下的用户组；用户在该文件夹下添加的文件或文件夹，这些实体的用户组权限跟该文件夹用户组权限相同

## 实战

### [Linux服务器遭受黑客攻击时的日志分析排除](https://blog.csdn.net/wxh0000mm/article/details/102948268)
