# docker practice

## [Install](https://docs.docker.com/engine/install/ubuntu/#uninstall-docker-engine)

### docker

步骤一(optional）：若有旧版的docker则进行卸载

```bash
$ sudo apt-get remove docker docker-engine docker.io containerd runc
```

步骤二：

```bash
$ sudo apt-get update
$ sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo apt-key fingerprint 0EBFCD88
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
$ sudo apt-get update
$ sudo apt-get install -y docker-ce docker-ce-cli containerd.io
```

步骤三：[postprocess](https://docs.docker.com/engine/install/linux-postinstall/)

- 不需要使用root权限启动docker

```bash
$ sudo groupadd docker           # 创建一个docker组
$ sudo usermod -aG docker $USER  # 将用户添加到该组中
$ newgrp docker                  # 使配置生效
```

- docker自启动

```bash
$ sudo systemctl enable docker
```

- 检验是否安装成功

```bash
$ docker run hello-world
```

### [docker-compose](https://docs.docker.com/compose/install/)

该工具可用来缩短个人配置开发环境的时间（暂未深入体会这个工具）

- 安装

```bash
$ sudo curl -L "https://github.com/docker/compose/releases/download/1.29.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
$ sudo chmod +x /usr/local/bin/docker-compose
```

- 常用命令行

```bash
# --- 需在docker-compose.yml文件所在目录运行
# 列举compose管理中的容器
$ docker-compose ps  
# 删除compose管理下的容器 -v(删除匿名卷) -f（跳过confirm stage）
$ docker-compose rm   
```

### [ade](https://ade-cli.readthedocs.io/en/latest/install.html#requirements)

- 安装

```bash
$ cd /usr/local/bin
$ sudo wget https://gitlab.com/ApexAI/ade-cli/uploads/f6c47dc34cffbe90ca197e00098bdd3f/ade+x86_64
$ sudo mv ade+x86_64 ade
$ sudo chmod +x ade
$ sudo ade update-cli
```

### [nvidia-container2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

- 安装（或要科学上网）

```bash
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2
$ sudo systemctl restart docker
```

---

**NOTE**

- `Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`：重装nvidia-docker即可（ `apt install` + `重启服务` ）

- `gpg: no valid OpenPGP data found`，[使用代理](https://github.com/NVIDIA/nvidia-docker/issues/1367)

---

## command

### 镜像

```bash
$ docker pull <image_name>      # 从远程仓拉取镜像
$ docker rmi  <image_name>      # 删除镜像
```

### 容器

```bash
# 启动、重启、暂停容器
$ docker start/restart/stop
# 显示正在运行的容器
$ docker ps  (-a 显示所有的容器，包括暂停的)
# 创建容器
$ docker run <image_name>
# 删除容器
$ docker rm  <container_name>
# 删除所有暂停的容器
$ docker container prune
# or q: Only display container IDs
$ docker rm $(docker ps --filter status=exited -q)
# 在已启动的容器中再开一个终端
$ docker exec -it /bin/bash
```

### 构建容器的选项说明

```bash
# --gpus all: 容器可用的GPU ('all' to pass all GPUs)
# --privileged: 提供更多的访问权限
# -t: 在容器中启动一个终端
# -i: 与容器的标准输入进行交互（一般跟-t一起使用）
# -d: 后台运行
```

### 查看docker占用的空间

```bash
$ docker system df
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/3HacQGLIn8pYe8Fp.png!thumbnail)

### 压缩/导出镜像

```bash
# 导出镜像
# docker save sleipnir-trt7.2.3 -o sleipnir-trt7.2.3.tar
$ docker save <image_name> -o <sleipnir-trt7.2.3.tar>
# 导入镜像
$ docker load -i <tar file>
```

## Dockerfile

### 从文件构建容器

```bash
$ docker build .
# option:
# -q:             构建时终端不输出任何信息
# -f:             指定构建时用到到文件名 
# -t:             镜像名 repository/img_name:version 
# --network host: 使用主机的网络模式
# .               Dockerfile文件的所在路径
```

### Dockerfile指令

1. 只有RUN、COPY、ADD才会生成镜像层，[使用基础镜像：FROM](https://docs.docker.com/engine/reference/builder/#from)

2. `ARG` 是唯一可放在 `FROM` 前的参数

3. 重命名： `AS name` to the `FROM` instruction.  

```bash
FROM ubuntu:${DISTRIBUTION} AS lanelet2_deps
```

4. 设置环境变量：ENV

```bash
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# also: ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 
```

5. 设置入口位置：WORKDIR

```bash
# 即设置执行docker exec或run后进入的目录
WORKDIR <dir>
```

6. ADD / COPY 本地文件拷贝

- ADD虽有解压功能，但不是所有都能解压

>官网：need a local tar archive in a recognized compression format (identity, gzip, bzip2 or xz)

- 使用场景：可以离线下载完安装包再copy进入镜像中（Due to the network access problem）

7. [修改容器中的默认用户](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user)

```bash
RUN useradd --no-log-init -m helios -G sudo
USER helios
```

8. 设置入口函数

```bash
ENTRYPOINT ["/bin/bash"]
```

## [template](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#dont-install-unnecessary-packages)

[pcdet](https://github.com/open-mmlab/OpenPCDet/blob/v0.1/docker/Dockerfile)：custom linux环境/cuda环境/cudnn环境/自建pytorch环境

## [阿里云镜像托管](https://cr.console.aliyun.com/cn-hangzhou/instance/repositories)

```bash
# 登录
$ docker login --username=<...> registry.cn-hangzhou.aliyuncs.com
# 拉取
$ docker pull registry.cn-hangzhou.aliyuncs.com/gdut-iidcc/sleipnir:<镜像版本号>
# 推送
$ docker login --username=<...> registry.cn-hangzhou.aliyuncs.com
$ docker tag <ImageId> registry.cn-hangzhou.aliyuncs.com/gdut-iidcc/sleipnir:<镜像版本号>
$ docker push registry.cn-hangzhou.aliyuncs.com/gdut-iidcc/sleipnir:<镜像版本号>
```

## 构建镜像技巧

1. 为减小镜像大小，需要及时删除缓存，例如删除 `apt packages lists`

```bash
$ rm -rf /var/lib/apt/lists/*
```

2. 不需要显式触发apt clean

>Official Debian and Ubuntu images [automatically run](http://www.smartredirect.de/redir/clickGate.php?u=IgKHHLBT&m=1&p=8vZ5ugFkSx&t=vHbSdnLT&st=&s=&url=https%3A%2F%2Fgithub.com%2Fmoby%2Fmoby%2Fblob%2F03e2923e42446dbb830c654d0eec323a0b4ef02a%2Fcontrib%2Fmkimage%2Fdebootstrap%23L82-L105&r=https%3A%2F%2Fdocs.docker.com%2Fdevelop%2Fdevelop-images%2Fdockerfile_best-practices%2F%23dont-install-unnecessary-packages)`apt-get clean`, so explicit invocation is not required.

## docker远程连接服务器(for pycharm)

- 要专业版pycharm

- 假定容器端口已进行了映射  -p  13300<host_port>:22<container_port>

- 容器中需要下载ssh

```bash
$ apt install openssh-server
```

- 修改ssh的配置文件

```bash
# 将PermitRootLogin prohibit-passwd 改为 PermitRootLogin yes
$ vim /etc/ssh/sshd_config
```

- 使配置文件生效

```bash
$ service ssh restart
```

- 设置ssh登录密码

```bash
$ passwd 
```

- (test) 在当前电脑上测试看是否能连通

```bash
$ ssh root@host_ip -p <host_port>
```

- pycharm配置

- 在tools的configuration deployment中配置相关的映射目录
- 没找到相关文件时，看看是不是root path弄错了

### [设置容器自启动](https://www.cnblogs.com/royfans/p/11393791.html)

```bash
# 启动时设置
$ docker run --restart=always
# 已启动时使用如下命令（ps：不是所有配置都能update）
$ docker update --restart=always <container_id>
```

## DEBUG

- /usr/bin/dockerd 文件缺失，需重新安装docker

```bash
# Uninstall the Docker Engine, CLI, and Containerd packages:
$ sudo apt-get purge docker-ce docker-ce-cli containerd.io
# reinstall docker
# ...
```

- [D-Bus not built with -rdynamic so unable to print a backtrace](https://answers.ros.org/question/301056/ros2-rviz-in-docker-container/)
  - [即通过升级权限，使用privileged](https://shimo.im/docs/h6qXyV9PkwKy9Gdv#anchor-Fd7q)来规避问题

- Invalid MIT-MAGIC-COOKIE-1 keyError

之前还能显示rviz，现在则显示如上报错，尝试重启电脑
