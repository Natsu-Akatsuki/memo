# Sleipnir docker环境

本模块作用：

- 提供纯净的docker ubuntu环境
- 提高测试速度：充分利用服务器，相关编译在服务器上进行
- 依赖追踪：相关依赖记录于Dockerfile中，便于环境复现



## 01. 依赖

[docker](https://docs.docker.com/engine/install/ubuntu/#uninstall-docker-engine)  
[nvidia-container2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)



## 02. dockerfile构建镜像

``` bash
# 相关文件放于tmp/install下
$ gdown https://drive.google.com/uc?id=1Tm1PZnzVNFF0hpWAaCH0inaAI6W3toqs
# 同时把TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz放过去
$ ...
# 在docker目录下构建镜像，构建容器时将共享主机网络
$ docker build -t sleipnir --network=host .
```

## 03. docker构建容器

```bash
# 当前文件夹下执行
$ bash build.sh
```

## 04. 容器的使用

```bash
$ docker exec -it Sleipnir /bin/bash
```

PS：

- 若容器未启动则在终端

```bash
$ docker start Sleipnir
```

- 若需要文件的交互可将文件传到  `/home/ah_chung/chang_ws`  ，相关文件挂载在 `/home/change_ws`  

  

