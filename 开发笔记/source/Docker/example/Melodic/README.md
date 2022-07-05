# Melodic

## 环境

- ubuntu 18.04
- ros melodic
- cudnn 11.1, cudnn8
- pcl 1.8
- ceres 2.1
- sophus (github latest)
- g2o (github 20200410_git) 
- gtsam (apt)  
- protobuf: 3.15
- display manager: lxde

### 创建镜像

```bash
# 搭建基础镜像
$ bash build_image.sh
# 镜像改名
$ docker tag helios:melodic-1.0 877381/helios:melodic-1.0
```

### 用例

- 构建容器

```bash
$ bash build_container.sh
# 在容器中启动vncserver
(helios) $ vncserver
# 若容器关闭再启动，则需要移除原先的配置再执行一次vncserver
$ vncserver --kill :0
$ vncserver
```

- 主机端启动vncviewer

```bash
# 需要依赖tigervnc-viewer
$ sudo apt install tigervnc-viewer
$ vncviewer localhost:25900
```

### 日志

- 2022.07.05：kaho工程基础环境搭建
