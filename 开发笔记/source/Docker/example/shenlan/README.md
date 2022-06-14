# README

深蓝学院自动驾驶环境感知课程的docker环境

## Environment

ubuntu 20.04, ros noetic  
cudn11.1, cudnn8   
apt mirrors: ustc   
display manager: gnome 

## 直接拉取镜像

```bash
$ docker pull 877381/shenlan_perception:1.0
```

## 构建镜像

```bash
$ docker-compose --build
```

## 注意项

- 文件挂载位置为：shenlan_perception
- 镜像大小约为20G，包含了一些冗余的安装，可自行修改`Dockerfile`

## 用例

- 启动容器

```bash
# docker-compose.yml目录下启动容器
$ docker-compose up -d
$ bash build_container.sh
```

- 连接容器（密码为: admin）

```bash
# 使用前需安装vnc client（此处使用tigervnc-viewer）
$ sudo apt install tigervnc-viewer
$ vncviewer localhost:15901
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220331171241039.png" alt="image-20220331171241039" style="zoom:67%;" />

