# README

## Environment

ubuntu 18.04, ros melodic  
cudn11.1, cudnn8  
apt mirrors: aliyun  
pcl 1.8  
ceres 2.0  
sophus(github latest)  
g2o(github 20200410_git)  
gtsam(apt)  
prtobuf: 3.15  
display manager: lxde

## Pull Imgae

```bash
$ docker pull 877381/slam:1.0
```

## Create Image

For person who want to build from scratch. You could try as follows.

```bash
$ bash build_image.sh
```

---

**Attention**

Don't forget to modify the image name in file "build_container.sh"

---

## Usage

- build and enter the container

```bash
$ bash build_container.sh
```

- use vncserver

after enter the container, you could launch the vnc server

```bash
(container root) $ vncserver --localhost no :0
```

for client(the login password is `admin`) : 

```bash
# if vncviewer doesn't install, try
$ sudo apt install tigervnc-viewer
$ vncviewer localhost:15901
```
