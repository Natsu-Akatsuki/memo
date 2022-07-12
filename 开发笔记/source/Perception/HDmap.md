# HDmap

## Application

## Josm

Java Open StreetMap Editor

### [Install](https://josm.openstreetmap.de/wiki/Download#Ubuntu)

```bash
# 加源
$ echo "deb [signed-by=/usr/local/share/keyrings/josm-apt.gpg] https://josm.openstreetmap.de/apt $(lsb_release -sc) universe" | sudo tee /etc/apt/sources.list.d/josm.list > /dev/null

# Create the directory for manually downloaded keys if it was not already created
$ sudo mkdir -p /usr/local/share/keyrings
# 下载秘钥
$ wget -q https://josm.openstreetmap.de/josm-apt.key -O- | sudo gpg --dearmor -o /usr/local/share/keyrings/josm-apt.gpg

# You may need to install ssl support for apt in advance:
$ sudo apt install apt-transport-https
# 更新源
$ sudo apt update

# You can skip this first line if these packages were not installed before.
$ sudo apt purge josm josm-plugins
# For the tested version
$ sudo apt install josm

# For the development version
# sudo apt install josm-latest
```

## Unity3D

### Install

test in 18.04 / 20.04 / 22.04

- [unity hub](https://docs.unity3d.com/hub/manual/InstallHub.html#install-hub-linux)

```bash
# 导入源
$ sudo sh -c 'echo "deb https://hub.unity3d.com/linux/repos/deb stable main" > /etc/apt/sources.list.d/unityhub.list'
# 导入秘钥
$ wget -qO - https://hub.unity3d.com/linux/keys/public | sudo apt-key add -

$ sudo apt update
$ sudo apt install unityhub

# 日志位置为：~/.config/UnityHub/logs
```
