#!/bin/bash

sudo apt update
sudo apt install -y \
    htop \
    wondershaper \
    ethstatus

sudo apt install -y \
    htop \
    wondershaper \
    ethstatus

# 安装文件夹跳转工具jump
wget https://github.com/gsamokovarov/jump/releases/download/v0.40.0/jump_0.40.0_amd64.deb && sudo dpkg -i jump_0.40.0_amd64.deb
eval "$(jump shell)" >> ~/.bashrc

# 配置语法高亮版cat
# echo 'alias bat="batcat --paging=auto"' >> ~/.bashrc
# echo "export MANPAGER=\"/bin/bash -c 'col -bx | bat -l man -p'\"" >> ~/.bashrc

# 安装zenith
cd ~/application
wget -c https://github.com/bvaisvil/zenith/releases/download/0.12.0/zenith_0.12.0-1_amd64.deb
sudo dpkg -i zenith_0.12.0-1_amd64.deb