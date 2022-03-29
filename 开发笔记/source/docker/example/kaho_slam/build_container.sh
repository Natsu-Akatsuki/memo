#!/bin/bash

# set -x # for debug

# 参数配置
set_container_name="--name=slam"
image_name="877381/slam:1.0"

# 文件挂载
set_volumes="--volume=${HOME}/clion:/change_ws:rw"

# 开启端口
# pycharmPORT="-p 31111:22" 
# jupyterPORT="-p 8888:8888" 
# tensorboardPORT="-p 6006:6006"
vncserverPORT="-p 15900:5900"
# set_host="--network=host"

# 设备限制
set_shm="--shm-size=8G"

docker run -it --gpus all \
    ${set_volumes} \
    ${vncserverPORT} \
    ${set_shm} \
    ${set_container_name} \
    $image_name
    
    


