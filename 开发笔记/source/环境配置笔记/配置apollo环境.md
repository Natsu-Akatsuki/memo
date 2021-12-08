# apollo环境配置

1. [预安装依赖环境](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/prerequisite_software_installation_guide.md)

* ubuntu 18.04 / 20.04
* docker engine (＞19.03)
* nvidia driver
* nvidia cuda toolkit
* nvidia docker

2. [安装](https://github.com/ApolloAuto/apollo/blob/master/docs/quickstart/apollo_software_installation_guide.md)（构建镜像、配置和启动容器）

```bash
# 进入容器
$ bash docker/scripts/dev_into.sh
```

3. [编译apollo](https://github.com/ApolloAuto/apollo/blob/master/docs/howto/how_to_launch_and_run_apollo.md#run-apollo)

```bash
$ ./apollo.sh build_opt_gpu
```
