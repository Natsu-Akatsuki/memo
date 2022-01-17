# README

## [pytorch cmake module](https://pytorch.org/cppdocs/installing.html#minimal-example)

- torch cmake 位置

```bash
$ cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```

![image-20220113095316608](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220113095316608.png)

- torch.cmake自带cuda, cudnn的检测

![image-20220113101837929](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220113101837929.png)