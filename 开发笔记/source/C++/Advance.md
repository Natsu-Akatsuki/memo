# Advance

## Alignment Issue

- 指令的操作数的地址没有对齐，则会报错（段错误），例如AVX指令集需要32位内存对齐（即该操作数的地址需要被32整除）

![image-20220811094137580](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220811094137580.png)

- 指定变量的内存对齐

```c++
// e.g. 指定分配的内存地址能被32整除
// 等价于： __attribute__ ((aligned (32))) double input1[4] = {1, 1, 1, 1};
alignas(64) double input1[4] = {1, 1, 1, 1};
```

- 查看当前CPU支持的指令集

```bash
$ cat /proc/cpuinfo
```

## Reference

- [从Eigen向量化谈内存对齐](https://zhuanlan.zhihu.com/p/93824687)
- [ROS Eigen Alignment issue](http://library.isr.ist.utl.pt/docs/roswiki/eigen(2f)Troubleshooting.html)
