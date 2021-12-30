# cuda-practice

## 语法 

### 关键词（identifier）

- `__global__`：，用于告诉编译器，这个函数是在`device`上执行的，返回类型必须是`void`，不支持可变参数参数，不能成为类成员函数。注意用`__global__`修饰的`kernel`(核函数)是 `异步` 的，这意味着`host`(CPU)不会等待`kernel`(GPU)执行完才执行下一步。
- `__device__`：在`device`上执行，仅可以从`device`中调用，不可以和`__global__`同时用。
- `__host__`：在host上执行，仅可以从host上调用，一般省略不写，不可以和`__global__`同时用，但可和`__device__`同时使用，此时函数会在device和host都编译。

### 内置变量

- 内置变量 `blockdim` ：线程块中每一维最大的**线程数**
  - 线程块索引值 `blockdim.x` ；
- 最高是三维线程块 `griddim` ：线程格中每一维的**线程块**数量；最高是二维线程格

- `griddim`和`blockdim`都是三维

## 术语

`cu file`：又称为`kernels`，能并行运行在N卡的处理单元上。`kernels`由 `nvcc` 编译

```
$ nvcc <file>
#
```



`device`：GPU和其内存

`host`：CPU和其内存

`kernel function`：核函数，在GPU上执行的函数，能在N个GPU线程中并行地执行这个函数





# Day1 第三章

- 编写一段`CUDA` code
- 认识`host`和`device`代码的去呗
- 如何从`host`上运行`device`代码
- 使用`device memory`
- 查询系统支持的`CUDA device`信息



- 函数名<<<num1,num2>>>(arg1, arg2);  中各个参数的作用

  `num1`,`num2`用于告诉编译器，运行时如何启动在GPU上运行的`device function/ kernel`；num1表示在device执行核函数时的并行`block线程块`的个数；num2表示在线程块中启动的线程个数

  `arg1`, `arg2` 是运行时传递给kernel的参数

- 如何知道当前的函数拷贝是在哪一个线程块上的？

  使用内置变量 `blockIdx`，包含了执行kernel的线程块的索引



- 线程块的每一维度的最大数量不超过65 535



# Day2 第四章

- 了解CUDA C中的线程
  - 向量相加中，如何计算每个并行线程的初始索引和确定其递增量 
- 不同线程间的通信机制
- 并行执行线程的同步机制



GPU上的每个线程逻辑上都是可以并行执行的



## 使用线程块和使用线程的区别？

线程块中的`并行线程`能完成`并行线程块`无法完成的工作



