# cuda-practice

## [语法拓展](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions)

### 关键词（identifier）

- `__global__`：，用于告诉编译器，这个函数是在`device`上执行的，返回类型必须是`void`，不支持可变参数参数，不能成为类成员函数。注意用`__global__`修饰的`kernel`(核函数)是 `异步` 的，这意味着`host`(CPU)不会等待`kernel`(GPU)执行完才执行下一步。
- `__device__`：在`device`上执行，仅可以从`device`中调用，不可以和`__global__`同时用。
- `__host__`：在host上执行，仅可以从host上调用，一般省略不写，不可以和`__global__`同时用，但可和`__device__`同时使用，此时函数会在device和host都编译。

### 内置变量

- 内置变量 `blockdim` ：线程块中每一维最大的**线程数**
  - 线程块索引值 `blockdim.x` ；
- 最高是三维线程块 `griddim` ：线程格中每一维的**线程块**数量；

- `griddim`和`blockdim`都是三维

### 调用核函数

```c++
// 创建6个线程块，每个线程块含有18个线程
<<<6,18>>> 
// <<<numBlocks, threadsPerBlock>>>
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/iAyGHWLiRVXXnqtN.png!thumbnail)

### 数据传输（cudaMemcpy）

```c++
#include <cuda.h>
cudaMemcpy (void *dst, const void *src, size_t count, cudaMemcpyKind kind)
// cudaMemcpyKind kind:    
// cudaMemcpyHostToDevice
// cudaMemcpyDeviceToHost
// cudaMemcpyDeviceToDevice
// cudaMemcpyDefault(比较少用)
```

.. note:: 内存传输是同步的。

## 术语

- `cu file`：又称为`kernels`，能并行运行在N卡的处理单元上。`kernels`由 `nvcc` 编译，更多的命令行选项说明可参考[here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html?ncid=afm-chs-44270&ranMID=44270&ranEAID=a1LgFw09t88&ranSiteID=a1LgFw09t88-FBAQRR8XLx9L6QINUdzo9Q#nvcc-command-options)

```bash
$ nvcc <file>
# -o: rename file
```

- `device`：GPU和其内存

- `host`：CPU和其内存

- `kernels`：核函数，在GPU上执行的函数，能在N个GPU线程中并行地执行这个函数

.. note:: 实测printf函数在核函数中无效

- `grid`, `block`, `sm`: 多个线程可以组成一个block，多个block组成一个grid，block由一个sm（流式多处理器）管理

## cuda编程模型（software）

### 线程索引

- 每个线程都在一个线程块(block)中：每个线程都有一个thread ID，对应的内置变量为三维向量`threadIdx`。对于二维线程块： `(Dx, Dy)`，线程(x, y) 的ID为`(x + y Dx)`，对于三维线程块：`(Dx, Dy, Dz)`，线程 (x, y, z) 的ID为 `(x + y Dx + z Dx Dy)`

.. note:: 线程块=线程+共享内存

- 每个线程块都在一个线程格(grid)中：每个线程块都有一个block ID，对应的内置变量为`blockDim`

![image-20220103191259902](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220103191259902.png)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/gridWBlock.png" alt="Thread Indexing and Memory: CUDA Introduction Part 2" style="zoom: 33%;" />

## 实例

### event（测时间）

```c++
float time_elapsed = 0;
cudaEvent_t start, stop;
cudaEventCreate(&start);  // create a event
cudaEventCreate(&stop);

cudaEventRecord(start); // record the current time
// exec kernel function
// T
cudaEventRecord(stop);  // record the current time

cudaEventSynchronize(start);  // wait for an event to complete
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time_elapsed, start, stop);

cudaEventDestroy(start);  // destroy the event
cudaEventDestroy(stop);

std::cout << std::string("Time to calculate results:")
            << time_elapsed << "ms" << std::endl;
```

### 查看gpu硬件信息

```c++
#include "book.h"

int main( void ) {
    cudaDeviceProp  prop;

    int count;
    HANDLE_ERROR( cudaGetDeviceCount( &count ) );
    for (int i=0; i< count; i++) {
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate:  %d\n", prop.clockRate );
        printf( "Device copy overlap:  " );
        if (prop.deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n");
        printf( "Kernel execution timeout :  " );
        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );

        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
        printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
        printf( "Max mem pitch:  %ld\n", prop.memPitch );
        printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

        printf( "   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:  %d\n",
                    prop.multiProcessorCount );
        printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per mp:  %d\n", prop.regsPerBlock );
        printf( "Threads in warp:  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n",
                    prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                    prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                    prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                    prop.maxGridSize[0], prop.maxGridSize[1],
                    prop.maxGridSize[2] );
        printf( "\n" );
    }
}
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/vNGJgcejcflCWcbd.png" alt="img" style="zoom:80%;" />

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220103132111604.png" alt="image-20220103132111604" style="zoom:80%;" />

.. note:: 通常nvidia设备的warp所包含的线程为32，每个线程块最多的线程数为1024

### python extension

- 除将cu文件编译为可执行文件外，还可以基于setup.py进行编译，将其构建为python可调用的拓展库（一些实例可参考 `pcdet` , [pytorch API](https://pytorch.org/docs/stable/cpp_extension.html), [pytorch extension turorial](http://www.smartredirect.de/redir/clickGate.php?u=IgKHHLBT&m=1&p=8vZ5ugFkSx&t=vHbSdnLT&st=&s=&url=https%3A%2F%2Fpytorch.org%2Ftutorials%2Fadvanced%2Fcpp_extension.html%23writing-a-c-extension&r=https%3A%2F%2Fshimo.im%2Fdocs%2FWR8X9kJG9JWYXXjJ)）

- 以下案例节选自[here](https://github.com/Natsu-Akatsuki/memo/tree/master/%E5%BC%80%E5%8F%91%E7%AC%94%E8%AE%B0/source/deep-learning/example/cuda/python-extension)

```bash
# 步骤一：将build/lib*目录下的.so文件copy到python文件的同级目录
$ python setup.py build
# 若想直接在setup.py的当前目录下生成拓展库，直接：
$ python setup.py build_ext -i

# 步骤二：执行程序
$ python python test_extension.py
```

---

**NOTE**

- ImportError: libc10.so: cannot open shared object file: No such file or directory：在python文件中首先导入torch，即 import torch 

- 没有找到ninja：UserWarning: Attempted to use ninja as the BuildExtension backend but we could not find ninja. Falling back to using the slow distutils backend. warnings.warn(msg.format('we could not find ninja.'))

```bash
$ sudo apt-get install ninja-build
```

- The 'compute_35', 'compute_37', 'compute_50', 'sm_35', 'sm_37' and 'sm_50' architectures are depre
  cated, and may be removed in a future release：无法编译通过。一种解决方案是调整cuda的版本（未实测）；一种是使用sm>50的GPU

![image-20220105170824428](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220105170824428.png)

---

## cuda工具

编译器：nvcc

调试器：nvcc-gdb

性能分析：nsight, nvprof, nvvp

函数库：cublas, nvblas, cusolver, cufftw, cusparse

### nvvp/nvprof

cuda自带，用来进行性能分析。如查看加速了多少。

- [nvvp的启动](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#setup-jre)

```bash
# 图形化界面
$ nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
# 命令行工具
$ nvprof <executable file>
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/K3IIhhVe5QptvptJ.png!thumbnail" alt="img" style="zoom:67%;" />

## Q&A

### [为什么线程块中的线程数尽量设计为32？](https://stackoverflow.com/questions/10460742/how-do-cuda-blocks-warps-threads-map-onto-cuda-cores)

![image-20220105095059278](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220105095059278.png)

> GPU指令的执行是以一个block中的32个线程(called warp)为执行单位的；指令执行总数 = 一个线程/一个warp将要执行的指令数 × warp数；不同线程的组织方式，warp的数量也不同。

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220105105148776.png" alt="image-20220105105148776" style="zoom: 50%;" />

比如说有384个线程，每个线程要执行10个指令。可以分配8个线程块，每个线程块48的线程，其对应160个指令；另一方面也可以分配64个线程块，每个线程块6个线程，其对应为640个指令。所以前者的效率会更高。

### [nvidia硬件层级解读](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)#Hardware_perspective)

- GPU由多个SM组成。一个SM能处理多个block。当一个SM接收到一个block时，首先将它们划分为一个warp。处理完一个block再处理下一个block。
- 为了并行执行成百上千的线程，SM采用了SIMT的架构

![image-20220105154645612](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220105154645612.png)

### 显卡计算能力

- [version number](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability)：显卡计算能力用version number来表征。其中首位数字表征框架，如8对应安培架构，7对应伏打架构；第二位数字表征更多的特性。比如图灵架构(7.5)是伏打架构(7.0)的升级版。
- [具体的参数指标](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)

### cuda流如何加速应用程序

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/SmpAVdpZwGzjXfEF.png!thumbnail)（1）理解流水线前传机制，该机制如何使cpu效率显著增加

（2）CPU的三级缓存的特点，哪些内容适合放在哪一级别的缓存上

（3）什么样的问题适合GPU，结合日常编程的任务 ![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/zBugODGh3u7sqNHY.png!thumbnail)（1)GPU控制单元和计算单元是如何结合的？或者说线程束是如何在软件和硬件端被执行。为什么说线程束是执行核函数的最基本单元。



## 参考资料

- [CUDA-by-Example](https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-)
- [官方教程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

- 分配锁页内存(page-locked host memory)
- 学习cuda流，用cuda流(stream)来加速应用程序

- cuda流是一个内存队列，所有的cuda操作(kernels，内存拷贝）都在流上执行
- cuda流有两类，一种是显式流（同步执行），一种是隐式流（异步执行）

## opinion

- 函数拷贝数=线程块个数  p43
- 线程块越多越好还是线程越多越好？ P43
- 在实际测试中，为什么线程块中的线程超过1024后并没有直接的报错（如：段错误）
- 一个SM管理32个线程，这32个线程称为warp
- GPU控制单元简单，没有分支预测和数据转发
- GPU和CPU的区别？前者是以吞吐量（单位时间内执行更多的指令）为导向；后者是以时延（执行一条指令的时间）为导向

## DEBUG

- .cu文件的主函数需要返回 `int` 

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/S3FvLbZPKhCEMjgX.png!thumbnail)

