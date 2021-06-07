# [CUDA Programming](https://nyu-cds.github.io/python-numba/05-cuda/)

## 问题

1. 如何用numba进行cuda编程？
2. numba支持哪些cuda特性？

## 目标

1. 用numba进行cuda编程
2. 用numba处理cuda线程
3. 理解numba支持cuda内存模型的机制

## 背景

- numba能将一部分Python代码编译为CUDA的核函数和device函数。其一个重要的特性是能够让核函数直接访问numpy数组而极大的方便了核函数的实现。numpy数组作为参数传入核函数时能够自动的实现数据在CPU和GPU的转换（尽管这可能会存在问题）

- numba尚没有实现所有CUDA API，但当前的功能已经足以。

## 术语

- host

  the CPU

- device

  the GPU

- host memory

  the system main memory

- device memory

  onboard memory on a GPU card

- kernel

  `a GPU function`  launched by the host and executed on the device

- device function

  `a GPU function` executed on the device which can only be called from the device (i.e. from a kernel or another device function)

## GPU 配置

- 获取GPU信息

```
from numba import cuda
print(cuda.gpus)
```

- 若没支持cuda的gpu则会有如下报错信息

```
numba.cuda.cudadrv.error.CudaDriverError: CUDA initialized before forking
CudaSupportError: Error at driver init: 
[3] Call to cuInit results in CUDA_ERROR_NOT_INITIALIZED:
numba.cuda.cudadrv.error.CudaDriverError: Error at driver init:
CUDA disabled by user:
```

- 若有则显示

```
<Managed Device 0>
```

- 有多个时，需要进行GPU选择，默认为device 0

```
numba.cuda.select_device( device_id )
```

- This creates a new CUDA context for the selected `device_id`. `device_id` should be the number of the device (starting from 0; the device order  is determined by the CUDA libraries). The context is associated with the current thread. Numba currently allows only one context per thread.



## 没cuda支持的GPU时的仿真配置

If you don’t have a CUDA-enabled GPU (i.e. you received one of the  error messages described previously), then you will need to use the CUDA simulator.  The simulator is enabled by setting the environment variable `NUMBA_ENABLE_CUDASIM` to 1.

### Mac/Linux

Launch a terminal shell and type the commands:

```
export NUMBA_ENABLE_CUDASIM=1
```



## 核函数的书写

CUDA has an execution model unlike the traditional sequential model  used for programming CPUs. In CUDA, the code you write will be executed  by multiple  threads at once (often hundreds or thousands). Your solution will be  modeled by defining a thread hierarchy of grid, blocks, and threads.



Numba 支持三种GPU内存：

- global device memory

- shared memory

- local memory

  

For all but the simplest algorithms, it is important that you  carefully consider how to use and access memory in order to minimize  bandwidth  requirements and contention.

NVIDIA recommends that programmers focus on following those recommendations to achieve the best performance:

- Find ways to parallelize sequential code
- 减少GPU和CPU的数据传递量
- Adjust kernel launch configuration to maximize device utilization
- Ensure global memory accesses are coalesced
- Minimize redundant accesses to global memory whenever possible
- Avoid different execution paths within the same warp



## 核函数的声明

- 核函数就是被CPU调用在GPU执行的函数，有两大特性：
  - 核函数没有返回值，要得到返回结果则需要写到传入的数组中（如果返回结果是标量，则应传一个只有一个元素的数组）
  - 核函数需显式声明它的线程组织结构(thread hierarchy)，即线程块的个数，每个线程块有多少个线程(note that  while a kernel is compiled once, it can be called multiple times with  different block sizes or grid sizes).

```
from numba import cuda

# 要加上这个装饰器
@cuda.jit
def my_kernel(io_array):
    """
    Code for kernel.
    """
    # code here
```



## 核函数的调用

```
import numpy

# Create the data array - usually initialized some other way
data = numpy.ones(256)

# Set the number of threads in a block
threadsperblock = 32 

# 计算每个网格的线程块个数
blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock

# 启动核函数（此处不同于c++用尖括号，此处用方括号）
my_kernel[blockspergrid, threadsperblock](data)

# Print the result
print(data)
```

- 此处有两个重要的步骤
  - 合理地实例化核函数，指定线程格中块的个数，线程块中线程的个数。二者的乘积则为总共启动的线程个数。

1. Kernel instantiation is  done by taking the compiled kernel function and indexing it  with a tuple of integers.
   - 传入数据并运行，默认情况下，核函数的运行是同步的，the function returns when the kernel has finished executing and the data is synchronized back.



### 如何选取线程块的大小

The two-level thread hierarchy is important for the following reasons:

- 软件层面上：线程块决定了多少线程共享一份共享内存
- 硬件层面上：
- On the hardware side, the block size must be large enough for full occupation of execution units; recommendations can be found in the CUDA C Programming Guide.



线程块的尺寸设计取决于：

- 输入数据的大小
- 每个线程块共享内存的大小 (e.g. 64KB)
- 硬件上每个线程块所支持的线程个数 (e.g. 512 or 1024)
- The maximum number of threads per multiprocessor (MP) (e.g. 2048)
- The maximum number of blocks per MP (e.g. 32)
- The number of threads that can be executed concurrently (a “warp” i.e. 32)

The execution of threads in a warp has a big effect on the  computational throughput. If all threads in a warp are executing the  same instruction  then they can all be executed in parallel. But if one or more threads is executing a different instruction, the warp has to be split into  groups of threads, and these groups execute serially.

Rules of thumb for threads per block:

- Should be a round multiple of the warp size (32)
- A good place to start is 128-512 but benchmarking is required to determine the optimal value.

Each streaming multiprocessor (SP) on the GPU must have enough active warps to achieve maximum throughput. In other words, the blocksize is  usually selected to maximize the “occupancy”. See the  [CUDA Occupancy Calculator spreadsheet](http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls) for more details.



## 线程位置

- 线程执行核函数时，需要知道这个核函数是被哪个线程所调用的，以知道该分配哪个数据给它处理。
- 为了处理多维数组，CUDA支持多维的线程格和线程块

In the example above, you  could  make `blockspergrid` and `threadsperblock` tuples of one, two or three integers. 

Compared to 1-dimensional  declarations of equivalent sizes,  this doesn’t change anything to the efficiency or behaviour of generated code, but can help you write your algorithms in a more natural way.



- 实例（获取当前线程在grid和block的位置）

```python
@cuda.jit
def my_kernel(io_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < io_array.size:  # Check array boundaries
        io_array[pos] *= 2 # do the computation
```

以下对象用于获取当前线程的所在位置：

- `numba.cuda.threadIdx` ：当前线程在线程块的索引。The thread indices in the current thread block. For 1-dimensional blocks, the index (given by the x attribute) is an  integer spanning the range from 0 to `numba.cuda.blockDim` - 1.多维亦然
- `numba.cuda.blockDim` ：线程块的大小，在实例时进行指定。
- `numba.cuda.blockIdx` ：线程块在线程格的位置  For a 1-dimensional grid, the index (given by the `x` attribute)  is an integer spanning the range from 0 to `numba.cuda.gridDim` - 1. A similar rule exists for each dimension when more than one dimension is used.
- `numba.cuda.gridDim` - The shape of the grid of blocks, i.e. the total number of blocks launched by this kernel invocation, as declared when  instantiating the kernel.

These objects can be 1-, 2- or 3-dimensional, depending on how the  kernel was invoked. To access the value at each dimension, use the `x`, `y` and `z`  attributes of these objects, respectively.



### 绝对位置

Simple algorithms will tend to always use thread indices in the same  way as shown in the example above. 

Numba provides additional facilities  to  automate such calculations:

- `numba.cuda.grid(ndim)` - 获取当前线程在整个线程格的位置； `ndim` should correspond to the  number of dimensions declared when instantiating the kernel. 
- 若 `ndim=1` ，则返回整数；否则返回一个整型元组
- `numba.cuda.gridsize(ndim)` - Return the absolute size (or shape) in threads of the entire grid of blocks. `ndim` has the same meaning as in  `grid()` above.

Using these functions, the our example can become:

```
@cuda.jit
def my_kernel2(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2 # do the computation
```

在host代码中调用核函数. Notice that  the grid computation when instantiating the kernel must still be done  manually.

- 相关代码如下`cuda1.py`

```
from __future__ import division
from numba import cuda
import numpy
import math

# CUDA kernel
@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2 # do the computation

# Host code   
data = numpy.ones(256)
threadsperblock = 256
blockspergrid = math.ceil(data.shape[0] / threadsperblock)
# 先定每个线程块中线程的个数，再根据计算量计算每个线程格线程块的个数
my_kernel[blockspergrid, threadsperblock](data)
print(data)
```

- 执行程序 (remember to set `NUMBA_ENABLE_CUDASIM=1` if you don’t have an NVIDIA GPU):

```
python cuda1.py 
```

- 可看到如下结果

```
[ 2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.]
```

## 内存管理

- 调用核函数时，numba能够将数据自动转移到device上（但这仅限于核函数执行完成后，将GPU上的数据转移到CPU上；而不支持反向的操作，以避免对只读数组的转移）

```
# 在device上给一个数组分配空间
device_array = cuda.device_array( shape )
# 将host上的数组转移到device上
device_array = cuda.to_device( array )
```



### 实例：矩阵相乘

- 本例中每个线程读矩阵A的行和矩阵B的列并计算出其对应的C矩阵的元素值
- 输入矩阵为`A.shape == (m, n)` ， `B.shape == (n, p)` ，则输出结果为： `C.shape = (m, p)`.

![matrix multiplication](https://nyu-cds.github.io/python-numba/fig/05-matmul.png)

```
@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp
```



- 构建程序 `cuda2.py` 并调用，此处单位线程数和单位线程块数不重要，只是为了有足够的线程进行计算

```
from __future__ import division
from numba import cuda
import numpy
import math

# CUDA kernel
@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp
        
# Host code：在CPU上应该给输入的A,B,C数组分配内存，并传递到GPU中。在核函数执行完成后，需要将C数组传回CPU中。

# Initialize the data arrays
A = numpy.full((24, 12), 3, numpy.float) # matrix containing all 3's
B = numpy.full((12, 22), 4, numpy.float) # matrix containing all 4's

# Copy the arrays to the device
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)

# Allocate memory on the device for the result
C_global_mem = cuda.device_array((24, 22))

# Configure the blocks
threadsperblock = (16, 16)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Start the kernel 
matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)

# Copy the result back to the host
C = C_global_mem.copy_to_host()

print(C)
```

A problem with this code is that each thread is reading from the global memory containing the copies of `A` and `B`. In fact, the `A` global memory is read `B.shape[1]` times and the `B` global memory is read `A.shape[0]` times. 



- 因为global内存较慢，所以本实例不能充分利用GPU



### Shared memory and thread synchronization

A limited amount of shared memory can be allocated on the device to  speed up access to data, when necessary. 

That memory will be shared  (i.e. both readable and writable) amongst all threads belonging to a  given block and has faster access times than regular  device memory. It also allows threads to cooperate on a given solution.  

You can think of it as a manually-managed data cache.

- 创建共享内存

```
shared_array = cuda.shared.array(shape,type)
```

`shape`：整型或整型元组，对应数据的维数

`type`：数组的元素类型

The memory is allocated once for the duration of the kernel, unlike traditional dynamic memory management.

Because the shared memory is a limited resource, it is often  necessary to preload a small block at a time from the input arrays. All  the threads  then need to wait until everyone has finished preloading before doing  the computation on the shared memory.

Synchronization is then required again after the computation to  ensure all threads have finished with the data in shared memory before  overwriting it in the next loop iteration.

- 线程同步的函数，用于对同一个线程块的线程进行同步：直到所有这些线程都调用了这个函数，核函数才会继续

```
cuda.syncthreads()
```



### 实例：优化矩阵相乘

In this example, each thread block is responsible for computing a square sub-matrix of `C` and each thread for computing an element of the sub-matrix.  The sub-matrix is equal to the product of a square sub-matrix of `A` (`sA`) and a square sub-matrix of `B` (`sB`). In order to fit into the device  resources, the two input matrices are divided into as many square sub-matrices of dimension `TPB` as necessary, and the result computed as the  sum of the products of these square sub-matrices.

Each product is performed by first loading `sA` and `sB` from global memory to shared memory, with one thread loading each element of each sub-matrix.  Once `sA` and `sB` have been loaded, each thread accumulates the result into a register (`tmp`). Once all the products have been calculated, the  results are written to the matrix `C` in global memory.

By blocking the computation this way, we can reduce the number of global memory accesses since `A` is now only read `B.shape[1] / TPB` times and `B` is read `A.shape[0] / TPB` times.

![matrix multiplication using shared memory](https://nyu-cds.github.io/python-numba/fig/05-matmulshared.png)

```
from __future__ import division
from numba import cuda, float32
import numpy
import math

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

# The data array
A = numpy.full((TPB*2, TPB*3), 3, numpy.float) # [32 x 48] matrix containing all 3's
B = numpy.full((TPB*3, TPB*1), 4, numpy.float) # [48 x 16] matrix containing all 4's

A_global_mem = cuda.to_device(mat1)
B_global_mem = cuda.to_device(mat2)
C_global_mem = cuda.device_array((TPB*2, TPB*1)) # [32 x 16] matrix result

# Configure the blocks
threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[1]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[0]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Start the kernel 
fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
res = C_global_mem.copy_to_host()

print(res)
```

Create a new program called `cuda3.py` using your new kernel and the host program to verify that it works correctly.





