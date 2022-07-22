# ProgrammingModel

## c++程序构建过程

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/d152ab606e08516d8369211b19da87fc29998.png@912w_155h_80q)

* `预处理器` 将.c/.cpp源代码进行 `预处理` (preprocess)得到.i文件；预处理包括**宏替换**、**导入头文件**、**去除注释**等行为
* `编译器` 对.i源文件进行 `编译操作` (compile)得到含**汇编指令**的.s汇编代码(assemble code)；该**步骤会进行语法检查**，但**不会发现逻辑错误**；汇编代码可用vim打开

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210917222609207.png" alt="image-20210917222609207" style="zoom:67%; " />

* `汇编器` 对.s文件进一步 `汇编操作` (assemble)得到.o 二进制目标(object)文件
* `链接器` 对.o文件进行 `链接` (link)得到可执行文件

对应的的命令行(e.g. gcc):

```bash
$ gcc -E <>.c -o <>.i
$ gcc -S <>.i -o <>.s
$ gcc -c <>.s -o <>.o
$ gcc <>.o -o <> # 无参数
# 一步到位
$ gcc -o <output_file_name> <input_file_name>
```

.. note:: 符号未定义、符号重定义 `` duplicate symbol `` 是链接时期的错误

其他常用编译选项：

```bash
-I：指定头文件目录 （e.g. -I.. -I ..可加空格也可不加空格，可相对路径）
-O：是否进行优化（0<1<2<3，其中0为不优化；1为default优化等级；3为最高级别的优化）
-D：指定宏（e.g. -D DEBUG，等效于程序里面的 #define DEBUG)
-L: 指定库目录搜素路径
-Wall：是否输出警告信息
-g：在生成的代码中添加调试信息，配合gdb调试工具使用
-std：设置c++标准（e.g. -std=c++14）
-fPIC: Position-Independent Code
-rpath: 将动态库路径写到可执行文件中(hard code)，会hide LD_LIBRARY_PATH的效果
```

## 动态库查询

```bash
# 查看一个target文件运行时需要链接的动态库
$ ldd <file>
# 查看一个正在运行的进程所链接的动态库
$ pldd <pid>
```

![image-20210916224735535](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210916224735535.png)

[动态库实战](http://cn.linux.vbird.org/linux_basic/0520source_code_and_tarball_5.php)

* 动态库的配置文档为 `etc/ld.so.conf` ，具体又引用了`/etc/ld.so.conf.d` 下的文件

```bash
# 详细查看已有的动态链接信息
# 该配置文档是有关加载到内存的动态链接库而非所有动态链接库
# ldconfig: configure dynamic linker run-time bindings
$ sudo ldconfig -v
```

* [改错(1)](https://askubuntu.com/questions/272369/ldconfig-path-lib-x86-64-linux-gnu-given-more-than-once?rq=1)

>/sbin/ldconfig.real: Path `/lib/x86_64-linux-gnu' given more than once

* strings([printable character](http://facweb.cs.depaul.edu/sjost/it212/documents/ascii-pr.htm)：应该就是指ASCII码)：输出一个文件（可二进制）的printable characters

## ccache

### 常用命令行

```bash
# 安装(ubuntu 20.04 version 3.7.7)
$ sudo apt install ccache
# 指定最大缓存量
$ ccache -M 1G
# 清除缓存
$ ccache -C
```

* [源码安装](https://github.com/ccache/ccache/blob/master/doc/INSTALL.md)

```bash
$ sapt install libhiredis-dev asciidoctor

$ wget -c https://github.com/ccache/ccache/releases/download/v4.6/ccache-4.6.tar.gz 
# 解压缩和路径跳转
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
```

* cmake

```cmake
find_program(CCACHE_FOUND ccache)
 if(CCACHE_FOUND)
 set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
 set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)
```

## Q&A

### [c++中的翻译单元是什么？](https://stackoverflow.com/questions/1106149/what-is-a-translation-unit-in-c)

According to [standard C++](http://www.efnetcpp.org/wiki/ISO/IEC_14882) （[wayback machine link](http://web.archive.org/web/20070403232333/http://www.efnetcpp.org/wiki/ISO/IEC_14882)）: A translation unit is the basic unit of compilation in C++. It consists of the contents of **a single source file**, plus the contents of any header files directly or indirectly included by it, minus those lines that were ignored using conditional preprocessing statements.

### [precompile source file的#号是什么？](https://stackoverflow.com/questions/25137743/where-do-we-use-i-files-and-how-do-we-generate-them)

一种特殊的注释

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211002140045353.png" alt="image-20211002140045353" style="zoom:67%; " />

### [编译耗时优化原理和实践](https://tech.meituan.com/2020/12/10/apache-kylin-practice-in-meituan.html)

### 为什么这样写不会发生符号重定义？

#### 相关文件

* A.cpp

```c++
int num_A1 = 0;
int num_A2 = 2;
```

* B.cpp

```c++
int num_A1 = 5;
extern int num_A2;
int main() {
  // num_A2 = num_A2 + 1;
}
```

* CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.11)
project(project)

set(CMAKE_BUILD_TYPE DEBUG)

add_library(B B.cpp)
add_executable(A A.cpp)
target_link_libraries(A B)
```

#### 实验

* 通过实验证明，最终生成的可执行文件并不包含`A.cpp/num_A1`这个变量。即通过对比使用和不使用 target_link_libraries(A B) 的可执行文件A的符号信息（nm命令行）来判别是否一致。

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/4SdVT5emzTdgn0IV.png!thumbnail)

* 另外：**A.cpp**一旦显式触发用上 `B.cpp` 那边的符号之后就会成功触发报错

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/cdvZSiNGWfQYhUWl.png!thumbnail)

.. note:: 另外如果使用的是动态库的话，反而可以顺利通过编译

.. note:: 测试平台windows/ubuntu20.04 g++9.4.0

#### 结论

暂无权威信息佐证，以下均为基于实验的猜测：

（1）静态库把所有symbol都加到target
（2）动态库是把只需要的symbol加到target
（3）对静态库链接时，如果target不需要静态库的任何symbol那`链接器ld`就干脆不导入静态库的任何symbol；但凡有参考的话，就会触发添加所有的symbol

### c++的编译特点

不同于一些高级语言，它们的编译单元是整个模块，**c++的编译单元是以文件为单位**。每个.c/.cc/.cxx/.cpp源文件是一个独立的编译单元，所以优化时只能基于当前文件进行优化，而不能从模块（多个文件）的角度进行优化。

![image-20220403135300524](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220403135300524.png)

* 在add_executable的文件不是合在一起进行编译的，而是依然基于文件单元进行编译的

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220416163848215.png" alt="image-20220416163848215" style="zoom:50%;" />

### 查看符号表

* nm

``` bash
# 可查看object/target的
$ nm
# -c: demangle
# -l：显示对应的行号
```

* readelf

```bash
# 查看符号的可见性
$ readelf -s B
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220416175011251.png" alt="image-20220416175011251" style="zoom: 80%;" />

.. note:: 每个.o文件都有一个符号表，符号有两种分类，一个是全局符号，一个是本地符号，本模块的非静态函数和全局变量都是其他模块可见和可用的；静态函数和静态变量都是只有本模块可见的，其他模块不可使用
