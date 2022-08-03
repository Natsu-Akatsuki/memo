# ProgrammingModel

## Programming Model

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

## Compile Feature

c++的编译特点：不同于一些高级语言，它们的编译单元是整个模块，**c++的编译单元是以文件为单位**。每个.c/.cc/.cxx/.cpp源文件是一个独立的编译单元，所以优化时只能基于当前文件进行优化，而不能从模块（多个文件）的角度进行优化。

![image-20220403135300524](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220403135300524.png)

* 在add_executable的文件不是合在一起进行编译的，而是依然基于文件单元进行编译的

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220416163848215.png" alt="image-20220416163848215" style="zoom:50%;" />

## Dynamic Library

### CLI

```bash
# 查看一个target文件运行时需要链接的动态库
$ ldd <file>
# 查看一个正在运行的进程所链接的动态库
$ pldd <pid>
```

![image-20210916224735535](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210916224735535.png)

* 动态库的配置文档为 `etc/ld.so.conf` ，具体又引用了`/etc/ld.so.conf.d` 下的文件

```bash
# 详细查看已有的动态链接信息
# 该配置文档是有关加载到内存的动态链接库而非所有动态链接库
# ldconfig: configure dynamic linker run-time bindings
$ sudo ldconfig -v
```

* [/sbin/ldconfig.real: Path /lib/x86_64-linux-gnu given more than once](/sbin/ldconfig.real: Path /lib/x86_64-linux-gnu given more than once)

## File

### File Type

`stripped`：说明该文件的符号表信息已被去除

[![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/9601UO7szhc9gPdn.png!thumbnail)](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/9601UO7szhc9gPdn.png!thumbnail)

`not stripped`：保留符号表信息和调试信息

[![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/mWrzleHIKytaPxz3.png!thumbnail)](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/mWrzleHIKytaPxz3.png!thumbnail)

.. hint:: 要符号表和调试信息（可以知道某个函数在哪个源文件的第几行）可以加入编译选项  ``-g`` （左边为无g，右边加上g）

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/m8mg8wUb5JHbzDTO.png!thumbnail)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/DYx5yZWGBCDjChLX.png!thumbnail)

### Output [Printable Character](http://facweb.cs.depaul.edu/sjost/it212/documents/ascii-pr.htm)

```bash
# 可打印的字符应该是ASCII码
$ strings <file_name>
```

## Symbol

* ELF文件有两种符号表：`.symtab` and `.dynsym`，动态链接库为ELF
* ``.dynsym`` 保留了 ``.symtab`` 中的全局符号(global symbols)。命令strip可以去掉动态库文件中的 ``symtab`` ，但不会去掉 ``.dynsym`` 。
* 使用nm时提示no symbol是因为默认找的 ``symtab`` 表被strip了，因此只能查看动态符号表 ``.dynsym``，加上-D
* `.symtab` 用于动态库自身链接过程，一旦链接完成了，就不再需要了； `.dynsym`表包含动态链接器(dynamic linker)运行期时寻找的symbol.
* `nm` 默认`.symtab` section.

### CLI

```bash
# 可查看object/target的
$ nm [option] <file>
# -A: 输出文件名
# -c: demangle
# -D：查看动态符号表
# -l：显示对应的行号
# -u: 显示未定义的符号

$ objdump [option] <file>

# 查看符号的可见性
$ readelf -s B
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220416175011251.png" alt="image-20220416175011251" style="zoom: 80%;" />

.. note:: 每个.o文件都有一个符号表，符号有两种分类，一个是全局符号，一个是本地符号，本模块的非静态函数和全局变量都是其他模块可见和可用的；静态函数和静态变量都是只有本模块可见的，其他模块不可使用

### Demangle

```bash
$ echo <...> | c++filt
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/9QW4LIXHJmMH6QW5.png!thumbnail" alt="img" style="zoom: 67%;" />

## Q&A

### [extern "C"](https://zhuanlan.zhihu.com/p/114669161)

告知`g++`编译器这部分代码是c库的代码（不需对该部分内容进行symbol mangling），使得C库的符号能够被顺利找到而链接成功

### [Translation Unit](https://stackoverflow.com/questions/1106149/what-is-a-translation-unit-in-c)

According to [standard C++](http://www.efnetcpp.org/wiki/ISO/IEC_14882) （[wayback machine link](http://web.archive.org/web/20070403232333/http://www.efnetcpp.org/wiki/ISO/IEC_14882)）: A translation unit is the basic unit of compilation in C++. It consists of the contents of **a single source file**, plus the contents of any header files directly or indirectly included by it, minus those lines that were ignored using conditional preprocessing statements.

### [precompile source file的#号是什么？](https://stackoverflow.com/questions/25137743/where-do-we-use-i-files-and-how-do-we-generate-them)

一种特殊的注释

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211002140045353.png" alt="image-20211002140045353" style="zoom:67%; " />

### [C++ "multiple definition of .. first defined here"](https://programmerall.com/article/99071342705/)

> Include guards only help against including the same header in one Translation unit (cpp file) multiple times, not against including and compiling the same function in multiple TUs.

头文件宏能够保证一个翻译单元没有重复的符号（symbol）；而不能保证多个翻译单元合起来后（A+B）没有重复的符号

### Why not duplicate symbol？

#### File

* c++

```c++
// A.cpp
int num_A1 = 0;
int num_A2 = 2;

// B.cpp
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

#### Experiment

* 通过实验证明，最终生成的可执行文件并不包含`A.cpp/num_A1`这个变量。即通过对比使用和不使用 target_link_libraries(A B) 的可执行文件A的符号信息（nm命令行）来判别是否一致。

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/4SdVT5emzTdgn0IV.png!thumbnail)

* 另外：`A.cpp`一旦显式触发用上 `B.cpp` 那边的符号之后就会成功触发报错

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/cdvZSiNGWfQYhUWl.png!thumbnail)

.. note:: 另外如果使用的是动态库的话，反而可以顺利通过编译

.. note:: 测试平台windows/ubuntu20.04 g++9.4.0

#### Conclusion

暂无权威信息佐证，以下均为基于实验的猜测：

（1）静态库把所有`symbol`都加到`target`

（2）动态库是把只需要的`symbol`加到`target`

（3）对静态库链接时，如果`target`不需要静态库的任何`symbol`那`链接器ld`就干脆不导入静态库的任何`symbol`；但凡有参考的话，就会触发添加所有的`symbol`

## Reference

* [美团：编译耗时优化原理和实践](https://tech.meituan.com/2020/12/10/apache-kylin-practice-in-meituan.html)
