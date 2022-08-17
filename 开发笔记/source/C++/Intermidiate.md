# Intermidiate

## Endian

- 目前接触的系统一般是小端模式即低字节存储低地址
  ![image-20220813143214031](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220813143214031.png)

- 在Python里面一般会用`>`或`<`来进行大小端的区别

```python
dt = np.dtype('b')  # byte, native byte order
dt = np.dtype('>H') # big-endian unsigned short
dt = np.dtype('<f') # little-endian single-precision float
dt = np.dtype('d')  # double-precision floating-point numbe
```

## 判断两个变量是否是相同类型/某个变量是否某个类型

``` c++
#include <type_traits>
using namespace std; 

int main() {

    vector<int> arr(5);
    // 注意使用尖括号
    cout << is_same_v<decltype(arr.size()), unsigned long> << endl;

}
```

## [全局对象的构造函数先于主函数执行](https://blog.csdn.net/Y673582465/article/details/72878053)

## 如何判断一个类型是左值还是右值

结合decltype的语法规则和[cpp insight](https://cppinsights.io/)可以判断其[value category](https://en.cppreference.com/w/cpp/language/decltype)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210816232635787.png)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210816233508495.png)

## 如何在编译期避免一个函数被调用？

使用function-body`=delete`可提示编译器避免调用某个函数([ref](https://en.cppreference.com/w/cpp/language/function))

```c++
#include <iostream>

using namespace std;
void test01() = delete;

int main() {
  test01();  // error: Call to deleted function 'test01'
  return 0;
}
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210925085531545.png" alt="image-20210925085531545" style="zoom:67%;" />

## 并不是所有编译器都能够支持新特性

可通过cpp reference来查询支持某特性的编译器

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210928215406043.png" alt="image-20210928215406043" style="zoom:50%;" />

## constexpr all things

使用constexpr让编译器将一些处理逻辑放在编译期中处理，一方面提高运行时速度，另一方面可以让bug在编译期就能提早浮现

## c++中别人的程序为什么不用c风格的类型转换操作？

c风格的类型转换需要按顺序尝试一系列c++风格的类型转，而用c++风格的类型转换则对症下药，不用尝试，一步到位

## 获取类型信息

```c++
#include <iostream>
#include <typeinfo>
#include <type_traits>

int main() {
    // c11判断类型
    int a[3] = {1, 2, 3};
    cout << typeid(a).name() << endl;
    // c17判断类型是否相同（type_traits）
    cout << is_same_v<decltype(a), int *> << endl; 
}
```



```c++
constexpr auto fib_impl = []{
    array<int, 31> fib{0, 1};
    for (int i = 2; i != 31; ++i) {
        fib[i] = fib[i - 1] + fib[i - 2];
    }
    return fib;
}();
class Solution {
public:
    int fib(int n) {
        return fib_impl[n];
    }
};
```



## 在写函数时，形参使用指针好还是引用好？

引用会更好，因为可以不用判断参数是否为 `null`

## 为什么在位运算时会有整型提升？

硬件对int型数据有更好的处理效率（是最常使用的数据类型），因此在进行位运算时会将数据转换为整型

``` c++
int main() {

    char a = 0x00;
    char b = 0x01;
    auto c = a & b;
    cout << is_same_v<int, decltype(c)> << endl;  // true

}
```

## 初始化和赋值的区别？

- 初始化（原来没值）：是将一个值存到对象中（i.e. 将value和object进行绑定）；初始化包含在内存中开辟空间，将值存到该空间中和在编译器中构造符号表，将标识符和相关内存空间关联起来（如此，如果要用到一个变量名为a的变量时，就知道在内存的哪个地方找到这个变量）
- 赋值（原来有值）：改变一个对象中的值

``` bash
# hint: 查看符号表
# -C: demangle
$ nm -C <.o文件>
```

## [分配了内存，对象的生存期就开始了吗](https://zh.cppreference.com/w/cpp/language/lifetime#Access_outside_of_lifetime)

可能分配了内存，但生存期还未开始；亦有生存期结束但存储空间尚未释放或重用的时间段

## [变量在内存的存储位置？（内存模型）](https://www.bilibili.com/video/BV1et411b73Z?p=84)

c++程序所占的内存可以划分为4个区域（谐音梗：四驱兄弟）

- 代码区：存放函数体的二进制代码；由操作系统进行管理的；其中的内容是只读（防止修改程序的执行指令）和共享的（只有一份数据，避免拷贝浪费）
- 全局区 ：存放全局变量、静态变量、字符串字面值常量；该部分数据由操作系统释放
- 栈区：存放函数的参数值，局部变量；由编译器自动分配释放，数据的生存周期由编译器管理）
- 堆区：存放由程序员自己管理的数据（数据的生存周期由程序员管理）；若不释放，程序结束时由操作系统释放

## std::enable_if实现中的成员是什么？

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210816175135379.png" alt="image-20210816175135379" style="zoom:50%;" />

- c++中的类的成员除了`data member`，`memeber function`，还可以有`member typedefs`等，对其的使用需要使用`className::type`语法（不能使用`object.type`）

  ```c++
  struct TypeClass {
    typedef int IntType;
  };
  
  int main() {
    TypeClass typeclass;
    // scope resolution的左操作数应为class, namespace, enumeration
    TypeClass::IntType a = 1;
    return 0;
  }
  ```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210816174218380.png" alt="image-20210816174218380" style="zoom: 50%;" />

- [member typedefs](https://en.cppreference.com/w/cpp/language/class)



## 段错误问题

- 段错误（生成的汇编代码没有RET指令）：定义了一个有返回值的函数，但RELEASE实现提供返回值

![image-20220429130701631](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220429130701631.png)

#### Undefined Behavior

- 未定义行为不一定立刻触发

### ![image-20220805213829446](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220805213829446.png)

## [typename可出现的位置？](https://en.cppreference.com/w/cpp/keyword/typename)

- 模板形参声明
- 模板的声明和定义
- Requires表达式（Type requirements）
