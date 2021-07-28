# c++语法

## 01.类型别名

引入类型别名这个特性是为了方便程序员写代码，比如说不再需要写完整的较长的类型名，而只需要写简短的类型别名

### 语法规则

```plain
1）语法一：typedef src_type alias_type
2）语法二：using alias_type = src_type（from c++11）
```

（attention）一般来说，推荐使用`using`这种语法，因为当接触到数组类型时，`using`会更直观

```c++
int arr[4];
typedef int IntArr[4];  // [4]需要写在后面
using IntArr = int[4];
```

### size_t

`size_t`类型是一个类型别名；是[sizeof](https://en.cppreference.com/w/c/language/sizeof)函数返回对象的类型(size type)，是一个无符号的整型，它的大小是由操作系统所决定的

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210728200535276.png" alt="image-20210728200535276" style="zoom: 67%;" />

attention：标准库中的operator[]也涉及`size_t`，所以遍历时用unsigned或者int去访问可能会出错

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210728200948093.png" alt="image-20210728200948093" style="zoom: 50%;" />