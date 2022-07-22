# STL

## Algorithm

|                             函数                             |                             作用                             |               例子                |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :-------------------------------: |
| [accumulate](https://en.cppreference.com/w/cpp/algorithm/accumulate) |          输入一个序列(range)和给定初值，计算累积值           |                 —                 |
|   [copy](https://en.cppreference.com/w/cpp/algorithm/copy)   |                         拷贝一个序列                         |                 —                 |
|   [fill](https://en.cppreference.com/w/cpp/algorithm/fill)   |     输入一个序列(range)和值，将序列的所有元素设置为该值      |                 —                 |
| [fill_in](https://en.cppreference.com/w/cpp/algorithm/fill_n) |                              —                               |                 —                 |
|   [find](https://en.cppreference.com/w/cpp/algorithm/find)   | 输入一个序列和某个值，有该值则返回该值对应的迭代器，否则返回last |                 —                 |
|   [sort](https://en.cppreference.com/w/cpp/algorithm/sort)   | 输入一个序列(range)，对这个range的元素进行排序（默认为升序排列） |  std::sort(s.begin(), s.end());   |
| [transform](https://en.cppreference.com/w/cpp/algorithm/transform) | 输入一个序列(range)，将函数作用于这个range上的每个元素/修改每个元素，得到一个新的序列 |                 —                 |
| [unique](https://en.cppreference.com/w/cpp/algorithm/unique) |               输入一个序列，删除接连重复的元素               |                 —                 |
|                             max                              |                         返回最大元素                         |                 —                 |
|                         max_element                          |            输入一个序列，返回最大值所对应的迭代器            |                 —                 |
|                           reverse                            |                          字符串反转                          | std::reverse(a.begin(), a.end()); |
|                             swap                             |                         交换两个元素                         |                 —                 |
|                             swap                             |                           数据交换                           |            swap(a, b);            |

## [Bind](https://en.cppreference.com/w/cpp/utility/functional/bind)

bind给人感觉是一种`函数适配器`（function adapter），能修改可调用对象对外的接口：比如说一个可调用对象原本有5个形参，经过bind函数后对外的形参就变成3个；又或者经过bind的封装可以[生成多个具有不同默认参数的可调用对象]([修改函数形参的默认实参](https://www.geeksforgeeks.org/bind-function-placeholders-c/))

```c++
#include <functional>
#include <iostream>
using namespace std;

void f(int i, int j) {
    cout << i + j << endl;
}

int main() {
    using namespace std::placeholders;
    /*
     * 返回的是一个函数对象
     * 对函数加不加取地址符都行，存在function->pointer的隐式转换
     */
    auto bind_f1 = bind(&f, _1, 2); // j的默认实参为2
    auto bind_f2 = bind(f, _1, 3);  // j的默认实参为3
    bind_f1(3);  // 5
    bind_f2(3);  // 6
}
```

![image-20210809203143984](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210809203143984.png)

- [bind传参时默认是拷贝传参](https://blog.csdn.net/zzzyyyyyy66/article/details/80285723)

## [Containers](https://en.cppreference.com/w/cpp/container)

- 模板库官方文档索引

|                              —                               |                              —                               |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|  [array](https://en.cppreference.com/w/cpp/container/array)  | [vector](https://en.cppreference.com/w/cpp/container/vector) |
|    [map](https://en.cppreference.com/w/cpp/container/map)    | [unordered_map](https://en.cppreference.com/w/cpp/container/unordered_map) |
|    [set](https://en.cppreference.com/w/cpp/container/set)    | [unordered_set](https://en.cppreference.com/w/cpp/container/unordered_set) |
| [priority_queue](https://en.cppreference.com/w/cpp/container/priority_queue) | [span](https://en.cppreference.com/w/cpp/container/span) （C++20） |
| [sequence](https://en.cppreference.com/w/cpp/container#Sequence_containers) | [associative](https://en.cppreference.com/w/cpp/container#Associative_containers) |
| [unordered associative](https://en.cppreference.com/w/cpp/container#Unordered_associative_containers) | [adaptors](https://en.cppreference.com/w/cpp/container#Container_adaptors) |
|  [deque](https://en.cppreference.com/w/cpp/container/deque)  |                           multiset                           |
|                           multimap                           |                              —                               |

- 容器（container）是存储对象的对象，同时这种对象需要满足一定的[要求](https://en.cppreference.com/w/cpp/named_req/Container)
- [容器底层数据结构一览表](https://interview.huihut.com/#/?id=stl-%e5%ae%b9%e5%99%a8) 红黑树 or 哈希表
- 序列容器的元素在内存中是连续的
- 为什么有这么多不同的序列容器？不同的序列容器的实现是不同的，在不同的任务中会表现出不同的时间复杂度

### Feature

|                     容器                      |                  功能                  |                              —                               |
| :-------------------------------------------: | :------------------------------------: | :----------------------------------------------------------: |
| [array](http://c.biancheng.net/view/411.html) |          静态数组，长度不可变          |                              —                               |
|                    vector                     |            能动态的调整长度            |                              —                               |
|                      set                      | 关联容器；数据存放有序（从小到大存放） |         set迭代器指向的是const对象，不能修改这个元素         |
|                 unordered_set                 |              数据存放无序              |                              —                               |
|                      map                      |       map会按`key`进行自动排序；       |                  map的每一个元素是一个pair                   |
|                 unordered_map                 |                   —                    | 支持[ ]索引；<br />（键值的自动插入）如果索引了不存在的键，c++的字典会自动添加该键（类似python）；<br />unordered_map在插入情况下少的时候用到 |

### Q&A

- vector的模板形参可否是内置数组？

不能，根据[cppreference-vector](https://en.cppreference.com/w/cpp/container/vector)，该类型需要满足 `CopyAssignable` 和 `CopyConstructible` 两种属性。而 `int []` 这种类型不满足 `CopyAssignable` 的属性，因为内置数组不能用于构造vector。

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/qvdJoCaDaGAGjHu9.png!thumbnail" alt="img" style="zoom:67%;" />

- 判断是否满足`CopyAssignable` 属性

```c++
#include <iostream>
#include <type_traits>
using namespace std;

int main() {
    std::cout << std::boolalpha
    << "int[2] is copy-assignable? "
    << std::is_copy_assignable<int[2]>::value << '\n';
}
```

- [如何对vector<vector\<int\>>进行emplace_black](https://stackoverflow.com/questions/20391632/how-to-use-stdvectoremplace-back-for-vectorvectorint)

```c++
# 不能从{}类型推导出std::initializer_list()类型，所以需要显式指明
vec.emplace_back(std::initializer_list<int>{1,2});
```

- [clang-tidy推荐用emplace_back，它和push_back的区别在于?](https://yasenh.github.io/post/cpp-diary-1-emplace_back/)

两者构造元素的方式不一样，前者的效率会更高：前者追加生成vector的元素，是把emplace_back的实参传递给元素的构造函数的形参，然后**直接构造对象**。没有临时对象的构造和析构。后者追加生成vector的元素，是通过**拷贝或移动构造函数**来生成，因此需要先创建一波临时对象。多了临时变量的构造和释放。

```c++
// example
vector.emplace_back(1, 2);
vector.push_back(MyClass(1, 2));
```

- 什么时候使用emplace_back吗？

> Very often the performance difference just won’t matter. As always, the rule of thumb is that you should avoid “optimizations” that make the code less safe or less clear, unless the performance benefit is big enough to show up in your application benchmarks.

- [sizeof(vector)恒为24？](https://www.quora.com/STL-C++-Why-does-sizeof-return-the-same-value-for-all-vectors-regardless-of-the-type-and-number-of-elements-in-that-vector)

其首先存的是三个指针（3×8字节）：`_M_start`, `_M_finish`, `_M_end_of_storage`

## Chrono

```c++
#include <time.h>
#include <chrono>
#include <thread>

// sleep
std::this_thread::sleep_for(std::chrono::milliseconds(3000));

auto startTime = std::chrono::high_resolution_clock::now();
// TODO
auto endTime = std::chrono::high_resolution_clock::now();
float totalTime = std::chrono::duration<float, std::milli> (endTime - startTime).count();
```

## Mutex

- lock

```cpp
#include <mutex>

// you can use std::lock_guard if you want to be exception safe 
// lock_guard类似智能指针
std::mutex m;
int i = 0; 
void lock() 
{
    m.lock();    
    i++; //no other thread can access variable i until m.unlock() is called
    m.unlock();
}
```

**NOTE**

- B线程没有获取到锁的时候，B线程会做什么操作？最简单的是在spin（wait）

- [Mutex tutorial and example](https://nrecursions.blogspot.com/2014/08/mutex-tutorial-and-example.html)
