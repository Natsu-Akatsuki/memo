# C++

Someone who is sophomoric about concepts and reasoning may try things randomly, and wastes a lot of time.

## [Attribute](https://en.cppreference.com/w/cpp/language/attributes)

- 使用场合：如想在编译时告知开发者某个函数即将deprecated，则使用 `deprecated` 属性（当程序中调用这个函数时，编译期时则会弹出相关的信息）

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210811202856933.png" alt="img" style="zoom: 67%;" />

### Container Adapter

修改一个 **底层序列容器**，使其表现出新的功能和接口，比如[stack](https://en.cppreference.com/w/cpp/container/stack) 使`deque`表现出了`先进后出`的栈功能

- 容器适配器有 `stack`， `queue`，[priority_queue](https://en.cppreference.com/w/cpp/container/priority_queue)

```c++
#include <deque> 
#include <iostream> 
#include <list> 
#include <stack> 
#include <vector> 
using namespace std;

int main() {
    // stack的构造 
    stack<int> stackA; 
    stack<int, deque<int>> stackB; // deque is the default container 
    stack<int, vector<int>> stackC; 
    stack<int, list<int>> stackD; 
    return 0;
}
```

## [Object](https://en.cppreference.com/w/cpp/language/object)

- （定义）An object, in C, is **region** of [data storage](https://en.cppreference.com/w/c/language/memory_model) in the execution environment, the contents of which can represent *values* (a value is the meaning of the contents of an object, when interpreted as having a specific [type](https://en.cppreference.com/w/c/language/type)). @[ref](https://en.cppreference.com/w/c/language/object)

- Object的分类（complete / subobject / polymorphic）可参考 @[ref](https://en.cppreference.com/w/cpp/language/object)
- 一个Object具有各种属性：size；alignment requirement；[storage duration](https://en.cppreference.com/w/cpp/language/storage_duration) (automatic, static, dynamic, thread-local)；[type](https://en.cppreference.com/w/cpp/language/type)；value；optionally, a [name](https://en.cppreference.com/w/cpp/language/name).

### Expression

- 操作数（operand）和操作符（operator）的组合
- [expression evaluation](https://en.cppreference.com/w/cpp/language/eval_order)
- [full expression](http://eel.is/c++draft/intro.execution#def:full-expression)
- [ID expression](https://en.cppreference.com/w/cpp/language/identifiers)：该表达式只包含标识符，其结果为其具名的实体（`entity`）

#### [Value category](https://en.cppreference.com/w/cpp/language/value_category)

- 泛左值、纯右值和将亡值实际上就是表达式
- 根据表达式的取值（evaluation）结果进行分类，就能得到泛左值、纯右值和将亡值若干类别

|  类别  |                             概念                             | —                                                            |
| :----: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| 泛左值 |                 其取值标识了一个对象或者函数                 | 可取址；可修改的左值可以放在内置赋值操作符左边；<br />可用于初始化左值引用 |
| 纯右值 | **其取值能对一个对象进行初始化**<br />能够得到一个内置操作符的值 | 不可取址；                                                   |
| 将亡值 |                     资源能够复用的泛左值                     | —                                                            |

![C++ expression value categories.](https://docs.microsoft.com/en-us/cpp/cpp/media/value_categories.png?view=msvc-170)

- （作用：明确了编译器evaluation的规则）The value categories are the basis for rules that compilers must follow when creating, copying, and moving temporary objects during expression evaluation. @[ref](https://docs.microsoft.com/en-us/cpp/cpp/lvalues-and-rvalues-visual-cpp?view=msvc-170)
- 在过去，左右值的区别比较容易，处于赋值语句左边的操作数为左值，处于赋值语句右边的操作数为右值。但放在现在是不恰当的，比如说

```c++
const int a = 1; // a是左值，但是不能放赋值语句左边，因为不可修改
```

## Data Structure

> a data structure is a collection of data values, the relationships among  them, and the functions or operations that can be applied to the data

数据结构是数据值、数据关系、作用于数据的函数和操作的集合

## [Declaration](https://en.cppreference.com/w/cpp/language/declarations) and [Definition](https://en.cppreference.com/w/cpp/language/definition)

- 定义：定义是一种特殊的声明，能够让一个实体足以被编译器使用
- 非定义性声明：告知编译器存在一个实体，等下可以使用它

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/venn_declaration.png" alt="img" style="zoom: 50%;" />

### [ODR](https://en.cppreference.com/w/cpp/language/definition#ODR-use)

- 一处定义原则：定义只允许在`翻译单元`/`程序单元`出现一次
- 可以不用满足`程序单元级别`，只需要满足`翻译单元级别`的一处定义原则的包括：`class type`，`enumeration type`， `inline function`，`inline variable` 

- 

## [Entity](https://en.cppreference.com/w/cpp/language/basic_concepts)

- c++程序中的实体包括值（`value`)，对象（`object`），引用（`reference`），函数（`function`），类型（`type`），模板（`template`）

- 预处理宏（`prepocessor macro`）不是c++实体（有人从它不是c++语法的内容去理解）

## [Identifier](https://en.cppreference.com/w/cpp/language/identifiers)

- 标识符是一段任意长序列，该序列由大小写拉丁字母，数字，下划线，和大部分Unicode字符组成

- 标识符能够给如下实体具名：`objects`, `references`, `functions`, `enumerators`, `types`, `class members`, `namespaces`, `templates`, `template specialization`等

### [Literals](https://www.tutorialspoint.com/cplusplus/cpp_constants_literals.htm)

字面值示例：

```c++
// integer literal
211...
// string literal
"hello world"...
// boolean literal    
true
false
// character literal
'a'...
// the pointer literal
nullptr
// the user-defined literal    
...
// floating point literal    
```

.. attention:: 注意 ``literals`` 和 ``literal type`` 是不一样的

## [Type](https://en.cppreference.com/w/cpp/language/type)

类型（type）是函数（function）、表达式（expression）、对象（object）的属性；类型决定了**二进制值的转译方式**（这些二进制值可能存储在对象中，也可能是表达式求值（evaluation）后的结果）

```c++
#include <iostream>

int main() {
  // 二进制1100001的int型表示为  ->  97
  // 二进制1100001的char型表示为 ->  'a'
  using namespace std; 
  int int_num = 97; 
  char char_num = int_num; 
  cout << "int_num: " << int_num << endl; 
  cout << "char_num: " << char_num << endl; 
}
```

### alignment requirement

- [结构体的对齐操作](https://zhuanlan.zhihu.com/p/412092275)

### alias

用简短的类型别名替换完整的、较长的类型名

``` c++
// 语法一
typedef src_type alias_type
// 语法二（from c++11）
using alias_type = src_type
```

.. note:: 一般来说，推荐使用 `using` 这种语法，因为在标识数组类型时， `using` 会更直观（如下例所示）；另外[typedef不支持模板类别名](https://www.cnblogs.com/yutongqing/p/6794652.html)

``` c++
int arr[4]; 
typedef int IntArr[4]; // [4]需要写在后面
using IntArr = int[4]; 
```

### size_t

* `size_t`类型是一个特殊的类型别名，是`sizeof()`函数的返回值类型
* 其是一个无符号的整型，大小由操作系统所决定的；在进行动态内存分配时很有用。

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210728200535276.png" alt="img" style="zoom:67%;" />

.. attention:: 标准库中的`operator[]`涉及`size_t`，所以遍历时用`unsigned`或者`int`类型的数据去访问可能会出错
<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210728200948093.png" alt="img" style="zoom: 50%;" />

### [sizeof](https://en.cppreference.com/w/c/language/sizeof)

- `sizeof(type/object)`：用来获取`对象`或`类型所对应的对象`的大小（单位：字节）
- `sizeof`无法测动态数组的内存大小，因为得到的只是指向首元素的指针而不是数组名（数组名也是个地址，但其类型指向整个数组）"Used when actual size of the object must be known"

```cpp
auto ptr = new bool[40];
cout << sizeof(ptr) << endl; // 8
cout << sizeof(*ptr) << endl; // 1（获得指向首元素的对象，并得其大小）
bool ptr2[40];
cout << sizeof(ptr2) << endl; // 40
```

### incomplete type

> .... has initializer but incomplete type

[incomplete type](https://en.cppreference.com/w/c/language/type)：这种类型的对象，编译器是不知道它的大小的，如int[]

## [Reference](https://en.cppreference.com/w/cpp/language/reference)

- 引用不是对象，不占用`data storgae`

- const左值引用和右值引用能够延展临时变量（或称不具名变量/匿名变量）的生存期；但是const左值引用

``` c++
#include <iostream>
using namespace std;

int main() {
   double tc = 21.5;
   /*
   * 以下近似于：创建一个临时变量，然后对其进行绑定
   * double temp = 7.07;
   * const double &rd1 = temp;
   * 在语法上只有const左值引用才能绑定临时变量，否则有如下报错：
   * “non-const lvalue reference to double can not bind a temporary of type double”
   */
   const double &rd1 = 7.07;

   // 通过const左值引用或右值引用对临时变量起别名，临时变量的lifetime会被extend
   double &&rd2 = 7.07;
   rd2 = 33;
}
```

### [Forward Reference](https://en.cppreference.com/w/cpp/language/reference)

### [Reference Collapse](https://en.cppreference.com/w/cpp/language/reference)

- 引用折叠只会发生在类型别名和模板

```c++
typedef int &lref;
typedef int &&rref;
int n;

lref &r1 = n; // type of r1 is int&   左值引用 + 左值引用 -> 左值引用
lref &&r2 = n; // type of r2 is int&  左值引用 + 右值引用 -> 右值引用
rref &r3 = n; // type of r3 is int&   右值引用 + 左值引用 -> 右值引用
rref &&r4 = 1; // type of r4 is int&& 右值引用 + 右值引用 -> 右值引用
```

### [Dangling Reference](https://en.cppreference.com/w/cpp/language/reference)

当引用的对象的`lifetime`已经结束了，那此时的引用则为 `dangling reference`，相关的报错信息为段错误

``` c++
std::string &f() {
    std::string s = "Example";
    return s; // exits the scope of s:
    // its destructor is called and its storage deallocated
}

int main() {
    std::string &r = f(); // dangling reference
    std::cout << r;       // undefined behavior: reads from a dangling reference
    std::string s = f();  // undefined behavior: copy-initializes from a dangling reference
}
```

### Q&A

> a是T的左值引用，T是int&&类型的，那a实际上是什么（类型）？

- A1：int&（根据引用折叠的说法，右值引用的左值引用是左值引用）


### 类型转换

- 操作符对操作数会有要求，或操作数满足某种特定的类型，或操作数都需要同样的类型；如果不满足相应的需要，就会出现类型转换
- [隐式类型转换](https://en.cppreference.com/w/cpp/language/implicit_conversion)

#### [C风格类型转换](https://en.cppreference.com/w/cpp/language/explicit_cast)

c风格的显式类型转换包含了一系列的转换操作（也就是它会尝试一组转换操作，例如首先进行const_cast，然后进行static_cast, reinterpret_cast等...）；c++中一般都使用细颗粒度的，更具体的c++风格的类型转换操作（即static_cast, const_cast, reinterpret_cast...等）

![image-20210930163600592](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210930163600592.png)

.. note:: ``reinterpret_cast``一般针对的是指针

.. note:: 建议通过程序的改良，来减小对类型转换的使用。（李伟老师：设计static_cast这些要打这么长而麻烦的函数，就是为了降低开发者使用类型转换的频率）

### array

- c++没有引用的数组。因为数组的元素应该是对象而引用不是对象。

### 类型退化

- 数组到指针（array to pointer）的隐式转换

```c++
int a[3]; 
// 该指针指向数组的首元素
auto b = a; // b->int* 而不是 int*[3]
```

![image-20210815211847957](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210815211847957.png)

- [why-the-address-of-dynamic-array-is-different-from-the-first-element-address](https://stackoverflow.com/questions/63114268/why-the-address-of-dynamic-array-is-different-from-the-first-element-address)

### Pointer

#### operator

```c++
int arr[4] = {1, 2, 3, 4};
int *ptr = arr;
// 等价于输出ptr[0]
cout << *ptr << endl;
// 等价于prt[3]
cout << *(ptr + 3) << endl;
```

### 往字节流中存储任意类型的数据

步骤一：reinterpreter_cast字节流到待存储数据的类型，然后存值

步骤二：将指针指向下一个位置

```c++
template <typename T> void write(char *&buffer, const T &val) {
  // T buffer[] = val;
  *reinterpret_cast<T *>(buffer) = val;
  buffer += sizeof(T);
}

void serialize(void *buffer) {
  char *d = static_cast<char *>(buffer);  
  int mClassCount = 3;  
  write(d, mClassCount);  
}
```

## Class

- 在对类进行实例化时，先会对数据成员进行初始化，初始化的优先级为：`初始化列表`，`类内初始化`。换句话说`初始化列表`会替换`类内初始化`的行为。（@[ref](https://en.cppreference.com/w/cpp/language/data_members)）

```c++
#include <iostream>
using namespace std;
class Test {
public:
  Test() : var_(3) {
    cout << var_ << endl;  // 输出3
  }
  int var_ = 1;
};

int main() {
  Test test;
}
```

- 成员函数（`member function`）和数据成员（`data member`）是分开存储的；只有实例属性是存放在对象上的
- 空类的大小为1，是为了保证每个（空）对象都有独一无二的内存地址，以使编译器区分他们

### Access Right

#### Inherit

| —        | public    | protected | private |
| -------- | --------- | --------- | ------- |
| 公有继承 | public    | protected | 不可见  |
| 私有继承 | private   | private   | 不可见  |
| 保护继承 | protected | protected | 不可见  |

- 公有继承：继承的`public`, `protected`成员保持原来的访问权限；子类无法访问基类的`prviate`成员
- 保护继承：继承的`public`, `protected`成员变为`protect`成员；子类无法访问基类的`prviate`成员
- 私有继承：继承的`public`, `protected`成员变为继承类的`private`成员；子类无法访问基类的`prviate`成员

### Construct Function

- 创建一个类时编译器至少给一个类添加如下特殊的函数， `默认构造函数`（函数体为空，无参）； `默认拷贝函数` ，对属性进行值拷贝（浅拷贝）；赋值运算符（assignment operator），对属性进行值拷贝

- 构造函数分类：

| 类型                                                         | 描述                                                         | 案例                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| （类型）转移构造函数（`convert constructor`）                | A constructor that is **not declared with the specifier explicit** and which **can be called with a single parameter** (until C++11) is called a converting constructor. |                                                              |
| `LiteralType`                                                | 有`constexpr specifier`                                      |                                                              |
| 拷贝构造函数（`copy constructor`） / 移动构造函数（`move constructor`） | 调用构造函数时有实参（实参为同类型的对象）                   | [CSDN](https://blog.csdn.net/weixin_42492218/article/details/124386107) |
| 目标构造函数 / 委托构造函数                                  |                                                              |                                                              |

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/TUuFl421J2PPxDeO.png!thumbnail)

#### [Converting](https://en.cppreference.com/w/cpp/language/converting_constructor)

- 声明时`explicit`和**能够用一个实参调用的构造函数**（C++11之后只看有没有`explicit`了）即为类型转换构造函数（`converting constructor`）
- 转移构造函数描述了一个隐式类型转换的函数（该函数能将实参类型的对象转换（`implicit conversion`）到类类型对象）
- 可以通过`explicit`禁止类型的隐式转换

```cpp
class Test {
public:
  explicit Test(int i) {
  }
};

int main() {
  Test test = 10; // error：禁止类型的隐式转换
}
```

#### Copy

- 一个`T类型`的对象被另一个`T类型`的对象初始化（包括[直接初始化](https://en.cppreference.com/w/cpp/language/direct_initialization)和[拷贝初始化](https://en.cppreference.com/w/cpp/language/copy_initialization)）时，则会触发拷贝构造

```c++
// b为T类型
T a = b; // 拷贝初始化
T a(b); // 直接初始化
```

#### Default

- `默认构造函数`：不需要提供实参就能调用的构造函数

- 当类中没有提供任何构造函数时，且满足一定条件下（如数据成员没有引用），编译器将会隐式提供一个默认构造函数

- 使用默认构造时要避免`most vexing parse`，即编译器将对象的初始化解析为函数声明。

```c++
using namespace std;
class Test {
public:
  Test() = default;
};

int main() {
  Test test; // ok
  Test test0{}; // ok
  Test test1(); // most vexing parse
}
```

#### [Delegating](https://en.cppreference.com/w/cpp/language/constructor)

- 委托构造放在成员初始化列表中（当前仅有它一个成员）
- 用于复用构造函数的逻辑

```c++
#include <iostream>
using namespace std;
class Test {
public:
  Test(int i) : Test() {
    std::cout << "simple constructor" << std::endl;
  }
  Test() {
    std::cout << "delegating constructor" << std::endl;
  }
};

int main() {
  Test test(1);
}

// output:
// delegating constructor
// simple constructor
```

#### Move

- 接收右值引用的构造函数

```c++
#include <vector>
#include <iostream>
#include <cassert>
using namespace std;
int main() {
  std::vector<int> v{1, 2, 3, 4, 5};
  std::cout << "pre-move: " << (void *) (v.data()) << std::endl;
  std::vector<int> v2(std::move(v));
  assert(v.empty());
  std::cout << "post-move: " << (void *) (v.data()) << std::endl;
  std::cout << (void *) (v2.data()) << std::endl;

  return 0;
}

// pre-move: 0x5628b59daeb0
// post-move: 0
// 0x5628b59daeb0
```

### Const Function

常成员函数，说明在该成员函数内不能修改成员变量（在成员属性声明时加关键词mutable，在常函数中则可以修改）

### Assignment Function

- 一般来说返回值为自身的引用，用于保持连等的操作

```c++
struct Str {
  Str &operator=(const Str &str) {
    this->val = str.val; // 拷贝赋值
    return *this;
  }

  Str &operator=(Str &&str) {
    this->val = str.val; // 移动赋值
    return *this;
  }

  int val = 0;
};
```

### Compiler Action

- 如果自定义了构造函数，编译器不会提供默认构造函数（无参默认构造函数），但会提供默认拷贝构造函数；如果定义了**拷贝构造函数**，则编译器不会提供其他构造函数

- 总的来说，至少会有一个拷贝构造函数

### Class Inherit

- 基本语法

```c++
class class_name : public base_class1, private base_class2
{
	// todo
};
```

#### 父类指针调用子类方法

```c++
// 非调用虚函数时需要使用dynamic_cast将父类指针转换为子类指针
dynamic_cast<cl::NetTensorRT *>(net_.get())
      ->paintPointCloud(*pointcloud_ros, color_pointcloud, labels.get());
```

#### 类函数声明和构造函数

```c++
#include <iostream>
using namespace std;
class Test {

public:
  Test() : num_(3) { // 自定义无参构造函数
    cout << num_ << endl;
  }
  int num_ = 1;
};

int main() {
  Test t0(); // 类函数声明，不调用构造函数
  Test t1; // 输出3
  return 0;
}
```

### [Member Initializers List](https://en.cppreference.com/w/cpp/language/constructor)

```c++
// constructor definition
ClassName([param-list]) <function-body[member-initializers-list]/brace-init-list>
```

.. note:: expression-list属于函数定义的函数体部分

```c++
struct S {
    int n;
    S(int); // constructor declaration
    S() : n(7) {} // constructor definition.
                  // ": n(7)" is the initializer list
                  // ": n(7) {}" is the function body
};
S::S(int x) : n{x} {} // constructor definition. ": n{x}" is the initializer list
int main()
{
    S s; // calls S::S()
    S s2(10); // calls S::S(int)
}
```

### Type

#### [LiteralType](https://en.cppreference.com/w/cpp/named_req/LiteralType)

- 构造函数具备`constexpr specifier`则其实例化的对象类型为`LiteralType`
- 字面值类型包含了：`cv-qualified class type`，其具备以下属性：
  - 具备平凡析构函数（until C++20）；具备`constexpr`修饰的析构函数（since C++）
  - 至少含有一个`constexpr`修饰的构造函数（可能带模板），另外该构造函数不能是拷贝或移动构造函数

### Virtual Function

#### Abstract Class

- 声明了一个**纯虚函数**或继承了一个**纯虚函数**但是未提供实现的类（有纯虚函数但没有实现的类）；**抽象类无法实例化**。（只是含虚函数的类是可以实例化的）
- `Abstract Class`可作为`Concrete Class`的基类





- 接口类（`interface class`）：成员中**仅有虚函数**的类
- [纯虚函数](https://en.cppreference.com/w/cpp/language/abstract_class)：**声明**虚函数的基础上加上一个等于0

```c++
declarator virt-specifier(optional) = 0
```

- 纯虚函数的修饰符（`sequence=0`）不能出现在定义中

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/Xn98yfvbrscGU5X1.png!thumbnail" alt="img" style="zoom:50%;" />

- 虚函数是一种特殊的成员函数，其实现能被子类重写（`override`）
- 带虚函数的类不能实例化

```c++
#include <iostream>
using namespace std;

class Base {
public:
  virtual void test();
};

int main() {
  /*  
   * ERROR: undefined reference to `vtable for Base'
  */
  Base base;
}
```

- [非纯虚函数需要提供实现](https://stackoverflow.com/questions/8642124/should-a-virtual-function-essentially-have-a-definition)

```c++
#include <iostream>
using namespace std;

class Base {
public:
  virtual void test();
};

class Children : public Base {

public:
  void test() { cout << "hello" << endl; }
};

int main() {
  /*
   * ERROR：undefined reference to `typeinfo for Base'
   */
  Children children;
  children.test();
}
```

- 父类允许对**纯虚函数**提供**类外实现**，允许在子类中调用父类的纯虚函数（父类不可以调用，因为该父类不可以被实例化）

```c++
#include <iostream>
using namespace std;

class Base {
public:
  virtual void test() = 0;
};

void Base::test() {
  cout << "纯虚函数可以提供类外实现" << endl;
}

class Children : public Base {

public:
  void test() { cout << "hello" << endl; }
};

int main() {
  Children children;
  children.test();
  // 父类的纯虚函数可以被子类调用
  children.Base::test();
}
```

#### Compiler

对于一个有 `虚函数` **实现**的类，编译器会在其对象的首位置添加 `虚函数指针`  ，虚函数指针指向虚函数，表虚函数表中存放了各个虚函数的地址

#### 静态绑定和动态绑定函数

```c++
#include <iostream>

using namespace std;

class Animal {
public:
  virtual void speak() { cout << "动物在叫" << endl; } // 表明这个函数需要运行期确定
};

class Cat : public Animal {
public:
  void speak() { cout << "猫在叫" << endl; }
};

// 用父类的指针接收子类地址
void doSpeak(Animal &animal) { animal.speak(); }  

void test01() {
  Cat cat;
  doSpeak(cat);
}

int main() {
  test01();
  return 0;
}
```

- 只有非静态成员函数和非构造函数才能声明为虚函数
- 对于一个提供了 `虚函数` **实现**的类，编译器会在其对象的首位置添加 `虚函数指针`  ；虚函数指针指向虚函数表；虚函数表(virtual table function) 中存放了各个**重写**（overload）后的虚函数的地址

```c++
#include <iostream>
using namespace std;

class Base {
public:
  virtual void test() = 0;
};

void Base::test() { cout << "base" << endl; };

class ChildrenA : public Base {
public:
  void test() { cout << "childrenA" << endl; }
};

class ChildrenB : public Base {
public:
  void test() { cout << "childrenB" << endl; }
};

int main() {
  ChildrenA childrenA;
  ChildrenB childrenB;
  Base *base_ptr = nullptr;
  base_ptr = &childrenA;
  base_ptr->test();
}

(gdb) gdb
(gdb) p /R childrenA
(ChildrenA) {<Base> = {_vptr.Base = 0x55c1d978ed40 <vtable for ChildrenA+16>}, <No data fields>}

(gdb) info vtbl childrenA
vtable for 'ChildrenA' @ 0x55c1d978ed40 (subobject @ 0x7fffe0d4cb30):
[0]: 0x55c1d978c2e2 <ChildrenA::test()>

(gdb) info vtbl childrenB
vtable for 'ChildrenB' @ 0x55c1d978ed28 (subobject @ 0x7fffe0d4cb38)
[0]: 0x55c1d978c31e <ChildrenB::test()>
```

### [This Pointer](https://en.cppreference.com/w/cpp/language/this)

- `this指针`是一个隐含于每一个非静态成员函数中的特殊指针，它指向调用该成员函数的那个对象；对一个对象调用其成员函数时，编译器先将对象的地址赋给this指针，然后调用成员函数，每次成员函数存取数据成员时，都隐式地使用this指针
- this指针不是一个常规的变量，而是一个右值，不能取地址即 `&this`
- 一般用来解决名称冲突：区分同名的形参；二来通过返回对象本身（ `return *this` ），实现链式编程

## Memory

### Smart Pointer

``` c++
/*
* 用法：
* 1.导入<memory>头文件
* 2.构造（两种方法：单纯用share_ptr；调用make_shared）   
*/
#include <memory>
int main() {
   // 管理动态内存，创建的智能指针的形参为相应的地址
   std::shared_ptr<int> foo = std::make_shared<int> (10);
   // same as:
   std::shared_ptr<int> foo2 (new int(10));

   auto bar = std::make_shared<int> (20);
   auto baz = std::make_shared<std::pair<int,int>> (30,40);

   std::cout << "*foo: " << *foo << '\n';
   std::cout << "*bar: " << *bar << '\n';
   std::cout << "*baz: " << baz->first << ' ' << baz->second << '\n';
   return 0;
}
```

- unique智能指针对象支持下标索引底层数据

```c++
#include <memory>

int main() {
    auto arrA = std::make_unique<int[]> (10);
    auto arrB = std::make_shared<int[]> (10);
    arrA[0] = 1;
    arrB[0] = 1;
    return 0;
}
```

.. note:: 注意类型带[]

引入智能指针是为了更好地**管理指针**和**管理动态内存空间**。以前管理动态内存是通过`new` 来分配内存空间，通过 `delete` 来释放内存空间。但容易发生一种情况，用 new 在分配了内存空间，但是忘了使用 delete释放内存空间，或者由于异常的抛出，程序无法调delete，这就会造成内存的泄露（该释放的内存空间没有被释放）。于是就有人提出能不能有一种指针，在它销毁的时候，它所指向的对象也会被销毁，于是就引入了智能指针类，它包含了一个满足这种一并销毁需求的析构函数。

.. hint:: 存在一种情况，一个对象由多个指针管理，那就可能会导致多次的释放，于是就引入了包含引用计数技术的共享指针 `shared_ptr`（每有一个共享指针，引用计数+1），只有引用计数为0时，指向的对象才会释放

.. hint:: 有的时候希望只能有一个智能指针管理对象，那就可以使用 `unique_ptr`

- 语法：

```c++
// 创建一个智能指针，该指针指向含10个整型数据的空间
auto pointer = std::make_shared<int>(10);
// ...，初值为0
auto pointer = std::make_shared<int>(10, 0);
```

- 拓展资料：[csdn](https://blog.csdn.net/icandoit_2014/article/details/56666277)

### 动态内存分配：new/delete

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210805103706659.png)

``` c++
auto ptr = new int (6); 
auto ptr = new (int) {6};

// 构造数组
int *arr = new int[10];
// 构造数组，数据默认初始化为0
int *arr = new int[10]();
```

### [内存泄露](https://en.cppreference.com/w/cpp/language/new)

- new返回的指针被释放，导致原来被指向的对象不能通过该指针来访问和不能使用delete来释放

## Namespace

![image-20220728084329476](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220728084329476.png)

## Function

### Default Argument

- from c++98

- 默认实参不能在函数声明和实现中同时出现

- 某个位置参数有默认值后，则从这个位置往后从左到右到必须有默认值

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220227093101436.png" alt="image-20220227093101436" style="zoom:67%;" />

### Inline

- 内联更倾向于是一种编译器的优化策略，编译器会将调用处的代码，替换为函数体的内容



- 内联函数是为了减少函数调用的开销（编译器处理内联函数时相当于将函数的调用，替换为内联函数的函数体）
- 相当于宏，但比宏多了类型检查，真正具有函数特性
- 编译器一般不对包含循环、递归、switch等复杂操作的函数进行内联
- **在类声明中定义的函数**，除了虚函数的其他函数都会自动隐式地当成内联函数

``` c++
// 声明1（可加可不加inline，推荐加inline）
inline int functionName(int first, int second,...);
// 声明2
int functionName(int first, int second,...);

// 定义
inline int functionName(int first, int second,...) {
   ; // todo      
}

// 类内定义，隐式内联
class A {
   int doA() { return 0; }
}

// 类外定义，需要显式内联
class A {
   int doA();
}
inline int A::doA() { return 0; }   // 需要显式内联
```

.. attention:: 实例的内联函数语法需要类内声明，类外定义

### [auto形参(c++20)](https://en.cppreference.com/w/cpp/language/auto)

![image-20210929184924216](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210929184924216.png)

```c++
template <typename T>
void print_container(T set) {
  for (auto &element : set) {
    cout << element << " ";
  }
  cout << endl;
}

// 等价
void print_container(auto set) {
  for (auto &element : set) {
    cout << element << " ";
  }
  cout << endl;
}
```

## [Template](https://en.cppreference.com/w/cpp/language/templates)

- template有三种参数：类型模板参数（`type template parameters`），非类型模板参数（`non-type template parameters`），模板的模板参数（`template template parameters`）

### [Constraint](https://en.cppreference.com/w/cpp/language/constraints)

- 20标准引入`constaint`对模板进行约束
- Named sets of such [requirements](https://en.cppreference.com/w/cpp/language/requires) are called concepts. 
- Each concept is a predicate, evaluated at compile time, and becomes a part of the interface of a template where it is used as a constraint
- 对模板实参的约束称为`requirement`，`requirement`的名字称为concept，

#### [Concept](https://en.cppreference.com/w/cpp/language/constraints)

- 想要对模板参数进行相应的限制

```c++
// concept的定义语法
template <template-parameter-list>
concept concept-name = constraint-expression;

// concept
template<class T, class U>
concept Derived = std::is_base_of<U, T>::value;
```

#### [Requires](https://en.cppreference.com/w/cpp/language/requires)

- requires表达式返回一个`bool`值
- `Simple requirements`：不进行取值，编译器检验实参的某些操作是否合法

```c++
template<typename T>
concept Addable = requires(T a, T b)
{
  // requires {requirement-seq}
  a + b; // "the expression a+b is a valid expression that will compile"
};
```

- `Type requirements`：检验类型实参是否为指定类型
- `Compound requirements`：检验实参是否满足某种操作，这种操作的返回值类型是否合法

- `Nested requiements`：含各种复合语句

### Declaration

### Definition

### Instantiated

- 触发显式地实例化（一个程序只能有一次触发），生成相应的代码（@[ref](https://en.cppreference.com/w/cpp/language/class_template)）
- 类模板显式实例化

```c++
template class-key template-name <argument-list>; (1)
```

- 函数模板显式实例化

```c++
template return-type name <argument-list> (parameter-list);	(1)	// 不需要模板实参类型推导
template return-type name (parameter-list); (2) // 需要模板实参类型推导（根据函数形参）

template<typename T>
void f(T s)
{
    std::cout << s << '\n';
}
 
template void f<double>(double); // instantiates f<double>(double)
template void f<>(char);         // instantiates f<char>(char), template argument deduced
template void f(int);            // instantiates f<int>(int), template argument deduced
```

### [Specialization](https://en.cppreference.com/w/cpp/language/template_specialization)

编译器根据模板生成函数/类定义（`definition`）的行为称为模板实例化（`template instantiation`）。其中生成的definition称为特化（`specialization`）；一个被特化的模板称为`primary template`

#### Explicit template specialization

自定义模板代码

- 语法

```cpp
template <> declaration
```

- 示例

```c++
#include <iostream>
 
template<typename T> // primary template
struct is_void : std::false_type {};
template<>           // explicit specialization for T = void
struct is_void<void> : std::true_type {};
 
int main()
{
    // 模板实参为void之外的类型，则继承false_type
    std::cout << is_void<char>::value << '\n'; 
    // 模板实参为void时，则继承true_type
    std::cout << is_void<void>::value << '\n';
}
```

### Perfect forward

基于引用折叠和`std::forward`函数便能实现完美转发（即一个函数能够保证实参的value category保持一致地转发）

### Compiler

- 链接阶段，不同翻译单元相同的实例模板（`instantiations`）将被合并





### Q&A

- 写一个既能接收左值又能接收右值的函数模板

```c++
template<class T>
void fun(T &&param) {
  return g(std::forward<T>(param));
}
```

## [Operator](https://en.cppreference.com/w/cpp/language/operator_precedence)

### 优先级

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210924234049109.png" alt="image-20210924234049109" style="zoom: 80%; " />

### 加法操作符

- 一元加法操作符能触发整型提升(integral promotion)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/aVEipBB1RGJa00DO.png!thumbnail)

### 逻辑和关系操作符

其操作数和结果均为右值（结果的类型为bool）

### 自增自减表达式

- 后缀表达式（i++）的优先级高于前缀表达式(++i)
- 后缀表达式的取值(evaluation)为 `右值` （返回是操作数的拷贝/临时变量）；前缀表达式的取值为`左值`（在原来的值的基础上+1）

- 一般推荐使用**前缀表达式**，避免拷贝的开销；但现在一些编译器已经能够对后缀表达式进行优化，使其不用构建临时变量

> Post-increment and post-decrement creates a copy of the object, increments or decrements the value of the object and returns the copy from before the increment or decrement.

``` c++
int main() {
   int arr[] = {1, 2, 3}; 
   auto ptr = arr; 
   cout << *ptr << endl; // 1
   cout << *ptr++ << endl; // 1 后缀表达式的evaluation为原值的copy；此处等价于*(ptr++)->*ptr
   cout << *++ptr << endl; // 3 后缀表达式处理完后，地址已+1，此处地址再+1；此处等价于*(++ptr)
}

```

### 逗号操作符表达式

虽然左操作数也要进行取值(evaluate)，但逗号操作符表达式的取值(evaluation)只跟第二个操作数的取值有关（它们的type, value, value category将保持一致）

![image-20210925000412285](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210925000412285.png)

### 域解析运算符

- 全局作用域符(::name)：用于类、类成员、成员函数、变量前，表示作用域为**全局命名空间**
- 类作用域符(class::name)：用于表示指定类型的作用域范围是具体某个类的
- 命名空间作用域符(namespace::name)：用于表示指定类型的作用域范围是具体某个命名空间的

### [Operator Overload](https://en.cppreference.com/w/cpp/language/operators)

- 操作符实际上是一种特殊的函数，比如以下就等价于调用`operator.+`这个函数

```c++
std::string str = "Hello, ";
str.operator+=("world"); // same as str += "world";
```

- 重载的操作符的参数个数取决于操作符本身，形参的类型至少有一个是类类型
- 除`operator()`外其他运算符都不能使用默认参数
- 操作符支持两种方式的重载，一种是作为成员函数，一种是作为非成员函数。（`=`，`[]`，`->`，`()`都是需要作为成员函数进行重载）

- 对于输入输出流的操作符，**不能作为成员函数进行重载**，因为操作符的左操作数应该为输入/输出流对象

- 逻辑运算符的重载会丢失短路逻辑

- 实例：加法运算符作为成员函数的重载

```c++
class Str {
public:
  int val = 2;
  Str &operator+(const Str &str) {
    this->val += str.val;
    return *this;
  }
};

using namespace std;
int main() {
  Str str1;
  Str str2;
  str1 = str1 + str2;
  cout << str1.val << endl;
}
```

## [Initialization](https://en.cppreference.com/w/cpp/language/initialization)

- 初始化即在变量构造时提供初值；函数调用和函数返回时也存在初始化；

- 初始化器(initializer)有三种： {exp list} 花括号，圆括号 (initializer list) ，等号  = exp

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210927194119218.png" alt="image-20210927194119218" style="zoom:67%;" />

---

**NOTE**

- expression-list: comma-separated list

- 数组不支持expression-list的形式(array initializer must be an initializer-list)

---



### [Copy Initialization](https://en.cppreference.com/w/cpp/language/copy_initialization)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220814211034567.png" alt="image-20220814211034567" style="zoom: 80%;" />

## Macro

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210803143453257.png" alt="img" style="zoom:67%;" />

### [pragma once](https://en.wikipedia.org/wiki/Pragma_once)

`#pragma`和`#ifdef`/`#ifndef`/`#endif`一样都是preposess directive（预处理指令），前者是编译器特性（部分版本较老的编译器不支持），后者是c++标准（所有编译器都支持该语法）；都能保证一个头文件不会被重复包含(include)。前者的作用单位是一个文件，后者的作用单位是代码块。前者对于某些编译器能够提高编译速度；后者需要避免有重复的宏名。

## Syntactic sugar

### range-based loop

又称为range-for，是for循环的语法糖，用于遍历序列容器、字符串和内置数组

### [Lambda expressions](https://en.cppreference.com/w/cpp/language/lambda)

一般构建可调用对象(callable object)可以通过对类的`()`操作符(operator)进行重载来构建，但自己写起来比较长，所以有了lambda表达式这种简化和灵活的写法。匿名表达式可以认为是一种语法特性，该表达式会被**编译器翻译为类进行处理**；能够用来生成一个**可调用对象**（该对象的类型是一个**类**）/[又或者说构建一个不具名的函数对象，同时该对象能够使用（捕获capture）该函数对象所在域的变量（这样的对象又称为：closure）](https://en.cppreference.com/w/cpp/language/lambda)

![image-20210821210326787](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210821210326787.png)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210821223323813.png" alt="image-20210821223323813"  />

- 有关捕获，个人理解是描述了可以在`function body`使用的外部变量，具体来说即构建的函数对象所在域的变量

![image-20210821223919209](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210821223919209.png)

### [structured binding](https://en.cppreference.com/w/cpp/language/structured_binding)

为**c++17**的特性，所用类似python的解包；structured binding是一个已存在对象的别名(alias)，类似引用，但又有区别

### static

根据不同的对象，表现出不同的作用

1. 修饰普通变量，修改变量的存储区域和生命周期（lifetime），使变量存储在静态区，在main函数运行前就分配了空间
2. 修饰普通函数（描述该函数具有内部链接性）。在多人开发项目中，为了**防止与他人命名空间里的函数重名**，可以将函数定位为static
3. [修饰类成员](https://en.cppreference.com/w/cpp/language/static)
   - 修饰成员变量（相当于声明类属性）：所有对象能共享同一份数据；在**编译阶段分配内存**；其是类内声明，类外定义（**语法**）；访问时可通过对象或类名进行访问；也可以有访问权限的(private,
     protected, public)
   - 修饰成员函数（相当于声明**类方法**）：所有对象共享同一个函数；该方法不能访问实例属性；类内声明，类外定义；访问时可通过对象或类名进行访问；也可以有访问权限(private, protected, public)

## Qualifier

### [const qualifier](https://en.cppreference.com/w/cpp/language/cv)

- `const`用于描述对象的可读性

- 常被修饰的实体

| 实体       | 作用                                                         |
| ---------- | ------------------------------------------------------------ |
| 普通对象   | 该对象是只读的，尝试修改会报编译期错误                       |
| 指针       | 常指针，该指针的指向不发生变化                               |
| 类对象     | 该对象只能调用`const成员函数`（@[ref](http://faculty.cs.niu.edu/~mcmahon/CS241/Notes/const_objects_and_member_functions.html)）。析构函数和构造函数可以修改数据成员。 |
| 类成员函数 | 该对象的`*this`对象是被`const`修饰的，只能够调用其他`const`成员函数 |

- 示例

``` c++
// 类
class A
{
private:

    const int a;                // 常数据成员，只能在初始化列表赋值

public:

    // 构造函数
    A() : a(0) { };
    A(int x) : a(x) { };        // 初始化列表

    // const可用于对重载函数的区分
    int getValue();             // 普通成员函数
    int getValue() const;       // 常成员函数，该函数不得修改类中的任何数据成员的值

}; 

void function()
{

    // 对象
    A b;                        // 普通对象，可以调用全部成员函数、更新常成员变量
    const A a;                  // 常对象，只能调用常成员函数
    const A *p = &a;            // 指针变量，指向常对象
    const A &q = a;             // 指向常对象的引用

    // 指针
    char greeting[] = "Hello";
    char* p1 = greeting;                // 指针变量，指向字符数组变量
    const char* p2 = greeting;          // 指向常量的指针
    char* const p3 = greeting;          // 常指针（指针的指向不能发生改变）
    const char* const p4 = greeting;    // 指向常量的常指针

}

// 函数
void function1(const int Var); // 形参只读
void function2(const char* Var); // 形参为指针，指向的对象只读
void function3(char* const Var); // 形参为指针，指针为常指针
void function4(const int& Var); // 常量的引用

// 函数返回值
const int function5(); // 返回一个常数
const int* function6(); // 返回一个指针，指针指向常量
int* const function7(); // 返回一个常指针
```

## Specifier

### [constexpr](https://en.cppreference.com/w/cpp/language/constexpr)

- `constexpr`可修饰函数和变量，表明这些函数和变量的值**可能**能在编译期确定（evaluation），可以放在`constant expression`中
- 能在编译期进行evaluation的表达式称为[constant expression](https://en.cppreference.com/w/cpp/language/constexpr)

### explicit

- 告知编译器某个构造函数或者类型转换函数只能被显式调用，而不能进行隐式类型转换和拷贝初始化

### [noexcept](https://en.cppreference.com/w/cpp/language/noexcept_spec)

- 指示编译器该函数不会抛出异常，可进一步进行优化

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/eaIZEX1pn1JFAHBM.png!thumbnail)

- exception specification of overriding function is more lax than base version：原本是`noexcept`，继承后也需要是`noexcept`

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/4XmFkx42qA6XRCHa.png!thumbnail)


### [override](https://en.cppreference.com/w/cpp/language/override)

让编译器检查这个函数是否是虚函数，而且是否经过重写（`override`）

### [final](https://en.cppreference.com/w/cpp/language/final)

- 修饰类成员函数：指示这个成员函数是虚函数且不能再被重写，让编译器进行检查
- 修饰类：指示这个类不能被继承，让编译器进行检查

## [Variable](https://en.cppreference.com/w/cpp/language/basic_concepts)

被声明的对象和引用如果不是**非静态数据成员**则他们为变量
