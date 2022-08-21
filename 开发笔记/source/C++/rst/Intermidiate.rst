.. role:: raw-html-m2r(raw)
   :format: html


Intermidiate
============

Compiler
--------


* 
  并不是所有编译器都能够支持新标准

  可通过\ `cppreference <https://en.cppreference.com/w/>`_\ 来查询编译器是否支持特定的标准

  :raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210928215406043.png" alt="image-20210928215406043" style="zoom:50%;" />`

Endian
------


* 
  目前接触的系统一般是小端模式即低字节存储低地址

  .. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220813143214031.png
     :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220813143214031.png
     :alt: image-20220813143214031


* 
  在Python里面一般会用\ ``>``\ 或\ ``<``\ 来进行大小端的区别

.. code-block:: python

   dt = np.dtype('b')  # byte, native byte order
   dt = np.dtype('>H') # big-endian unsigned short
   dt = np.dtype('<f') # little-endian single-precision float
   dt = np.dtype('d')  # double-precision floating-point numbe

判断两个变量是否是相同类型/某个变量是否某个类型
-----------------------------------------------

.. code-block:: c++

   #include <type_traits>
   using namespace std; 

   int main() {

       vector<int> arr(5);
       // 注意使用尖括号
       cout << is_same_v<decltype(arr.size()), unsigned long> << endl;

   }

`全局对象的构造函数先于主函数执行 <https://blog.csdn.net/Y673582465/article/details/72878053>`_
---------------------------------------------------------------------------------------------------

如何判断一个类型是左值还是右值
------------------------------

结合decltype的语法规则和\ `cpp insight <https://cppinsights.io/>`_\ 可以判断其\ `value category <https://en.cppreference.com/w/cpp/language/decltype>`_


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210816232635787.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210816232635787.png
   :alt: img



.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210816233508495.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210816233508495.png
   :alt: img


如何在编译期避免一个函数被调用？
--------------------------------

使用function-body\ ``=delete``\ 可提示编译器避免调用某个函数(\ `ref <https://en.cppreference.com/w/cpp/language/function>`_\ )

.. code-block:: c++

   #include <iostream>

   using namespace std;
   void test01() = delete;

   int main() {
     test01();  // error: Call to deleted function 'test01'
     return 0;
   }

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210925085531545.png" alt="image-20210925085531545" style="zoom:67%;" />`

constexpr all things
--------------------

使用constexpr让编译器将一些处理逻辑放在编译期中处理，一方面提高运行时速度，另一方面可以让bug在编译期就能提早浮现

c++中别人的程序为什么不用c风格的类型转换操作？
----------------------------------------------

c风格的类型转换需要按顺序尝试一系列c++风格的类型转，而用c++风格的类型转换则对症下药，不用尝试，一步到位

获取类型信息
------------

.. code-block:: c++

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

在写函数时，形参使用指针好还是引用好？
--------------------------------------

引用会更好，因为可以不用判断参数是否为 ``null``

为什么在位运算时会有整型提升？
------------------------------

硬件对int型数据有更好的处理效率（是最常使用的数据类型），因此在进行位运算时会将数据转换为整型

.. code-block:: c++

   int main() {

       char a = 0x00;
       char b = 0x01;
       auto c = a & b;
       cout << is_same_v<int, decltype(c)> << endl;  // true

   }

std::enable_if实现中的成员是什么？
----------------------------------

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210816175135379.png" alt="image-20210816175135379" style="zoom:50%;" />`


* 
  c++中的类的成员除了\ ``data member``\ ，\ ``memeber function``\ ，还可以有\ ``member typedefs``\ 等，对其的使用需要使用\ ``className::type``\ 语法（不能使用\ ``object.type``\ ）

  .. code-block:: c++

     struct TypeClass {
       typedef int IntType;
     };

     int main() {
       TypeClass typeclass;
       // scope resolution的左操作数应为class, namespace, enumeration
       TypeClass::IntType a = 1;
       return 0;
     }

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210816174218380.png" alt="image-20210816174218380" style="zoom: 50%;" />`


* `member typedefs <https://en.cppreference.com/w/cpp/language/class>`_

Segement Fault
--------------


* 段错误（生成的汇编代码没有RET指令）：定义了一个有返回值的函数，但RELEASE实现提供返回值


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220429130701631.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220429130701631.png
   :alt: image-20220429130701631


Undefined Behavior
^^^^^^^^^^^^^^^^^^


* 未定义行为不一定立刻触发


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220805213829446.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220805213829446.png
   :alt: image-20220805213829446

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
