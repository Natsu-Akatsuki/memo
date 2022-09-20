.. role:: raw-html-m2r(raw)
   :format: html


STL
===

Algorithm
---------

.. list-table::
   :header-rows: 1

   * - 函数
     - 作用
     - 例子
   * - `\ **accumulate** <https://en.cppreference.com/w/cpp/algorithm/accumulate>`_
     - 输入一个序列(range)和给定初值，计算累积值
     - —
   * - `copy <https://en.cppreference.com/w/cpp/algorithm/copy>`_
     - 拷贝一个序列
     - —
   * - `fill <https://en.cppreference.com/w/cpp/algorithm/fill>`_
     - 输入一个序列(range)和值，将序列的所有元素设置为该值
     - —
   * - `fill_in <https://en.cppreference.com/w/cpp/algorithm/fill_n>`_
     - —
     - —
   * - `find <https://en.cppreference.com/w/cpp/algorithm/find>`_
     - 输入一个序列和某个值，有该值则返回该值对应的迭代器，否则返回last
     - —
   * - `sort <https://en.cppreference.com/w/cpp/algorithm/sort>`_
     - 输入一个序列(range)，对这个range的元素进行排序（默认为升序排列）
     - std::sort(s.begin(), s.end());
   * - `transform <https://en.cppreference.com/w/cpp/algorithm/transform>`_
     - 输入一个序列(range)，将函数作用于这个range上的每个元素/修改每个元素，得到一个新的序列
     - —
   * - `unique <https://en.cppreference.com/w/cpp/algorithm/unique>`_
     - 输入一个序列，删除接连重复的元素
     - —
   * - **max**
     - 返回最大元素
     - —
   * - **max_element**
     - 输入一个序列，返回最大值所对应的迭代器
     - —
   * - reverse
     - 字符串反转
     - std::reverse(a.begin(), a.end());
   * - swap
     - 交换两个元素
     - —
   * - swap
     - 数据交换
     - swap(a, b);


`Bind <https://en.cppreference.com/w/cpp/utility/functional/bind>`_
-----------------------------------------------------------------------

bind给人感觉是一种\ ``函数适配器``\ （function adapter），能修改可调用对象对外的接口：比如说一个可调用对象原本有5个形参，经过bind函数后对外的形参就变成3个；又或者经过bind的封装可以\ `生成多个具有不同默认参数的可调用对象 <[修改函数形参的默认实参](https://www.geeksforgeeks.org/bind-function-placeholders-c/>`_\ )

.. code-block:: c++

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


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210809203143984.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210809203143984.png
   :alt: image-20210809203143984



* `bind传参时默认是拷贝传参 <https://blog.csdn.net/zzzyyyyyy66/article/details/80285723>`_

`Containers <https://en.cppreference.com/w/cpp/container>`_
---------------------------------------------------------------


* 模板库官方文档索引

.. list-table::
   :header-rows: 1

   * - 序列容器
     - 关联容器
     - 容器适配器
   * - `array <https://en.cppreference.com/w/cpp/container/array>`_
     - `set <https://en.cppreference.com/w/cpp/container/set>`_\ ，\ `unordered_set <https://en.cppreference.com/w/cpp/container/unordered_set>`_
     - stack
   * - `vector <https://en.cppreference.com/w/cpp/container/vector>`_
     - `map <https://en.cppreference.com/w/cpp/container/map>`_
     - queue
   * - `deque <https://en.cppreference.com/w/cpp/container/deque>`_
     - `unordered_map <https://en.cppreference.com/w/cpp/container/unordered_map>`_
     - `priority_queue <https://en.cppreference.com/w/cpp/container/priority_queue>`_
   * - `forward_list <https://en.cppreference.com/w/cpp/container/forward_list>`_\ ，\ `list <https://en.cppreference.com/w/cpp/container/list>`_
     - —
     - —



* 容器（container）是存储对象的对象，同时这种对象需要满足一定的\ `要求 <https://en.cppreference.com/w/cpp/named_req/Container>`_
* `容器底层数据结构一览表 <https://interview.huihut.com/#/?id=stl-%e5%ae%b9%e5%99%a8>`_ 红黑树 or 哈希表
* 为什么有这么多不同的序列容器？不同的序列容器的实现是不同的，在不同的任务中会表现出不同的时间复杂度

`Type <https://en.cppreference.com/w/cpp/container>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* c++的容器包含了序列容器（\ ``sequence container``\ ），有序关联（\ ``associatice containter``\ ）容器，无序关联容器（\ ``unordered associative containers``\ ），容器适配器（\ ``container adaptors``\ ）
* 序列容器的元素可以进行序列访问
* 关联容器（unordered_map）、序列容器中的（\ ``array``\ ，\ ``vector``\ ，\ ``deque``\ ）都支持\ ``[] operator``\ （通过下标索引运算符支持随机访问）

Feature
^^^^^^^

.. list-table::
   :header-rows: 1

   * - 容器
     - 功能
     - —
   * - `array <http://c.biancheng.net/view/411.html>`_
     - 静态数组，长度不可变
     - —
   * - vector
     - 能动态的调整长度
     - —
   * - set
     - 关联容器；数据存放有序（从小到大存放）
     - set迭代器指向的是const对象，不能修改这个元素
   * - unordered_set
     - 数据存放无序
     - —
   * - map
     - map会按\ ``key``\ 进行自动排序；
     - map的每一个元素是一个pair
   * - unordered_map
     - —
     - 支持[ ]索引；\ :raw-html-m2r:`<br />`\ （键值的自动插入）如果索引了不存在的键，c++的字典会自动添加该键（类似python）；\ :raw-html-m2r:`<br />`\ unordered_map在插入情况下少的时候用到


Modifier
^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 容器
     - Modifier（头）
     - Modifier（尾）
     - 访问
   * - ``vector``\ （动态数组）
     - —
     - ``push_back()``\ :raw-html-m2r:`<br />`\ ``emplace_back()``\ :raw-html-m2r:`<br />`\ ``pop_back()``
     - ``front()``\ :raw-html-m2r:`<br />`\ ``back()``
   * - ``deque``
     - ``push_front()``\ :raw-html-m2r:`<br />`\ ``pop_front()``
     - ``push_back()``\ :raw-html-m2r:`<br />`\ ``emplace_back()``\ :raw-html-m2r:`<br />`\ ``pop_back()``
     - ``front()``\ :raw-html-m2r:`<br />`\ ``back()``
   * - ``queue`` （队列）
     - ``pop()``\ :raw-html-m2r:`<br />`
     - ``push()``
     - ``front()``
   * - ``list``\ （双向链表）
     - ``push/emplace_front/back``\ :raw-html-m2r:`<br />`\ ``pop_front/back()``\ :raw-html-m2r:`<br />`
     - ``push_back()``\ :raw-html-m2r:`<br />`\ ``emplace_back()``\ :raw-html-m2r:`<br />`
     - ``front``\ :raw-html-m2r:`<br />`\ ``back``
   * - ``string``\ （字符串）
     - 
     - ``push_back``\ （单个元素）\ :raw-html-m2r:`<br />`


Vector
^^^^^^


* vector的模板形参可否是内置数组？

不能，根据\ `cppreference <https://en.cppreference.com/w/cpp/container/vector>`_\ ，该形参类型需要满足 ``CopyAssignable`` 和 ``CopyConstructible`` 两种属性。而内置数组类型不满足 ``CopyAssignable`` 的属性。

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/qvdJoCaDaGAGjHu9.png!thumbnail" alt="img" style="zoom:67%;" />`


* `如何对vector<vector\<int&gt;>进行emplace_black <https://stackoverflow.com/questions/20391632/how-to-use-stdvectoremplace-back-for-vectorvectorint>`_

.. code-block:: c++

   # 不能从{}类型推导出std::initializer_list()类型，所以需要显式指明
   vec.emplace_back(std::initializer_list<int>{1,2});


* `clang-tidy推荐用emplace_back，它和push_back的区别在于? <https://www.zhihu.com/question/438004429>`_

两者构造元素的方式不一样，前者的效率会更高：前者追加生成vector的元素，是把emplace_back的实参传递给元素的构造函数的形参，然后\ **直接构造对象**\ 。没有临时对象的构造和析构。后者追加生成vector的元素，是通过\ **拷贝或移动构造函数**\ 来生成，因此需要先创建一波临时对象。多了临时变量的构造和释放。

.. code-block:: c++

   // example
   vector.emplace_back(1, 2);
   vector.push_back(MyClass(1, 2));

Chrono
------

.. code-block:: c++

   #include <time.h>
   #include <chrono>
   #include <thread>

   // sleep
   std::this_thread::sleep_for(std::chrono::milliseconds(3000));

   // 计时，方法一：
   auto start = std::chrono::system_clock::now();
   // TODO
   auto end = std::chrono::system_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);

   // 方法二：高精度
   auto startTime = std::chrono::high_resolution_clock::now();
   // TODO
   auto endTime = std::chrono::high_resolution_clock::now();
   float totalTime = std::chrono::duration<float, std::milli> (endTime - startTime).count();


   // 方法三：
   #include <chrono>
   #include <cstdio>

   const auto& start = std::chrono::steady_clock::now();
   const auto& end = std::chrono::steady_clock::now();
   double duration = (end - start).count() / 1000000.0;
   printf("  processing:  %9.3lf [msec]\n", duration);

CString
-------

.. list-table::
   :header-rows: 1

   * - 
     - 
   * - ``std::strcpy``
     - 拷贝一个const string（包括\ ``null terminator``\ ）到某个空间
   * - 
     - 
   * - 


.. code-block:: cpp

   // char -> string
   string str;
   str = to_string(8); // 8 -> "8"
   str = char(8);  // 8 -> '\008'
   str.push_back(char(8 + '0')) // push_back后面只接字符

IOstream
--------


* 
  需要创建一个流（stream）对象来管理文件的读写

* 
  流即c++用于管理文件和内存的模板类；文件流对象有打开和关闭的状态，处于打开状态后无法再次打开，可以用\ ``is_open``\ 来判断该对象是否有绑定/关联一个文件

* 
  C++处理文件，有三个类模板，\ ``basic_ifstream``\ ，\ ``basic_ofstream``\ ，\ ``basic_fstream``

* 
  其析构函数会调用close来取消关联，所以不一定要显式close

Format
^^^^^^


* 全局格式化

.. code-block:: c++

   // 显示"+"符号
   std::cout.setf(std::iso_base::showpos)
   // 输出长度（被触发后会重置）
   std::cout.width(10)
   // 占位所填充值
   std::cout.fill('.')


* 局部格式化（操纵符）

manupilator ≠ operator

.. code-block:: c++

   #include <iomanip>
   std::cout << std::setw(10) << x << std::endl;

Stream Status
^^^^^^^^^^^^^

.. code-block:: c++

   #include <fstream>
   #include <iostream>
   std::ifstream file(filename, std::ios::in | std::ios::binary);
   std::cout << std::cin.good() << std::cin.fail() << std::cout.bad() << file.eof();

Write
^^^^^

.. code-block:: c++

   #include <fstream>
   using namespace std;
   int main() {
     // 覆写文件（以二进制形式）
     // ofstream outFile("myfile.txt", std::ios::out | std::ios::ate |std::ios::binary);
     // 追加数据
     // ofstream outFile("myfile.txt", std::ios::out | std::ios::app);
     ofstream outFile("myfile.txt");
     outFile << "ABC";
     return 0;
   }

Q&A
^^^


* 判断一个文件是否存在

.. code-block:: c++

   #include <fstream>
   #include <iostream>
   using namespace std;

   void isFileExist() {
     std::string engine_path = "绝对路径";
     std::ifstream fs(engine_path);
     if (fs.is_open()) {
       cout << "file exists" << endl;
     } else {
       cout << "file doesn't exist" << endl;
     }
   }


* `创建目录 <https://en.cppreference.com/w/cpp/filesystem/create_directory>`_
* 不调用\ ``close()``\ 有什么影响？

使用close是为了释放/解绑其关联的文件，不调用的话就不能重新关联/绑定一个新的文件。另外，如果流对象销毁了，其析构函数是会自动地调用close方法


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/myl8KDpygMRlqx2X.png!thumbnail
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/myl8KDpygMRlqx2X.png!thumbnail
   :alt: img



* 
  `多个斜杠影响文件的读取吗？ <https://en.cppreference.com/w/cpp/filesystem/path>`_

  :raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220115110602802.png" alt="image-20220115110602802" style="zoom:67%;" />`

Mutex
-----


* lock

.. code-block:: cpp

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

Q&A
^^^


* B线程没有获取到锁的时候，B线程会做什么操作？最简单的是在spin（wait）

Reference
^^^^^^^^^


* `Mutex tutorial and example <https://nrecursions.blogspot.com/2014/08/mutex-tutorial-and-example.html>`_
