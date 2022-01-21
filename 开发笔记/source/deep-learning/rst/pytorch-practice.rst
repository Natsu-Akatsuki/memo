.. role:: raw-html-m2r(raw)
   :format: html


pytorch-practice
================

common API(for python)
----------------------

Tensor操作
^^^^^^^^^^

torch.full
~~~~~~~~~~

给Tensor填值

.. code-block:: python

   >>> torch.full((2, 3), 3.141592)
   tensor([[ 3.1416,  3.1416,  3.1416],
           [ 3.1416,  3.1416,  3.1416]])

矩阵相乘
~~~~~~~~


* 含minibatch

.. code-block:: python

   # A(10,3,4)  B(10,4,5)  C(10,3,5)
   C = A @ B

截断
~~~~

.. code-block:: python

   import torch
   out = torch.clamp(torch.Tensor([34, 54, -22, -9]), min=3)
   # out：tensor([34., 54.,  3.,  3.])

数学运算
~~~~~~~~

.. code-block:: python

   # 倒数：reciprocal
   <Tensor>.reciprocal()
   # 自然对数：返回（1+输入值）的自然对数
   torch.log1p(...)
   # 随机数：从标准正态分布中生成值
   torch.randn()

数据集划分
^^^^^^^^^^

.. code-block:: python

   from torch.utils.data import random_split
   subsetA, subsetB = random_split(range(集合的大小), [子集A需要划分的大小, 子集B的大小], generator = torch.Generator().manual_seed(42))

   # e.g. 将有10350帧数据的数据集均分出一个训练集和验证集
   # training_set, validation_set = random_split(range(10350), [5175, 5175])

.. note:: random_split() got an unexpected keyword argument 'generator'，该API在1.7版本后才有


common API(for c++)
-------------------

`Tensor创建 <https://pytorch.org/cppdocs/notes/tensor_creation.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Tensor索引 <https://pytorch.org/cppdocs/notes/tensor_indexing.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`查看版本信息 <https://pytorch.org/cppdocs/notes/versioning.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   #include <torch/torch.h>
   #include <iostream>

   int main() {
     std::cout << "PyTorch version from parts: "
       << TORCH_VERSION_MAJOR << "."
       << TORCH_VERSION_MINOR << "."
       << TORCH_VERSION_PATCH << std::endl;
     std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
   }

Q&A
---

difference between Aten and torch library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* Torch库算是Aten库的上层封装

:raw-html-m2r:`<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220118102156583.png" alt="image-20220118102156583" style="zoom:67%;" />`


* 其构建的Tensor有差别吗？\ `没有 <https://github.com/pytorch/pytorch/issues/14257>`_\ ，建议用torch namespace


.. image:: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220118102605290.png
   :target: https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220118102605290.png
   :alt: image-20220118102605290


.. code-block:: c++

   // the same
   torch::Tensor tensor_ones = torch::ones({2, 3});
   at::Tensor tensor_ones_at = torch::ones({2, 3});

version compatibility
^^^^^^^^^^^^^^^^^^^^^


* **error:** no matching function for call to ‘\ **torch::jit::RegisterOperators::RegisterOperators(const char [28], <**\ unresolved overloaded function type>)’

..

   ≥1.4的版本的pytorch将 ``torch::jit::RegisterOperators::RegisterOperators`` 改为了\ ``torch::RegisterOperators::RegisterOperators``


转置时数据在内存上是怎么变换的？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tensor的数据存储时本质上是存储在一个一维数组上（称为Storage）；要索引相关的值，则需要stride等属性。当进行转置时，Storage这个一维数组上的数值不变，只是修改stride属性。
