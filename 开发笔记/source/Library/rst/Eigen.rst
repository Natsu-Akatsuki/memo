
`Eigen <https://eigen.tuxfamily.org/dox/modules.html>`_
===========================================================

Version
-------

查看当前Eigen的版本

.. prompt:: bash $,# auto

   # for apt package
   $ dpkg -l | grep eigen

   # 看头文件宏
   $ more /usr/include/eigen3/Eigen/src/Core/util/Macros.h  | grep VERSION

CMake
-----

.. code-block:: cmake

   find_package(Eigen3 REQUIRED)
   include_directories(${EIGEN3_INCLUDE_DIRS})

Header
------

.. code-block:: C++

   #include <Eigen/Dense>

其内部实际包含了：

.. code-block:: c++

   #include "Core"
   #include "LU"
   #include "Cholesky"
   #include "QR"
   #include "SVD"
   #include "Geometry"
   #include "Eigenvalues"

Matrix
------

Accessors
^^^^^^^^^


* `使用括号操作符 <https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html>`_

.. code-block:: c++

   #include <iostream>
   #include <Eigen/Dense>

   int main()
   {
     Eigen::MatrixXd m(2,2);
     m(0,0) = 3;
     m(1,0) = 2.5;
     m(0,1) = -1;
     m(1,1) = m(1,0) + m(0,1);
   }

Assignment
^^^^^^^^^^


* 逗号操作符初始化

.. code-block:: C++

   A << 1, 2, 3,     // Initialize A. The elements can also be
        4, 5, 6,     // matrices, which are stacked along cols
        7, 8, 9;     // and then the rows are stacked.
   B << A, A, A;     // B is three horizontally stacked A's.
   A.fill(10);       // Fill A with all 10's.

Construct
^^^^^^^^^


* `Matrix模板类的基本参数 <https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html>`_
* 默认构造函数（不进行动态内存分配和值的初始化）

.. code-block:: c++

   using namespace Eigen;

   // 创建静态数组（进行空间分配）
   Matrix<double, 3, 3> A;               // Fixed rows and cols. Same as Matrix3d.
   Matrix3f P;                           // 创建3×3方阵
   Vector3f x, y, z;                     // 3x1 float matrix.
   RowVector3f a, b, c;                  // 1x3 float matrix.

   // 创建动态数组，不进行内存分配
   Matrix<double, 3, Dynamic> B;         // Fixed rows, dynamic cols.
   Matrix<double, Dynamic, Dynamic> C;   // Full dynamic. Same as MatrixXd.
   Matrix<double, 3, 3, RowMajor> E;     // 设置行优先
   VectorXd v;                           // Dynamic column vector of doubles

   // 创建随机矩阵
   Eigen::MatrixXi mat = Eigen::MatrixXi::Random(2, 3);

   // 单位阵
   MatrixXd mat = MatrixXd::Identity(5, 3);
   // Resizes to the given size, and writes the identity expression
   mat.setIdentity(5, 3);

   // 1矩阵
   MatrixXd mat = MatrixXd::Ones(5, 3);
   mat.setZero(5, 3);

   // 0矩阵
   MatrixXd mat = MatrixXd::Zero(5, 3);
   mat.setZero(5, 3);

   // 常数矩阵
   MatrixXd::Constant(5, 3, 10);

Equation
^^^^^^^^

.. code-block:: C++

   // Solve Ax = b. Result stored in x. Matlab: x = A b.
   x = A.ldlt().solve(b);  // A sym. p.s.d.    #include <Eigen/Cholesky>
   x = A.llt() .solve(b);  // A sym. p.d.      #include <Eigen/Cholesky>
   x = A.lu()  .solve(b);  // Stable and fast. #include <Eigen/LU>
   x = A.qr()  .solve(b);  // No pivoting.     #include <Eigen/QR>
   x = A.svd() .solve(b);  // Stable, slowest. #include <Eigen/SVD>
   // .ldlt() -> .matrixL() and .matrixD()
   // .llt()  -> .matrixL()
   // .lu()   -> .matrixL() and .matrixU()
   // .qr()   -> .matrixQ() and .matrixR()
   // .svd()  -> .matrixU(), .singularValues(), and .matrixV()

Hstack
^^^^^^

.. code-block:: c++

   auto a = Eigen::MatrixXd::Random(3, 2);
   auto b = Eigen::MatrixXd::Ones(3, 1);
   Eigen::MatrixXd c(a.rows(), a.cols() + b.cols());
   c << a, b;

Index
^^^^^

`Slice <https://eigen.tuxfamily.org/dox/group__TutorialSlicingIndexing.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* start from ``Eigen 3.4``

.. code-block:: C++

   Eigen::MatrixXd mat = Eigen::MatrixXd::Random(10, 4);
   // mat[:,0:2]，seq：为左闭右闭
   // note: 这种方法得到的矩阵为深拷贝
   Eigen::MatrixXd submat = mat(Eigen::all, Eigen::seq(0, 2));

`Map <https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   // vector(9) -> eigen(3, 3)
   // Eigen::Matrix<double, 3, 3> camera_intrinsic(camera_intrinsic_yaml.data());
   Eigen::Matrix<double, 3, 3> camera_intrinsic = Eigen::Map<Eigen::Matrix<double, 3, 3> >(camera_intrinsic_yaml.data());

Mathematics
^^^^^^^^^^^

.. code-block:: c++

   // Matrix-vector.  Matrix-matrix.   Matrix-scalar.
   y  = M*x;          R  = P*Q;        R  = P*s;
   a  = b*M;          R  = P - Q;      R  = s*P;
   a *= M;            R  = P + Q;      R  = P/s;
                      R *= Q;          R  = s*P;
                      R += Q;          R *= s;
                      R -= Q;          R /= s;


   MatrixXd mat = MatrixXd::Random(3, 3);
   cout << mat << endl;
   // 伴随矩阵
   cout << mat.adjoint() << endl;
   // 转置矩阵
   cout << mat.transpose() << endl;
   // 获取对角线上的元素
   cout << mat.diagonal() << endl;
   // 求逆
   cout << mat.reverse() << endl;

`Reshape <https://eigen.tuxfamily.org/dox/group__TutorialReshape.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 返回的是view（浅拷贝）
* 对于resize而言，如果resize前后元素个数不一样，则会返回一个0初值的数组

.. code-block:: c++

   MatrixXi m = Matrix4i::Random();
   auto m1 = m.reshaped(2, 8);

   // in-place reshape
   m.resize(2,8);

Size
^^^^

.. code-block:: C++

   // Eigen                 // comments
   x.size()                 // 向量的大小（元素的个数）
   C.rows()                 // 行数
   C.cols()                 // 列数

Type Cast
^^^^^^^^^

.. code-block:: C++

   MatrixXd mat = MatrixXd::Random(2, 2);
   // if the original type equals destination type, no work is done
   mat.cast<double>();
   mat.cast<float>();
   mat.cast<int>();
   mat.real();
   mat.imag();

   // note that for most operations Eigen requires all operands to have the same type
   MatrixXf F = MatrixXf::Zero(3, 3);
   // mat += F;                // illegal in Eigen.
   mat += F.cast<double>(); // F converted to double and then added (generally, conversion happens on-the-fly)

Q&A
---

Reference
---------


* 
  `Eigen short ASCII reference <http://eigen.tuxfamily.org/dox-devel/AsciiQuickReference.txt>`_

* 
  `Official Tutorial <https://eigen.tuxfamily.org/dox/GettingStarted.html>`_
