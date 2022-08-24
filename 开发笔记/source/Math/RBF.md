# 高斯核函数

又称为径向基核函数（Radial Basis Function）

## [函数形式](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)

$$
K(x,x')=exp^{-\frac{\left \| x-x' \right \| ^2}{2\sigma^2 } }
$$

$x$：向量A（样本A）
$x'$：向量B（样本B）
$σ$ ：带宽，控制径向作用范围（$\sigma$越大，局部影响范围越大）

计算高维空间下，**两样本的相似度**（二范数距离）

## 作用

计算高维空间下，两样本的相似度（度量为高维数据的内积）

## 代码

### opencv-python

```python
import cv2

gaussian_kernel = cv2.getGaussianKernel(ksize=3, sigma=2)
print("1D gaussian_kernel:\n", gaussian_kernel)
print("2D gaussian_kernel:\n", (gaussian_kernel * gaussian_kernel.T))

"""
1D gaussian_kernel:
 [[0.31916777]
 [0.36166446]
 [0.31916777]]
2D gaussian_kernel:
 [[0.10186806 0.11543164 0.10186806]
 [0.11543164 0.13080118 0.11543164]
 [0.10186806 0.11543164 0.10186806]]
"""
```

### opencv-c++

```cpp
#include <opencv2/imgproc/imgproc.hpp>
cv::Mat getGaussianKernel(int rows, int cols, double sigmax, double sigmay) {
  auto gauss_x = cv::getGaussianKernel(cols, sigmax, CV_32F);
  auto gauss_y = cv::getGaussianKernel(rows, sigmay, CV_32F);
  return gauss_x * gauss_y.t();
}

auto mat = getGaussianKernel(3, 3, 2.0, 2.0);
```

### pytorch-c++

三种中其速度最慢

```c++
#include <torch/torch.h>
/**
 * @brief 生成高斯核
 * @param kernel_size
 * @param sigma
 */
at::Tensor Postprocess::setGaussianKernel(int kernel_size = 3, float sigma = 2.0) {

  // 生成坐标网格点 (kernel_size, kernel_size, 2)
  at::Tensor x_coord = at::arange(0, kernel_size);
  auto grid = at::meshgrid({x_coord, x_coord});
  auto grid_x = grid[0];
  auto grid_y = grid[1];
  auto xy_coordinate = at::stack({grid_x, grid_y}, -1);

  double mean_coordinate = (kernel_size - 1) / 2.0;
  double variance = sigma * sigma;

  // 该二维高斯核函数 = x核函数和y核函数的乘积 (s,s)
  at::Tensor gaussian_kernel =
      at::exp(at::sum(at::pow(xy_coordinate - mean_coordinate, 2), -1) /
              (-2.0 * variance));

  // 高斯核函数的归一化
  gaussian_kernel = gaussian_kernel / at::sum(gaussian_kernel_);
  gaussian_kernel = gaussian_kernel.reshape({kernel_size, kernel_size});
  return gaussian_kernel;
}
```

## Q&A

### 为什么高斯核对应的映射能将有限维的数据映射到无限维上？

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/bfa7b22bf398f7520b42b9674d342f50_b.png)

### rangenet++使用高斯函数权值的作用？

给image上不同的领域分配不同的权值

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/T9845gtM7VJjXc9D.png!thumbnail" alt="img" style="zoom:67%;" />

## 知识点补充

### 特征映射

- 特征映射$\varphi$ (feature mapping)用于将低维的数据转换为高维的数据，从而可以找到适当的线性关系，让数据更好分类/分离
- feature mapping≠kernel function，kernel function是用来算升维数据的内积的

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/RxOClGPLXsCK3GXx.png!thumbnail" alt="img" style="zoom: 50%;" />

### 特征空间下数据属性

- 高维度空间下的距离 (distance in the feature space)

$$
{\left \| \varphi(X)-\varphi(Y) \right \| ^2} = \\
(\varphi(X)-\varphi(Y))^T(\varphi(X)-\varphi(Y)) = \\ 
\varphi(X)^T\varphi(X)-\varphi(X)^T\varphi(Y)-\varphi(Y)^T\varphi(X)+\varphi(Y)^T\varphi(Y)= \\
{\left \|\varphi(X) \right \| ^2}+{\left \| \varphi(Y) \right \| ^2} - 2{\left \| \varphi(X)\varphi(Y) \right \| ^2} = \\
K(XX)+K(YY)-2K(XY)
$$

 其中：$X$和$Y$都是行向量，所以$X^TY=Y^TX$

- 高维度空间下的角度 (angle in the feature space)

$$
cos\theta=-\frac{\left \|\varphi(X)  \varphi(Y) \right \| ^2}
{ \left | \varphi(X) \right| \left | \varphi(Y) \right |           } = \\ 
\frac {K(XY)} { \sqrt {K(X)} \sqrt{ K(Y)} }
$$

.. note:: 余弦角=向量的内积/向量模的积

- 内积矩阵（任意两个高维点的内积     ）

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220220102242866.png" alt="image-20220220102242866" style="zoom: 33%;" />

### 归一化的高斯函数

$$
p(x) = \frac {1}{\sqrt{2\pi} \sigma}exp(-\frac{1}{2} \frac{(x-\mu)}{\sigma^2})
$$

$$
K(x,x')=exp^{-\frac{\left \| x-x' \right \| ^2}{2\sigma^2 } }
$$

