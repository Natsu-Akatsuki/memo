# pytorch-practice

## common API

### Tensor操作

#### torch.full

给Tensor填值，

```python
>>> torch.full((2, 3), 3.141592)
tensor([[ 3.1416,  3.1416,  3.1416],
        [ 3.1416,  3.1416,  3.1416]])
```

#### 矩阵相乘

- 含minibatch

```python
# A(10,3,4)  B(10,4,5)  C(10,3,5)
C = A @ B 
```

#### 截断

```python
import torch
out = torch.clamp(torch.Tensor([34, 54, -22, -9]), min=3)
# out：tensor([34., 54.,  3.,  3.])
```

#### 数学运算

- 

````python
# 倒数：reciprocal
<Tensor>.reciprocal()
# 自然对数：返回（1+输入值）的自然对数
torch.log1p(...)
# 随机数：从标准正态分布中生成值
torch.randn()
````

- 自然多少



## Q&A

### difference between Aten with torch library



### version compatibility

- **error:** no matching function for call to ‘**torch::jit::RegisterOperators::RegisterOperators(const char [28], <**unresolved overloaded function type>)’ 

> ≥1.4的版本的pytorch将 `torch::jit::RegisterOperators::RegisterOperators` 改为了`torch::RegisterOperators::RegisterOperators`      



### 转置时数据在内存上是怎么变换的？

Tensor的数据存储时本质上是存储在一个一维数组上（称为Storage）；要索引相关的值，则需要stride等属性。当进行转置时，Storage这个一维数组上的数值不变，只是修改stride属性。

