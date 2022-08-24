# [DistanceAndSimilarity](https://zhuanlan.zhihu.com/p/138107999)

## 术语

### 去量纲

> 标准化可以去除单位对相似度的影响

去除数据的单位

## 欧氏距离

```python
import torch
from torch import nn

# 二范数
pdist = nn.PairwiseDistance(p=2)
input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
output = pdist(input1, input2) # (100)
```

## 曼哈顿距离

## 马氏距离

马氏距离(Mahalanobis distance)：可以定义为**两个服从同一分布**并且其**协方差矩阵**为Σ的**随机变量**之间的差异程度。如果协方差矩阵为单位矩阵，那么马氏距离就简化为欧氏距离，如果协方差矩阵为对角阵，则其也可称为正规化的欧氏距离。
$$
D= \sqrt{((a-b)^TΣ^{-1}(a-b)} \\
\Sigma=QD^DQ^T \\
\Sigma^{-1}=QD^{-\frac{1}{2}}D^{-\frac{1}{2}}Q^T \\
D= \sqrt{((a-b)^TΣ^{-1}(a-b)} = \sqrt{((a-b)^TQD^{-\frac{1}{2}}D^{-\frac{1}{2}}Q^T(a-b)} \\
= \sqrt{{[D^{-\frac{1}{2}}Q^T(a-b)]^T}[D^{-\frac{1}{2}}Q^T(a-b)]} = \sqrt{z^Tz}
$$

其中$D$对角阵，相关的元素为标准差

```python
from scipy.spatial import distance
import numpy as np

# The inverse of the covariance matrix.
iv = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
result = distance.mahalanobis([2, 0, 0], [0, 1, 0], iv)
print(result)

a = np.array([2, 0, 0]).reshape(-1, 1)
b = np.array([0, 1, 0]).reshape(-1, 1)
c = np.sqrt((a - b).T @ np.array(iv) @ (a - b))
print(c)
```

- 马氏距离可以认为是强化版的欧氏距离，统一了数据的量纲，排除了原本量纲的影响，考虑了数据的分布
- 马氏距离可以认为是正交基下的欧式距离
- 马氏距离度量了一个点到一个分布的距离（这个分布假定为高斯分布，参数为协方差矩阵$\Sigma$和均值$\mu$）
- 马氏距离有两种一种是单点马氏距离；另一种是双点马氏距离
- 马氏距离的应用场景：剔除离群点

### 应用

### [判断一个点到一个分布之间的距离](https://www.csdn.net/tags/NtjaEg5sMDEzOTctYmxvZwO0O0OO0O0O.html)

- 作为一种统计距离，用于找到离群点（find outliers in a set of data）

```python
import pandas as pd
import numpy as np


def mahalanobis(x=None, data=None, cov=None):
    """ Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns. e.g. (500,3)
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    note: 此处不用scipy API的原因：其输入需要一维arr
    """
    x_minus_mu = x - data.mean()
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/diamonds.csv'
df = pd.read_csv(filepath)
df = df.iloc[:, [0, 4, 6]]  # 选取类型为数值的三列
df_x = df[['carat', 'depth', 'price']].head(500)
df_x['mahala'] = mahalanobis(x=df_x, data=df[['carat', 'depth', 'price']])
```

### 标准化欧氏距离

```python
from scipy.spatial import distance
import numpy as np

SIGMA = np.array([[2, 1], [1, 2]])
q = [0, 0]
x_1 = [-3.5, -4]
d_1 = distance.seuclidean(q, x_1, np.diag(SIGMA))
```

## 余弦相似度

```python
import torch
from torch import nn

# 二范数
input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)
```

## KL散度

用来衡量两个概率分布的相似性（值越小，两个概率分布越相似）；以下公式可描述为：以$P$分布为权，以两个分布**自信息差**为待加权对象

数学定义：$D_{KL}(P||Q)=E_{X\sim{P}}{[log \frac{P(x)}{Q(x)}]}=E_{X \sim P}{[log{P(x)-log(Q(x)}]}$

```python
import torch
import torch.nn.functional as F

x = torch.randn((4, 5))
y = torch.randn((4, 5))

logp_x = F.log_softmax(x, dim=-1)
p_y = F.softmax(y, dim=-1)

# D_KL(py||px)
kl_sum = F.kl_div(logp_x, p_y, reduction='sum')
kl_mean = F.kl_div(logp_x, p_y, reduction='mean')

print(kl_sum, kl_mean)
```

## Q&A

### 欧式距离vs余弦相似度

前者更倾向于空间中的几何距离，后者倾向于度量方向的相似性

### 欧式距离小就代表两个样本相似了吗

- 不能，它没有考虑不同量纲的影响。

- 欧氏距离假定了量纲对相似度的影响是一样的。但比如1kg重量和1mm的长度，显然是对相似度的贡献是不一样的。

### 标准化和归一化

- 归一化的作用：即将数据的范围映射到特定的范围，比如[0,1]或者[-1,1]，在图像领域上则是[0,255]

- 标准化的作用：将数据的分布转换为0均值，1方差的分布（统一了数据的单位，都是1个单位方差）

### 马氏距离的优点

马氏距离的计算不受量纲的影响；马氏距离可以排除变量之间的相关性的干扰

### 马氏距离的解析

一、原始数据

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/vI1vj.png" alt="enter image description here" style="zoom: 80%;" />

二、由原始数据得到坐标轴

The origin will be at the centroid of points. The first coordinate axis will extend along the "spine" of the points, which is any direction in which the variance is the greatest. The second coordinate axis will extend perpendiculaly to the first one.

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/h9WzB.png" alt="enter image description here" style="zoom:80%;" />

三、进行缩放，以标准差作为单位(68-95-99.7原则)

- about two-thirds of the points should be **within one unit of the origin** (along the axis)
- about 95% should be within two units.

![enter image description here](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/fMqVj.png)

四、进一步缩放，得到单位元

![enter image description here](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/qgQkT.png)





## 参考资料

[1] [马氏距离的解析（stackexchange）](https://stats.stackexchange.com/questions/62092/bottom-to-top-explanation-of-the-mahalanobis-distance)

[2] [马氏距离（知乎）](https://zhuanlan.zhihu.com/p/46626607)