# [PCA](http://blog.codinglabs.org/articles/pca-tutorial.html)

## 目标

- 以对数据进行降维或者压缩为目标，将一组N维向量降到K维（将N维数据放到K维空间）。方法就是选择K个**单位正交基**，原始数据变换到这组基后，各字段间的协方差矩阵为0（数据间相互独立，没有太多冗余），字段自己的方差（数据不会重叠和冗余）尽可能大。
- 对平面进行PCA找法向量
- 基于数据的分布，生成一个坐标系：对三维平面进行PCA分析，PCA就是由数据确立一个正交坐标系（**let the data tell us how to construct a coordinate system**），其中第一个轴沿着方差最大的方向；第二个轴沿着方差次小的方向；最后一个轴垂直于它们。

## 推导

### 数据的协方差矩阵

将数据（列向量）按列组成矩阵$X$（已经归一化），易有
$$
C=\frac{1}{m}XX^T=
\begin{bmatrix}
 var(a) & cov(a,b)\\
 cov(b,a)  & var(a)
\end{bmatrix}
$$
设原始数据的协方差矩阵为$C$，其**变换后**的协方差矩阵$D$为对角阵（协方差为0），设$P$是一组基按行组成的矩阵，变换后的数据为$Y$。
$$
D = & \frac{1}{m}YY^T \\
  = & \frac{1}{m}(PX)(PX)^T \\
  = & \frac{1}{m}(PX)(PX)^T \\
  = & \frac{1}{m}PXX^TP^T \\
  = & PCP^T
$$

## 算法

应用：获取激光点的法向量（已用KD树提取最近邻的激光点，得到邻域激光点集）

步骤一：计算数据的协方差

步骤二：对协方差进行SVD分解

步骤三：提取**特征值最小所对应的特征向量**作为法向量

## 代码

```python
import numpy as np
import open3d as o3d


def PCA(data):
    points_xyz = data

    # 1.零均值（是否零均值不影响最终的结果）
    points_mean = np.mean(points_xyz, axis=0)
    points_normal = points_xyz - points_mean

    # 2.求协方差矩阵
    covariance = np.cov(points_normal.transpose())
    # 等价于 covariance = np.dot(points_normal.transpose(), points_normal) / points_normal.shape[0]

    # 3.求协方差矩阵进行svd分解（默认已进行了排序）
    eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(covariance)

    return eigenvalues, eigenvectors


def read_pointcloud(p_path):
    """
    read pointcloud from str file(only use x,y,z dimension)
    :param p_path:
    :return:
    """
    points = np.loadtxt(p_path, dtype=str)
    points_np = np.zeros((points.size, 6))
    for i in range(points.size):
        points_np[i, :] = np.fromstring(points[i], sep=',', dtype=float)
    return points_np[:, 0:3]


def main():
    # 加载原始点云
    file_name = '../data/bed_0001.txt'
    points_xyz = read_pointcloud(file_name)

    print('total points number is:', points_xyz.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points_xyz)
    point_cloud_vector = v[:, 0]  # 点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)

    # 可视化点云和前两个成分
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)

    point = [[0, 0, 0], v[:, 0], v[:, 1], v[:, 2]]
    # 由点构成线，[0, 1]代表点集中序号为0和1的点组成线，即原点和两个成分向量划线
    lines = [[0, 1], [0, 2], [0, 3]]
    # 为不同的线添加不同的颜色
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # 构造线段，Create a LineSet from given points and line indices
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    normals = []
    neighbor_num = 10
    for i in range(points_xyz.shape[0]):
        _, k_index, _ = pcd_tree.search_knn_vector_3d(point_cloud.points[i], neighbor_num)
        w, v = PCA(np.asarray(points_xyz[k_index]))
        # 对应特征值最小的特征向量
        normals.append(v[:, 2])

    normals = np.array(normals, dtype=np.float64)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries(width=800, height=800, geometry_list=[point_cloud, line_set])


if __name__ == '__main__':
    main()

```

## 数学公式

以下描述的统计量默认加上样本二字（如方差->样本方差）

### 方差

- 定义：（数据-均值）的均值

$$
Var(a)=\frac{1}{m}\sum_{i=1}^m{(a_i-\mu)^2}
$$

### 协方差

- 定义：

$$
Cov(a,b)=\frac{1}{m}\sum_{i=1}^m{(a_i-\mu_a)(b_i-\mu_b)}
$$

- 矩阵表示：X(m,n)表示数据, 数据有m个属性（字段），n个样本：

$$
Cov(a,a)=\frac{XX^T}{n}
$$

## Q&A

### 对一个三维平面进行PCA，为什么平面法向量是最小的特征值对应的特征向量？

- PCA能够根据**数据的分布**来生成一个新的坐标轴；那由于数据是分布在平面上的，则新的两个坐标轴是落在平面上的
- 在正交坐标系的约束下，最后一个轴就只能是法向量了。
