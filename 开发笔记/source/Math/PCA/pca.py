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
