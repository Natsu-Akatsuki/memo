import time

import numpy as np
import open3d as o3d


def load_pcd_data(file_path):
    pts = []

    with  open(file_path, 'r') as f:
        data = f.readlines()

    pts_num = data[9].strip('\n').split(' ')[-1]
    pts_num = int(pts_num)
    for line in data[11:]:
        line = line.strip('\n')
        xyzi = line.split(' ')
        x, y, z, intensity = [eval(i) for i in xyzi[:4]]
        pts.append([x, y, z, intensity])

    assert len(pts) == pts_num
    pointcloud = np.zeros((pts_num, len(pts[0])), dtype=np.float32)
    for i in range(pts_num):
        pointcloud[i] = pts[i]
    return pointcloud


def load_pcd_data_o3d(file_path):
    """
    note: 新版本的open3D已支持读取intensity
    :param file_path:
    :return:
    """
    pointcloud = o3d.io.read_point_cloud(file_path)
    pointcloud = np.asarray(pointcloud.points)
    return pointcloud


def load_bin_data(file_path):
    pointcloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return pointcloud


def load_pcd_data_pcl(file_path):
    import load_pcd_file_pcl
    pointcloud = load_pcd_file_pcl.load_pcd_file(file_path, True)
    return pointcloud


if __name__ == '__main__':
    pass
