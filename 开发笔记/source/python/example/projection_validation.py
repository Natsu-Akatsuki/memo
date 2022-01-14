# coding=utf-8
import matplotlib.pyplot as plt

import cv2
import numpy as np

SCALE_FACTOR = 1.5
import colorsys


def undistort_projection(points, intrinsic_matrix, extrinsic_matrix):
    """
    :param points:
    :param (3x3 matrix) intrinsic_matrix:
        fx s x0
        0 fy y0
        0  0  1
    :param (4x4 matrix) extrinsic_matrix:
    :return:
    """
    points = np.column_stack([points, np.ones_like(points[:, 0])])
    # 外参矩阵
    points = np.matmul(extrinsic_matrix, points.T)
    # 内参矩阵
    points = np.matmul(intrinsic_matrix, points[:3, :], ).T
    # 深度归一化
    points[:, :2] /= points[:, 2].reshape(-1, 1)
    return points


def pc_to_img(pc, img, extrinsic_matrix, intrinsic_matrix):
    projection_points = undistort_projection(pc[:, :3], intrinsic_matrix, extrinsic_matrix)

    projection_points = np.column_stack([np.squeeze(projection_points), pc[:, 3:]])
    projection_points = projection_points[np.where(
        (projection_points[:, 0] > 0) &
        (projection_points[:, 0] < img.shape[1]) &
        (projection_points[:, 1] > 0) &
        (projection_points[:, 1] < img.shape[0])
    )]

    board = np.zeros_like(img)
    board[...] = img[..., ::-1]

    # colors = plt.get_cmap('gist_ncar_r')(projection_points[:, 2] / np.max(projection_points[:, 2]))

    # for idx in range(3):
    #     board[np.int_(projection_points[:, 1]), np.int_(projection_points[:, 0]), 2 - idx] = colors[:, idx] * 255
    dist = projection_points[:, 2]
    dist_normalize = (dist - dist.min()) / (dist.max() - dist.min())
    color = [[int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 1)] for hue in dist_normalize]
    pts_2d = projection_points[:, :2].astype(np.int32).tolist()

    for (x, y), c in zip(pts_2d, color):
        cv2.circle(img, (x, y), 1, [c[2], c[1], c[0]], -1)

    cv2.imshow('Projection', img)
    cv2.waitKey(0)

    return board


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4]) \
           & (points[:, 2] >= limit_range[2]) & (points[:, 2] <= limit_range[5])
    return mask


def load_pcd_data(file_path):
    """
    pcd点云->numpy
    Args:
        file_path:
    """
    with open(file_path, 'r') as f:
        data = f.readlines()
        # note: 这种字符型的可以直接由loadtxt读取
    pointcloud = np.loadtxt(data[11:], dtype=np.float32)
    pointcloud[np.isnan(pointcloud)] = 0
    return pointcloud[:, :3]


if __name__ == '__main__':
    img = cv2.imread("/home/helios/cam-lid/images/0016.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pointcloud = load_pcd_data("/home/helios/cam-lid/pointCloud/0016.pcd")

    # 约束点云空间
    limit_range = [0, -20, -3, 70.4, 20, 3]
    mask = mask_points_by_range(pointcloud, limit_range)
    pointcloud = pointcloud[mask]
    # 读取配置文档
    intrinsic_matrix = np.loadtxt("/home/helios/cam-lid/intrinsic_matrix.txt")
    distortion = np.loadtxt("/home/helios/cam-lid/distortion.txt")
    extrinsic_matrix = np.loadtxt("/home/helios/cam-lid/extrinsic_matrix.txt")
    # # 消除图像distortion
    img = cv2.undistort(img, intrinsic_matrix, distortion)
    pc_to_img(pointcloud, img, extrinsic_matrix, intrinsic_matrix)
