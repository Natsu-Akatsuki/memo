import cv2
from cv_bridge import CvBridge
import numpy as np
import rospy
import std_msgs.msg
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, Image
from ros_numpy.point_cloud2 import pointcloud2_to_xyzi_array, xyzi_numpy_to_pointcloud2, \
    xyzirgb_numpy_to_pointcloud2
import matplotlib.pyplot as plt


def get_calib_from_file(calib_file):
    '''
    :param calib_file:
    :return:(dict)
        'intri_matrix': (3,3)
        'distor': 4
        'extri_matrix': (4,4) or (3,4)
    '''
    with open(calib_file) as f:
        lines = f.readlines()

    # 相机内参
    intri_matrix = np.loadtxt('')

    # 相机畸变系数
    distor = np.loadtxt('')

    # 相机和激光雷达外参
    extri_matrix = np.loadtxt('')

    return {'intri_matrix': intri_matrix,  # (3,3)
            'distor': distor,  # 5
            'extri_matrix': extri_matrix  # (4,4)
            }


class Calibration(object):
    def __init__(self, calib_file):
        # 若非字典则从文件读取
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.intri_matrix = calib['intri_matrix']
        self.distor = calib['distor']
        self.extri_matrix = calib['extri_matrix']

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_camera(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_camera: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        # 如果外参矩阵为（4，4）则(N,4) (4,4) -> (N,4) 所以出来的点要截取
        pts_camera = np.dot(pts_lidar_hom, self.extri_matrix.T)[:, 0:3]
        return pts_camera

    def camera_to_img(self, pts_camera):
        """
        本函数不直接越界判别
        :param pts_camera: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_camera = np.dot(pts_camera, self.intri_matrix.T)
        pts_camera_depth = pts_camera[:, 2]
        pts_img = (pts_camera[:, 0:2].T / pts_camera[:, 2]).T  # (N, 2)

        return pts_img, pts_camera_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
                pts_camera_depth
        """
        pts_camera = self.lidar_to_camera(pts_lidar)
        pts_img, pts_camera_depth = self.camera_to_img(pts_camera)
        return pts_img, pts_camera_depth

    # camera -> lidar
    def camera_to_lidar(self, pts_camera):
        """
        Args:pts_camera
            pts_camera: (N, 3)
        Returns:
            pts_lidar: (N, 3)
        """
        pts_camera_hom = self.cart_to_hom(pts_camera)
        pts_lidar = (pts_camera_hom @ np.linalg.inv(self.extri_matrix.T))[:, 0:3]
        return pts_lidar


if __name__ == '__main__':
    cal = {
        'intri_matrix': np.loadtxt(''),
        'distor': np.loadtxt(''),
        'extri_matrix': np.loadtxt('')
    }
    Calibration(cal)
