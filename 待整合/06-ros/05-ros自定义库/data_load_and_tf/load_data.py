import numpy as np


class PcdIo:
    @staticmethod
    def load_pcd_data(file_path, remove_nan=True):
        """
        pcd点云 -> numpy
        Args:
            remove_nan:
            file_path:
        """

        with open(file_path, 'r') as f:
            data = f.readlines()

        # note: 这种字符型的可以直接由loadtxt读取，不需要逐行遍历，但实测速度很慢
        # pointcloud = np.loadtxt(data[13:], dtype=np.float32)
        # todo: 目前数据段的识别需人工识别和修正，可更自动化

        pointcloud = []
        for line in data[13:]:
            line = line.strip().split()
            pointcloud.append([np.float_(line[0]), np.float_(line[1]),
                               np.float_(line[2]), np.float_(line[3])])
        pointcloud = np.asarray(pointcloud, dtype=np.float32)
        if remove_nan:
            pointcloud = pointcloud[~np.isnan(pointcloud[:, 0])]

        return pointcloud

    @staticmethod
    def save_pcd_data(pointcloud, filename):
        """
        numpy -> pcd点云
        Args:
            pointcloud:
            filename:
        """
        with open(filename, 'w') as f:
            f.write("# .PCD v.7 - Point Cloud Data file format\n")
            f.write("VERSION .7\n")
            f.write("FIELDS x y z intensity\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
            f.write("WIDTH {}\n".format(pointcloud.shape[0]))
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write("POINTS {}\n".format(pointcloud.shape[0]))
            f.write("DATA ascii\n")

            for i in range(pointcloud.shape[0]):
                f.write(
                    str(pointcloud[i][0]) + " " +
                    str(pointcloud[i][1]) + " " +
                    str(pointcloud[i][2]) + " " +
                    str(pointcloud[i][3]) + "\n")


if __name__ == '__main__':
    import time

    start = time.time()
    # 80ms
    pointcloud = PcdIo.load_pcd_data('/home/helios/calibration/0001.pcd')
    start = time.time()
    # 60ms
    PcdIo.save_pcd_data(pointcloud, '/home/helios/calibration/00011.pcd')
    print('TIME(ms)  is=', 1000 * (time.time() - start))
