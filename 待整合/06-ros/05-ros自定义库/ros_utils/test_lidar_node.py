import numpy as np
from cv_bridge import CvBridge
import cv2
import rospy

import std_msgs.msg
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, Image
from ros_numpy.point_cloud2 import (
    pointcloud2_to_xyzi_array,
    xyzi_numpy_to_pointcloud2,
    xyzirgb_numpy_to_pointcloud2,
)


class TestLidarNode:
    def __init__(self):
        rospy.init_node("test_lidar_node", anonymous=False)
        self.pointcloud_topic_sub = "/sensing/lidar/top/pointcloud_raw"
        self.pointcloud_sub = rospy.Subscriber(
            self.pointcloud_topic_sub, PointCloud2, self.lidar_cb
        )

    def lidar_cb(self, pointcloud_msg):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "livox"
        self.pointcloud_np = pointcloud2_to_xyzi_array(pointcloud_msg, remove_nans=True)
        pass


if __name__ == "__main__":
    test_lidar_node = TestLidarNode()
    rospy.spin()
