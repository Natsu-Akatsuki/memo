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


class TFPointcloud:
    def __init__(self):
        rospy.init_node("dynamic_tf")
        self.sub_topic = "/livox/lidar"
        self.pub_topic = "/livox/tf_lidar"

        self.pointcloud_pub = rospy.Publisher(
            self.pub_topic, PointCloud2, queue_size=10
        )
        self.pointcloud_sub = rospy.Subscriber(self.sub_topic, PointCloud2, self.pointcloud_cb)

    def pointcloud_cb(self, msg):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "rslidar"
        pointcloud_np = pointcloud2_to_xyzi_array(msg, remove_nans=True)
        pointcloud_msg = xyzi_numpy_to_pointcloud2(pointcloud_np, header)
        self.pointcloud_pub.publish(pointcloud_msg)

if __name__ == "__main__":
    tf_pointcloud = TFPointcloud()
    rospy.spin()
