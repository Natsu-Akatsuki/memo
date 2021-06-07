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


class ros_sub_or_pub:
    def __init__(self):

        rospy.init_node("ros_sub_or_pub", anonymous=False)
        # 发布器
        self.pointcloud_topic_pub = "/pointcloud_pub"
        self.img_topic_pub = "/img_pub"
        self.pointcloud_pub = rospy.Publisher(
            self.pointcloud_topic_pub, PointCloud2, queue_size=10
        )
        self.img_pub = rospy.Publisher(self.img_topic_pub, Image, queue_size=10)

        # 订阅器
        self.pointcloud_topic_sub = "/rslidar_points"
        self.img_topic_sub = "/usb_cam/image_raw"
        self.img_sub = rospy.Subscriber(self.img_topic_sub, Image, self.img_cb)
        self.pointcloud_sub = rospy.Subscriber(
            self.pointcloud_topic_sub, PointCloud2, self.livox_cb
        )

        self.bridge = CvBridge()

    def publish_pointcloud(self, pointcloud_np, color_mode=False):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "livox"

        if color_mode:
            pointcloud_msg = xyzirgb_numpy_to_pointcloud2(pointcloud_np, header)
        else:
            pointcloud_msg = xyzi_numpy_to_pointcloud2(pointcloud_np, header)
        self.pointcloud_pub.publish(pointcloud_msg)

    def publish_img(self, img_np):
        img_msg = self.bridge.cv2_to_imgmsg(cvim=img_np, encoding="passthrough")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = "livox"
        self.img_pub.publish(img_msg)

    def img_cb(self, img_msg):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "livox"
        self.img_cv2 = self.bridge.imgmsg_to_cv2(img_msg)

    def livox_cb(self, pointcloud_msg):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "livox"
        self.pointcloud_np = pointcloud2_to_xyzi_array(pointcloud_msg, remove_nans=True)
        pass


if __name__ == "__main__":
    ros_sub_or_pub = ros_sub_or_pub()
    rospy.spin()

# np.save('/home/ah_chung/data_sample/pcd/6.npy', self.pointcloud_np)
# cv2.imwrite('/home/ah_chung/data_sample/img/6.png', self.img_cv2)
