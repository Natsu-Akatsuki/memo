import rospy
import threading
from std_msgs.msg import Int8


class ros_sub_or_pub:
    def __init__(self):
        rospy.init_node("ros_sub_or_pub", anonymous=False)
        self.test_sub = "/test"
        rospy.Subscriber(self.test_sub, Int8, self.test_cb)

    def test_cb(self, dummy_msg):
        print(dummy_msg)


if __name__ == "__main__":
    ros_sub_or_pub = ros_sub_or_pub()
    rospy.sleep(rospy.Duration(100))
    while not rospy.is_shutdown():
        rospy.spin()
