import rospy
from msg_test01.msg import test01

rospy.init_node('msg_test01', anonymous=False)
test01 = test01()
print(test01.x)
