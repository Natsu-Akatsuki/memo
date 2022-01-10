import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2_parser
import numpy as np
import argparse


def save_pointcloud(pointcloud_np, export_format="npy", path_file="pointcloud.npy"):
    if export_format == "npy":
        np.save(path_file, pointcloud_np)
    elif export_format == "pcd":
        pass
    elif export_format == "bin":
        pass

def pointcloud_callback(msg):
    pointcloud_ros = pc2_parser.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
    pointcloud_np = np.asarray(list(pointcloud_ros), dtype=np.float32)

    # normalize the intensity
    # pointcloud_np[:, 3] = pointcloud_np[:, 3] / 255


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--topic', type=str, default='/livox/lidar', help='specify the topic of subscriber')
    args = parser.parse_args()
    pointcloud_topic = args.topic
    print("The subscribed topic is" + pointcloud_topic)

    rospy.init_node('subscribe_pointcloud', anonymous=False)
    rospy.Subscriber(pointcloud_topic, PointCloud2, pointcloud_callback)
    rospy.spin()
