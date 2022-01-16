#include <iostream>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>


/*
 * note: 回调函数的类型需要带时间戳，e.g.String不行；只有时间戳也不行，如Hearder
 * note: 似乎回调函数的参数都要是常指针
 */
using namespace message_filters;
typedef sync_policies::ApproximateTime<nav_msgs::Path, nav_msgs::Path>
    StringSyncPolicy;


void callback(const nav_msgs::PathConstPtr msgA,
              const nav_msgs::PathConstPtr msgB) {

  std::cout << msgA->poses[0].pose.position.x << "  "
            << msgB->poses[1].pose.position.x << std::endl;
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "temp");
  ros::NodeHandle nh;
  // 定义订阅器
  Subscriber<nav_msgs::Path> msg_subA(nh, "/HeaderA", 1);
  Subscriber<nav_msgs::Path> msg_subB(nh, "/HeaderB", 1);
  // 定义同步器
  Synchronizer<StringSyncPolicy> sync(StringSyncPolicy(10), msg_subA, msg_subB);
  // register 同步器回调函数（此处的bind，需要用boost，而不能用std库的）
  sync.registerCallback(boost::bind(&callback, _1, _2));
  ros::spin();
  return 0;
}