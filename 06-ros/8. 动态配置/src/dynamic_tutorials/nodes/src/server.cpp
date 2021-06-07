#include <ros/ros.h>

// 导入必要的头文件
#include <dynamic_reconfigure/server.h>
#include <dynamic_tutorials/TutorialsConfig.h>

// 参数动态更新时的回调函数
void callback(dynamic_tutorials::TutorialsConfig &config, uint32_t level) {
  ROS_INFO("Reconfigure Request: %d %f %s %s %d", 
            config.int_param, config.double_param, 
            config.str_param.c_str(), 
            config.bool_param?"True":"False", 
            config.size);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "dynamic_tutorials");

  // 传入配置类型，只要server一直存在，则会一直接收配置更新的request
  dynamic_reconfigure::Server<dynamic_tutorials::TutorialsConfig> server;
  // 构建回调函数并传入server
  dynamic_reconfigure::Server<dynamic_tutorials::TutorialsConfig>::CallbackType f;
  f = boost::bind(&callback, _1, _2);
  server.setCallback(f);

  ROS_INFO("Spinning node");
  ros::spin();
  return 0;
}