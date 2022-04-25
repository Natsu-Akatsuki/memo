# RosPracticePlus

## 回调函数

为了提高节点程序的清晰度，将回调函数封装为类，而跟节点文件进行解耦。

```c++
class GNSSSubscriber {
  public:
    GNSSSubscriber(ros::NodeHandle& nh, std::string topic_name, size_t buff_size);
    GNSSSubscriber() = default;
    void ParseData(std::deque<GNSSData>& deque_gnss_data);

  private:
    void msg_callback(const sensor_msgs::NavSatFixConstPtr& nav_sat_fix_ptr);

  private:
    ros::NodeHandle nh_;
    ros::Subscriber subscriber_;

    std::deque<GNSSData> new_gnss_data_;
};

// 接收和处理信息
void GNSSSubscriber::msg_callback(const sensor_msgs::NavSatFixConstPtr& nav_sat_fix_ptr) {
    GNSSData gnss_data;
    gnss_data.time = nav_sat_fix_ptr->header.stamp.toSec();
    gnss_data.latitude = nav_sat_fix_ptr->latitude;
    gnss_data.longitude = nav_sat_fix_ptr->longitude;
    gnss_data.altitude = nav_sat_fix_ptr->altitude;
    gnss_data.status = nav_sat_fix_ptr->status.status;
    gnss_data.service = nav_sat_fix_ptr->status.service;

    new_gnss_data_.push_back(gnss_data);
}

// 读取信息
void GNSSSubscriber::ParseData(std::deque<GNSSData>& gnss_data_buff) {
    if (new_gnss_data_.size() > 0) {
        gnss_data_buff.insert(gnss_data_buff.end(), new_gnss_data_.begin(), new_gnss_data_.end());
        new_gnss_data_.clear();
    }
}
```

节点文件的调用：

```c++
// 定义类对象指针
std::shared_ptr<GNSSSubscriber> gnss_sub_ptr = 
    std::make_shared<GNSSSubscriber>(nh, "/kitti/oxts/gps/fix", 1000000);
ros::Rate rate(100);
while (ros::ok()) {
    ros::spinOnce();
    //取数据
    gnss_sub_ptr->ParseData(gnss_data_buff);
    rate.sleep();
}
```

## 参考资料

- [https://zhuanlan.zhihu.com/p/105512661](任乾知乎)
