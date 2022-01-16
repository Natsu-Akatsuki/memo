# launch-guide

* 术语说明

[![image-20210807095158418](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210807095158418.png)](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210807095158418.png)

* roslaunch在启动节点前，会提前解析`substitution args`和导入参数(param)到参数服务器

## [substitution args](http://wiki.ros.org/roslaunch/XML#substitution_args)

## [if and unless attributes](http://wiki.ros.org/roslaunch/XML#if_and_unless_attributes)

.. hint:: 所有的tag都支持 ``if`` 和 ``unless`` 属性

## [tag reference](http://wiki.ros.org/roslaunch/XML#Tag_Reference)

## 实例

* 普通用例

```xml
<launch>
  <node name="节点名" pkg="包名" type="可执行文件名" ouput="输出日志到终端" />
  <node name="talker" pkg="rospy_tutorials" type="talker" ouput="screen" />
</launch>
```

* rosbag

```xml
<launch>
    <!-- run rosbag -->
    <node pkg="rosbag" type="play" name="play" output="screen" args="$(env HOME)/test.bag -l">
        <remap from="/camera/raw_img" to="/camera/depth/image_raw" />
    </node>
</launch>
```

* rviz

```xml
<launch>
    <!-- Run Rviz -->
    <node pkg="rviz" type="rviz" name="rviz" args="$(env HOME)/test01.rviz" />
</launch>
```

* [官方实例参考](http://wiki.ros.org/roslaunch/XML#Example_.launch_XML_Config_Files)

## 文件规范

* 代码格式化：vscode中使用 `XML Formatter` 进行格式化，缩进为4空格

* 使用substitution tag

* 添加注释和使用docs属性来添加描述信息

[![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/xUZKgvoo1W7666ia.png!thumbnail)](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/xUZKgvoo1W7666ia.png!thumbnail)

## 拓展插件

* vscode插件：ROS snippets（代码块/live template）
* vocode插件：Xml Formatter（格式化xml文件）
