# launch文件速查手册

## 常用的substitution args

- `$(env ENVIRONMENT_VARIABLE)` ：环境变量
- `$(find pkg)` ：返回包的绝对路径

````xml
$(find rospy)/manifest.xml
````

- `$(anon name)`：生成匿名节点名 

```xml
<node name="$(anon foo)" pkg="rospy_tutorials" type="talker.py" />
1. 出来的节点名形如：foo_helios_40184_5391378697892991360
```

- `$(arg)`

``` xml
<param name="foo" value="$(arg my_foo)" />
```

-  `$(dirname)` ：返回launch文件所在的路径

```xml
<include file="$(dirname)/other.launch" />
```



## if and unless属性

所有的tag都支持`if`和`unless`属性

- `if=value` (optional) 

  - If `value` evaluates to true, include tag and its contents. 

  `unless=value` (optional) 

  - Unless `value` evaluates to true (which means if `value` evaluates to false), include tag and its contents. 

```xml
<group if="$(arg foo)">
  <!-- stuff that will only be evaluated if foo is true -->
</group>

<param name="foo" value="bar" unless="$(arg foo)" />  
<!-- This param won't be set when "unless" condition is met -->
```



## 实例

### 实例

```xml
<launch>
  <!--节点名、包名、可执行文件名、输出日志到终端-->
  <node name="talker" pkg="rospy_tutorials" type="talker" ouput="screen"` />
</launch>
```



### 实例++

```xml
<launch>
  <!-- local machine already has a definition by default.
       This tag overrides the default definition with
       specific ROS_ROOT and ROS_PACKAGE_PATH values -->
  <machine name="local_alt" address="localhost" default="true" ros-root="/u/user/ros/ros/" ros-package-path="/u/user/ros/ros-pkg" />
  
    <!-- a basic listener node -->
  <node name="listener-1" pkg="rospy_tutorials" type="listener" />
  <!-- pass args to the listener node -->
  <node name="listener-2" pkg="rospy_tutorials" type="listener" args="-foo arg2" />
  <!-- a respawn-able listener node -->
  <node name="listener-3" pkg="rospy_tutorials" type="listener" respawn="true" />
  <!-- start listener node in the 'wg1' namespace -->
  <node ns="wg1" name="listener-wg1" pkg="rospy_tutorials" type="listener" respawn="true" />
  
    <!-- start a group of nodes in the 'wg2' namespace -->
  <group ns="wg2">
    <!-- remap applies to all future statements in this scope. -->
    <remap from="chatter" to="hello"/>
    <node pkg="rospy_tutorials" type="listener" name="listener" args="--test" respawn="true" />
    <node pkg="rospy_tutorials" type="talker" name="talker">
      <!-- set a private parameter for the node -->
      <param name="talker_1_param" value="a value" />
      <!-- nodes can have their own remap args -->
      <remap from="chatter" to="hello-1"/>
      <!-- you can set environment variables for a node -->
      <env name="ENV_EXAMPLE" value="some value" />
    </node>
  </group>
</launch>
```



## 导入参数到参数服务器

```xml
<launch>
  <param name="somestring1" value="bar" />
  <!-- force to string instead of integer -->
  <param name="somestring2" value="10" type="str" />

  <param name="someinteger1" value="1" type="int" />
  <param name="someinteger2" value="2" />

  <param name="somefloat1" value="3.14159" type="double" />
  <param name="somefloat2" value="3.0" />

  <!-- you can set parameters in child namespaces -->
  <param name="wg/childparam" value="a child namespace parameter" />

  <!-- upload the contents of a file to the server -->
  <param name="configfile" textfile="$(find roslaunch)/example.xml" />
  <!-- upload the contents of a file as base64 binary to the server -->
  <param name="binaryfile" binfile="$(find roslaunch)/example.xml" />

</launch>
```



## 设置顶层launch文件

```xml
<launch>
  <group name="wg">
    <include file="$(find pr2_alpha)/$(env ROBOT).machine" />
    <include file="$(find 2dnav_pr2)/config/new_amcl_node.xml" />
    <include file="$(find 2dnav_pr2)/config/base_odom_teleop.xml" />
    <include file="$(find 2dnav_pr2)/config/lasers_and_filters.xml" />
    <include file="$(find 2dnav_pr2)/config/map_server.xml" />
    <include file="$(find 2dnav_pr2)/config/ground_plane.xml" />

    <!-- The navigation stack and associated parameters -->
    <include file="$(find 2dnav_pr2)/move_base/move_base.xml" />
  </group>
</launch>
```

- 是将launch合并在一个文件启动还是分开启动，应具体情况具体分析。合并启动的优点是可以不用打开多个终端来启动launch；其缺点：有节点启动顺序要求时，不能满足。

  

## parameter

- 节选自`move_base.xml`

```xml
<node pkg="move_base" type="move_base" name="move_base" machine="c2">
  <!--此处的param是私有属性，带节点名命名空间-->
  <remap from="odom" to="pr2_base_odometry/odom" />
  <param name="controller_frequency" value="10.0" />  <!--move_base/controller_frequency-->
  <param name="footprint_padding" value="0.015" />
  <param name="controller_patience" value="15.0" />
  <param name="clearing_radius" value="0.59" />
  <rosparam file="$(find 2dnav_pr2)/config/costmap_common_params.yaml" command="load" ns="global_costmap" />
  <rosparam file="$(find 2dnav_pr2)/config/costmap_common_params.yaml" command="load" ns="local_costmap" />
  <rosparam file="$(find 2dnav_pr2)/move_base/local_costmap_params.yaml" command="load" />
  <rosparam file="$(find 2dnav_pr2)/move_base/global_costmap_params.yaml" command="load" />
  <rosparam file="$(find 2dnav_pr2)/move_base/navfn_params.yaml" command="load" />
  <rosparam file="$(find 2dnav_pr2)/move_base/base_local_planner_params.yaml" command="load" />
</node>
```



## Parameter覆写

```xml
<launch>
<include file="$(find 2dnav_pr2)/move_base/2dnav_pr2.launch" />
	<param name="move_base/local_costmap/resolution" value="0.5"/>
</launch>
```



## 碎碎念

1. roslaunch在启动节点前，会提前解析`substitution args`和导入param参数到参数服务器

2. tag详细说明的参考网站：[launch](http://wiki.ros.org/roslaunch/XML/launch), [node](http://wiki.ros.org/roslaunch/XML/node) , [machine](http://wiki.ros.org/roslaunch/XML/machine) , [include](http://wiki.ros.org/roslaunch/XML/include) , [remap](http://wiki.ros.org/roslaunch/XML/remap) , [env](http://wiki.ros.org/roslaunch/XML/env) , [param](http://wiki.ros.org/roslaunch/XML/param), [rosparam](http://wiki.ros.org/roslaunch/XML/rosparam), [group](http://wiki.ros.org/roslaunch/XML/group) , [test](http://wiki.ros.org/roslaunch/XML/test) , [arg](http://wiki.ros.org/roslaunch/XML/arg) 