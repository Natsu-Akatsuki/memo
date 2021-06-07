# roslaunch python API

## 方案一：启动一个节点（无launch文件）

```python
import roslaunch

# 1.实例化节点对象
package = 'rqt_gui'
executable = 'rqt_gui
node = roslaunch.core.Node(package, executable)

# 实例化`launch 节点`对象
launch = roslaunch.scriptapi.ROSLaunch()
launch.start()

# 启动launch
process = launch.launch(node)
print process.is_alive()
process.stop()
```

 

## 方案二：启动一个节点（带launch文件）

```python
import roslaunch
import rospy

uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)
launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/haier/catkin_ws/src/testapi/launch/test_node.launch"])
launch.start()

rospy.sleep(3)
# 3 seconds later
launch.shutdown()
```

- 启动`launch`文件必须调用`configure_logging`

-  `roslaunch.parent.ROSLaunchParent`需要在`main thread`中启动



## 方案三：以command-line style启动节点

- 只启动单个launch文件

```python
import roslaunch

uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)

cli_args = ['./rviz.launch','vel:=2.19']   # 可相对路径
roslaunch_args = cli_args[1:]
roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)

parent.start()
```

- 启动多个launch文件

```python
import roslaunch

uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)

cli_args1 = ['pkg1', 'file1.launch', 'arg1:=arg1', 'arg2:=arg2']
cli_args2 = ['pkg2', 'file2.launch', 'arg1:=arg1', 'arg2:=arg2']
cli_args3 = ['pkg3', 'file3.launch']

roslaunch_file1 = roslaunch.rlutil.resolve_launch_arguments(cli_args1)   # 可用于解释包名
roslaunch_args1 = cli_args1[2:]

roslaunch_file2 = roslaunch.rlutil.resolve_launch_arguments(cli_args2)
roslaunch_args2 = cli_args2[2:]

roslaunch_file3 = roslaunch.rlutil.resolve_launch_arguments(cli_args3)

launch_files = [(roslaunch_file1, roslaunch_args1), (roslaunch_file2, roslaunch_args2), roslaunch_file3]

parent = roslaunch.parent.ROSLaunchParent(uuid, launch_files)

parent.start()
```





# DEBUG

1. pycharm 启动方案二时显示Invalid  tag: Cannot load command parameter ...

- 实测 conda activate 虚拟环境后启动pycharm即可
- 不用 pycharm的话，直接命令行启动是没问题的



# 其他

- 中文路径可正常运行