# 用rosrun调用包中的程序

- 实际上使用默认的package.xml(即`catkin_create_pkg`创建的package.xml)

  ```bash
  // 记得修改相关执行路径
  $ catkin_create_pkg <package_name>   // src路径下执行
  $ catkin_make
  $ source 工作空间
  
  $ rosrun  <package_name> <可执行文件>
  ```
  - 只有有可执行权限的文件，其文件名才能被命令行补全 
  - python脚本记得在首行添加解释器的路径

  

