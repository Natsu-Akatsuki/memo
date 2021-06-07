# ros环境下导入别人的包

- 在ros的环境下，有时需要导入package下的python模块。

- 基本的解决步骤为：1. 创建python模块 2. 将python模块安装到ros空间下 3. 导入安装包
- 参考教程，参考[link](https://roboticsbackend.com/ros-import-python-module-from-another-package/)



## 简例

### 创建python模块

- 构建的目录树（遵循ros规范）如下所示，文件内容参考[link](https://roboticsbackend.com/ros-import-python-module-from-another-package/)

```
└── my_robot_common   # ros 包名
    ├── CMakeLists.txt
    ├── package.xml
    ├── setup.py
    └── src
        └── my_robot_common   # python包名
            ├── import_me_if_you_can.py
            └── __init__.py
```



## 安装

### 编写`setup.py`文件

```python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['my_robot_common'],
    package_dir={'': 'src'}
)
setup(**d)
```



### 编写`CMakeLists.txt` 文件

```cmake
cmake_minimum_required(VERSION 2.8.3)
project(my_robot_common)
find_package(catkin REQUIRED COMPONENTS
  rospy
)
catkin_python_setup()   # 用于调用当前CMakeLists文件所在目录下的setup.py
catkin_package()			   
```



