# ros Qt(python api)

* **用完一圈之后，不推荐使用这个rviz的python api，一是文档太少，二是坑很多**

- 本案例含：frame（rviz界面）、thickness_slider（滑动条）、按键；只显示 3D render

  

## [简例](http://docs.ros.org/en/lunar/api/rviz_python_tutorial/html/index.html)

- 对比原案例有删减

```
import roslib
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QPushButton, QSlider, QVBoxLayout, QWidget

import sys

# 导入所有的qt模块
# "python_qt_binding" 包可视情况导入PyQt和PySide
# RViz Python binding suse python_qt_binding internally, so you should use it here as well.
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *

# 导入rviz的binding库
import rviz
import rosnode


class RvizWidget(QWidget):
    def __init__(self, x, y, w, h, parent=None):
        QWidget.__init__(self, parent)
        # rviz.VisualizationFrame基本插件（带菜单、工具栏、状态栏、docked subpanels）
        self.frame = rviz.VisualizationFrame()

        # The "splash path" is the full path of an image file which
        # gets shown during loading.Setting it to the empty string
        # suppresses that behavior.
        self.frame.setSplashPath("")

        # 在.load调用前必须进行初始化。因为它会实例化一个很关键的类VisualizationManager
        self.frame.initialize()
        # self.frame.startUpdate()

        # 读取配置文档，文档对象，读取文档对象
        reader = rviz.YamlConfigReader()
        config = rviz.Config()
        reader.readFile(config, "config/qt.rviz")
        self.frame.load(config)

        # 隐藏顶部的主菜单、底部的状态栏、"hide-docks" buttons、the tall skinny buttons on the left and right sides of the
        self.frame.setMenuBar(None)
        self.frame.setStatusBar(None)
        self.frame.setHideButtonVisibility(True)

        # frame.getManager() 获取VisualizationManager实例，用于管理rviz实例
        self.manager = self.frame.getManager()

        # 步骤二：
        # 创建layout和放入rviz widget
        layout = QVBoxLayout()
        layout.addWidget(self.frame)
        self.setLayout(layout)
        self.setGeometry(QtCore.QRect(x, y, w, h))


# 实例化Qt并启动，start Qt's main event loop (app.exec_()).
if __name__ == '__main__':
    # test example
    app = QApplication(sys.argv)
    test01_rviz_widget = RvizWidget(0, 0, 0, 0)
    test01_rviz_widget.resize(1000, 1000)
    test01_rviz_widget.show()
    app.exec_()

    # [bugfix]：ugly patch for preventing from zombie process
    node_name_list = rosnode.get_node_names()
    rviz_node_list = []
    for node_name in node_name_list:
        if node_name.split('_')[0] == '/rviz':  # kill anonymous rviz node (e.g. /rviz_83219328)
            rviz_node_list.append(node_name)
    rosnode.kill_nodes(rviz_node_list)
```



## 未整理的线索

- 在实际调用中，发现rviz 的python API说明文档几乎没有，难以进行开发
- rviz的python API还有很多隐藏BUG，未能解决。比如退出Qt应用程序后，rviz节点将成为僵尸节点（即不能被rosnode kill掉，只能使用rosnode cleanup清理）；为了规避这个问题，我们使用了简例[bugfix]中的方法，应用程序退出后，python主进程退出前，先kill掉rviz节点（实测同样的例程，在c++中不存在这个问题，进程可以退出得很干净）
- 另外的问题，比如说不能够在Qt中的rviz中添加图像面板，否则会有段错误，程序会直接退出