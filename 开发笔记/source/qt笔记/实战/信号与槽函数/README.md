#### slot 函数（回调/触发时）

```python
self.sender()   				   # return the signal id
self.sender().objectName()		   # 获取触发槽函数的对象的名字（str）
```



- 未解：

```
# 此处不加装饰器，会报错
@pyqtSlot()
def ui_launch(self):
	"""（ui层级）触发处理launch进程"""
    sender = self.sender()
    object_name = sender.name

    if sender.checked:
      self.ui_launch_signal.emit(object_name)
    else:        
      self.ui_shutdown_signal.emit(object_name)
```



#### 信号

- 自定义信号（函数触发）

```python
import sys
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QMainWindow, QApplication


class Communicate(QObject):
    closeApp = pyqtSignal()

class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.c = Communicate()
        self.c.closeApp.connect(self.close)

        self.setGeometry(300, 300, 290, 150)
        self.setWindowTitle('Emit signal')
        self.show()

    def mousePressEvent(self, event):
        self.c.closeApp.emit()  # 软件触发自定义信号


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
```

