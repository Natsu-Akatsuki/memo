# 窗口

### [窗口设置](https://www.cnblogs.com/wiessharling/p/3750461.html)

- 具体效果可参考：[此处](https://github.com/baoboa/pyqt5/blob/master/examples/widgets/windowflags.py)





|                        |                                   |
| ---------------------- | --------------------------------- |
| 全屏显示               | showFullScreen()                  |
| 窗口最大化和窗口最小化 | showMaximized() / showMinimized() |
| 固定尺寸显示           | resize(h,w)                       |
| 设置最大尺寸           | setMaximumSize(w,h)               |
| 设置最小尺寸函数       | setMinimumSize(w,h)               |

---

**NOTE**

- 固定尺寸不影响全屏显示的效果



---



```
     但是 showFullScreen()只对顶级窗口有效果，对子窗口无效；
setWindowFlags (Qt::Window | Qt::FramelessWindowHint);
第一个参数表示此控件是窗口类型，
第二个表示去除边框，状态栏，没有框架。其实与showFullScreen() 函数的原理差不多。

 self.
子窗口全屏显示：
方法1：将要全屏的Qt主窗口中的子窗口调用函数setWindowFlags(Qt::Dialog) // setWindowFlags(Qt.QDialog)
方法2：调用setWindowFlags(Qt::Window)将其类型提升为顶级窗口模式,然后调用showFullScreen()函数将子窗口全屏显示。也就是先将子窗口全屏显示前设置为顶级窗口，然后进行全屏显示，注意顺序不能颠倒。因为showFullScreen()函数只对顶级窗口有效。


当然全屏后还要恢复正常，即调用setWindowFlags(Qt::subwindow)，或者setWindowFlags(Qt::Dialog),将子窗口设置为非顶级窗口，再调用showNormal()还原子窗口显示。直接调用mywindow.resize(x,y)是没有效果的。注意函数的调用顺序不能颠倒，否者不会还原。原因很简单，因为showNormal()也只对顶级窗口有效。所以必须将它设为非顶级窗口再调用。

```



```
from PyQt5.QtCore import Qt
# 窗口设置：置顶|
self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
# x, y, w, h
self....setGeometry(QtCore.QRect(30, 20, 311, 751))
```



#### 获取本机窗口大小

```
from PyQt5 import QtGui
QtGui.QGuiApplication.primaryScreen()
QGuiApplication.primaryScreen().availableVirtualGeometry().width()
QGuiApplication.primaryScreen().availableVirtualGeometry().height()
```





设置无边框

``` 
self.setWindowFlags(Qt.FramelessWindowHint)
```

