# QT设计

## 设计界面逻辑

## 设计业务逻辑

### 界面组件

#### 主窗口

- 详看`01主窗口.py`，包含增设logo，设计窗口大小，边框设计

#### 按钮

```c++
QPushButton *output_startNavi_button_;
output_startNavi_button_ = new QPushButton("保存轨迹");
```

#### 标签

- 图片相关：详看`02显示图片`
- 文本相关：

```python
label = QLabel("Hello World!")     # 创建标签
self.label.setText("字符串")    		 # 在标签中显示字体
label.show()  										  # 显示标签
label.text()                   							 # 或标签页的文本
```

#### 弹窗

```python
result = QtWidgets.QMessageBox.question(self,
                                        "Confirm Exit...",
                                        "Are you sure you want to exit ?",
                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

if result == QtWidgets.QMessageBox.Yes:
        pass
else:
        pass
```

#### 状态栏

- 一般位置位于主窗口下方

```
self.statusBar()
self.statusBar().showMessage("...")
```

#### 常用属性配置

```python
<widget>.setEnabled(False)
<widget>.setStyleSheet("QGroupBox{\n"
                                    "color: black;\n"
                                    "border: transparent;\n"
                                    "}")
```

- [setStyleSheet](https://doc.qt.io/qt-5/stylesheet-reference.html#checked-ps)

```css
background-color: red;    # 背景颜色

/*边框*/
border-style: outset;         
border-width: 2px;			 # 边界大小（占用原图）
border-color: beige;         # 边界颜色：米黄
border-radius: 10px;         # 边界圆角大小
border: None   				 # 去边框：也可以none
border: transparent          # 边框设置为透明
```

模板：

1. 圆角文本框

```css
QLabel{
	border-style: outset;         
	border-wiqdth: 1px;
	border-color: rgb(177,177,177);          
	border-radius: 8px;
}
```

2. 按钮 (`QRadioButton`,`QPushButton`)，按键的可选状态为：`hover`,`pressed`(单击时),`focus`(获得`focus`时), disabled(无法触发时)

```css
QPushButton{
	color: rgb(255, 255, 255);
	background-color: rgb(25, 35, 45);
	border-style: outset;       
	border-width: 1px;	
	border-color: beige;  
	border-radius: 5px; 
}

QPushButton::enabled{ 
	background-color: white;
	color: black;
}

QPushButton:disabled { 
background-color: rgbd(155,155,155,50);
}

QPushButton:focus { background-color: rgbd(255,0,0,200) }

/*用于自锁按钮：按下时*/
QPushButton:checked{ 
	background-color: white;
	color: black;
}
  
QPushButton:pressed { 
background-color: rgbd(255,0,0,200);
border-width: 3px;	
}

/*鼠标在上面停留时*/
QPushButton:hover { background-color: rgbd(155,155,155,155) }
```

3. 标签栏(`TabWidget`)

- 商务雅黑

```css
/*标签栏的字体色和背景色*/
QTabBar{
	color: white;
	background: rgbd(100,100,100,100);
}

/*选中的标签栏的背景色*/
QTabBar::tab:selected{
	background-color: rgbd(155,155,155,155);
}
QTabBar::tab{
	background-color: rgbd(155,155,155,155);
}

/*配置面板边框*/
QTabWidget:pane{
	border: 2px solid rgb(192,192,192);
	border-radius: 6px;
	border-style: outset;
};
```

4. GroupBox

- box标题栏白字、背景透明、边界透明

```css
QGroupBox{
	color: white;
	background-color: transparent;
	border: transparent;
}
```





### 功能

#### 定时器

```python
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

# 继承Q5 designer生成的界面类
class My_MainWindow(Ui_MainWindow):
    def __init__(self):
        
        # 以1s触发信号，激活update_log这个slot函数（每1s都会触发一次）
        self.timer = QTimer()
        self.timer.timeout.connect(self.slot_func)        
        self.timer.start(1000)  # ms

    def slot_func(self):
       pass

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    demo = My_MainWindow()
    demo.show()
    sys.exit(app.exec_())
```



#### 事件

- 常用的事件有鼠标事件`QMouseEvent`, 键盘事件`QKeyEvent`, 绘制事件`QPaintEvent`, 窗口尺寸改变`QResizeEvent`, 滚动事件`QScrollEvent`, 控件显示`QshowEvent`, 控件隐藏事件`QHideEvent`, 定时器事件`QTimerEvent`
- 内置的事件可以进行重写，例如：重写窗口关闭（按×）事件

```python
def closeEvent(self, event):
    result = QtWidgets.QMessageBox.question(self,
                                            "Confirm Exit...",
                                            "Are you sure you want to exit ?",
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    event.ignore()  			# 忽略事件：不关闭窗口
    if result == QtWidgets.QMessageBox.Yes:
        pass
        event.accept()          # 接受事件：关闭窗口
```

- 利用函数的触发

```python
self.close()					   # 触发窗口关闭事件
self.showMinimized()	           # 触发主窗口缩放
self.update()                      # 触发QPaintEvent
```

- 重写`paintEvent`（绘制方法必须放在QPainter对象的begin()和end()之间）

```python
from PyQt5.QtGui import QPainter, QFont

def paintEvent(self, evt):    
    painter = QPainter()
    painter.begin(self)
    
    # 启用反锯齿
    painter.setRenderHint(QPainter.Antialiasing)
    pass    
    # 设置字体样式
    font = QFont('Microsoft YaHei')
    font.setPixelSize(14)
    # 等价于: font=QFont('Microsoft YaHei',14)
    painter.setFont(font)
    # 设置颜色
    painter.setPen(QColor(168,34,3))    
    painter.end()
```

#### slot 函数（回调/触发时）

```c++
// 设置信号与槽的连接
connect(output_maxNumGoal_button_, SIGNAL(clicked()), this,
        SLOT(updateMaxNumGoal()));
connect(output_maxNumGoal_button_, SIGNAL(clicked()), this,
        SLOT(updatePoseTable()));
connect(output_reset_button_, SIGNAL(clicked()), this, SLOT(initPoseTable()));
connect(output_cancel_button_, SIGNAL(clicked()), this, SLOT(cancelNavi()));
connect(output_startNavi_button_, SIGNAL(clicked()), this, SLOT(startNavi()));
connect(cycle_checkbox_, SIGNAL(clicked(bool)), this, SLOT(checkCycle()));
connect(output_timer, SIGNAL(timeout()), this, SLOT(startSpin()));
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





#### action

- 保存、剪切，粘贴都可称为动作（action）；action类是用户动作的抽象类，我们**不关心用户是怎么触发动作**的，是快捷键打开还是通过工具栏触发，只关心这个action要实现什么

```python
"""
action_object.setIcon()			 	# 设置图标
action_object.setToolTip()	   		# 设置提示字体
action_object.setStatusTip()      	# 设置状态提示
action_object.setShortcut() 	   	# 设置快捷键
"""

# 配置action的依附对象
bar = self.menuBar()
file = bar.addMenu("File")   # 有&时，后方的字母会加下划线
file.addAction("New")

# 实例化一个action类
save = QAction("Save",self)
# 增设快捷键
save.setShortcut("Ctrl+S")	 # 快捷键触发
...setShortcut(Qt.Key_F11)   # 特殊按键(e.g.Key_Escape)
file.addAction(save)

rqt_action = QAction("rqt", self)
rqt_action.setShortcut("Ctrl+S")
file_menu.addAction(rqt_action)
rqt_action.triggered.connect(self.open_rqt)

# QAction的紧凑写法
self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++",
                enabled=False, triggered=self.zoomIn)

def open_rqt(self):  # 定义action触发的事件函数
    subprocess.call("rqt")
```

PS：

- 在实际的使用中不能单独只用`shortcut`，但还需要添加到菜单栏上





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



## 实战

### 分离界面与设计逻辑

- Qt的设计可分为界面逻辑和业务逻辑，在执行时要将他们放到不同的线程中，避免业务逻辑的处理影响界面的显示

### [窗口设置](https://www.cnblogs.com/wiessharling/p/3750461.html)

- 具体效果可参考[link](https://github.com/baoboa/pyqt5/blob/master/examples/widgets/windowflags.py)

```
Qt全屏显示函数       showFullScreen()
Qt最大化显示函数     showMaximized()
Qt最小化显示函数     showMinimized()
Qt固定尺寸显示函数   resize(x,y)
Qt设置最大尺寸函数   setMaximumSize(w,h)
Qt设置最小尺寸函数    setMinimumSize(w,h)

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



#### 显示gif图

```
# 使用label显示gif
self.movie = QMovie("spongebob.gif")
self.<label>.setMovie(self.movie)
self.movie.start()
```



### 显示图片

```
def show_image(self, img_msg):
    img_cv2 = self.bridge.imgmsg_to_cv2(img_msg)
    # need change the format
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    # rescale the size the same of qt label
    qim = cv2.resize(img_cv2, (991, 671))
    qim_h, qim_w, qimd = qim.shape
    qim = QImage(qim.data, qim_w, qim_h, qim_w * qimd, QImage.Format_RGB888)
    # self.<标签名>.显示图片；将图片显示在label中
    self.video_display.setPixmap(QPixmap.fromImage(qim))
```



### 设置主窗口icon

```
self.setWindowIcon(QIcon("image/spongebob.png")))
```



### 自定义弹窗

```python
class VideoWindow(QDialog):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("mindvision")
        self.resize(900, 300)
        self.setMinimumSize(400, 200)
        layout = QVBoxLayout()
        self.video = QLabel(None)
        layout.addWidget(self.video)
        self.setLayout(layout)
        self.bridge = CvBridge()

    @pyqtSlot()
    def toggle_vis(self):
        """窗口显示/关闭"""
        if self.isVisible():
            self.hide()
        else:
            self.show()

    def broadcast_video(self, img_msg):
        img_cv2 = self.bridge.imgmsg_to_cv2(img_msg)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        # 缩放的百分比
        scale_percent = 60
        width = int(img_cv2.shape[1] * scale_percent / 100)
        height = int(img_cv2.shape[0] * scale_percent / 100)
        dim = (width, height)

        qim = cv2.resize(img_cv2, dim, interpolation=cv2.INTER_AREA)
        qim_h, qim_w, qimd = qim.shape
        qim = QImage(qim.data, qim_w, qim_h, qim_w * qimd, QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(qim))

# 需要指明父类，以使其触发的事件能被主窗口处理
instance.video_window = VideoWindow(instance)
instance.video_window.show()
```

### 用vscode打开指定目录的文件

1. [有关文件和文件夹的选取](https://blog.csdn.net/qq_21240643/article/details/100927321)
2. python命令行的调用



- 模块

```python
from PyQt5.QtWidgets import QFileDialog
```

- 示例代码1 

```python
# 同时获取多个文件
def import_file(self):
    self.ocr_file_names = QFileDialog.getOpenFileNames(self, '选择文件(可多选)', '',
                                                       'Files(*.pdf *.jpg *.gif *.png *jpeg)')[0]
    for file_id, file_name in enumerate(self.ocr_file_names):
        ...
```

- 示例代码2

```python
# 获取单个文件                                                           
def editor(self):
    self.textEdit = QTextEdit()
    self.setCentralWidget(self.textEdit)

def file_open(self):
    # need to make name an tupple otherwise i had an error and app crashed
    name, _ = QFileDialog.getOpenFileName(self, 'Open File', options=QFileDialog.DontUseNativeDialog)
    file = open(name, 'r')
    self.editor()

    with file:
        text = file.read()
        self.textEdit.setText(text)
```

