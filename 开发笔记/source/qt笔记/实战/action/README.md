# ACTION

保存、剪切，粘贴都可称为动作（action）；action类是用户动作的抽象类，我们**不关心用户是怎么触发动作**的，是快捷键打开，是通过工具栏触发，还是通过按键触发，而只关心这个action要实现什么

## 绑定到工具栏

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

