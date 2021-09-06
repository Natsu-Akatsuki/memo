import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget

"""
涉及主窗口的各种参数配置
"""
if __name__ == '__main__':

    app = QApplication(sys.argv)

    main_window = QWidget()
    # Qt主窗口的尺寸大小
    main_window.resize(250, 150)
    # 主窗口的位置
    main_window.move(300, 300)
    # 等价于上两个步骤(resize和move)
    # main_window.setGeometry(300, 300, 250, 150)

    # 主窗口的标题
    main_window.setWindowTitle('学点编程吧出品')

    # 增设图标
    from PyQt5.QtGui import QIcon
    main_window.setWindowIcon(QIcon('Simple_PySide_Base/icons/24x24/cil-arrow-bottom.png'))

    # 去窗口边框
    # main_window.setWindowFlags(QtCore.Qt.FramelessWindowHint)
    # 约束窗口大小
    main_window.setMinimumSize(QtCore.QSize(1363, 895))
    main_window.setMaximumSize(QtCore.QSize(1363, 895))
    main_window.setFocusPolicy(QtCore.Qt.StrongFocus)
    # 透明度
    main_window.setWindowOpacity(0.8)

    main_window.show()
    sys.exit(app.exec_())
