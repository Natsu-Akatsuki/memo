import sys
import random
import sys

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QPropertyAnimation, QTimer, Qt
from PyQt5.QtWidgets import QGraphicsOpacityEffect, QLabel

from table_ui import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # for test
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(2000)

    def animate_refresh(self):
        """
        实现界面刷新（闪烁效果）
        :return:
        """
        self.mask = QLabel(self)
        self.mask.resize(self.status_table.size())
        self.mask.move(self.status_table.pos().x(), self.status_table.pos().y())
        self.mask.setStyleSheet("background-color: rgb(255, 255, 255);")

        eff = QGraphicsOpacityEffect(self)
        self.mask.setGraphicsEffect(eff)
        self.anim = QPropertyAnimation(eff, b"opacity")
        self.anim.setDuration(1500)
        self.anim.setStartValue(1)
        self.anim.setEndValue(0)

    def get_plc_date(self):
        """
        获取plc回传的设备状态信息
        :return:
        """

    def update_status(self):
        """
        更新ui数据的显示
        """
        color_dict = {"关闭": (255, 0, 0),
                      "开启": (0, 0, 255)
                      }
        # for test
        for row in range(0, 5):
            for col in range(1, 4):
                if row == 4 and col == 3:
                    continue
                if col == 2:
                    continue
                random_list = ["开启", "关闭"]
                status = random.choice(random_list)
                item = QtWidgets.QTableWidgetItem(status)
                item.setForeground(QtGui.QColor(*color_dict[status]))
                item.setTextAlignment(Qt.AlignCenter)
                self.status_table.setItem(row, col, item)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
