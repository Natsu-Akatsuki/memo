import sys

from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtWidgets import QAction, QApplication, QLabel, QWidget

from feeder_subwindow_ui import Ui_Form
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QPoint, Qt


class Win(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setWindowFlags(Qt.FramelessWindowHint)

        # 隐藏图标
        # pixmap = QPixmap(32, 32)
        # pixmap.fill(Qt.transparent)
        # self.setWindowIcon(QIcon(pixmap))

        self._startPos = None
        self._endPos = None
        self._isTracking = False

        self.exit_btn.clicked.connect(self.close)
        temp_action = QAction("Exit", parent=self, shortcut=Qt.Key_Escape, enabled=True, triggered=self.close)
        self.exit_btn.addAction(temp_action)

    def mouseMoveEvent(self, e: QMouseEvent):
        self._endPos = e.pos() - self._startPos
        self.move(self.pos() + self._endPos)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._isTracking = True
            self._startPos = QPoint(e.x(), e.y())

    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._isTracking = False
            self._startPos = None
            self._endPos = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    subwindow = Win()
    subwindow.show()
    sys.exit(app.exec_())
