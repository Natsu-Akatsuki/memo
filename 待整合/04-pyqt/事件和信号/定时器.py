import sys, math
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Tim(QWidget):
    def __init__(self, parent=None):
        super(Tim, self).__init__(parent)
        self.resize(300, 200)

        self.timer = QTimer()
        self.timer.timeout.connect(self.slot_func)
        self.timer.start(100)  # ms

    def slot_func(self):
        print("hello world!")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = Tim()
    demo.show()
    sys.exit(app.exec_())
