import sys

from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import img_qrc
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QAction
from PyQt5.QtCore import Qt
from img_ui import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.label.setPixmap(QtGui.QPixmap(":/temp.jpg"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
