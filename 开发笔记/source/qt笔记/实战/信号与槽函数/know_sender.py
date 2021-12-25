import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication


class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.status_bar = self.statusBar()
        self.initUI()

    def initUI(self):
        self.btn1 = QPushButton("Button 1", self)
        self.btn1.move(30, 50)

        self.btn2 = QPushButton("Button 2", self)
        self.btn2.move(150, 50)

        self.btn1.clicked.connect(self.buttonClicked)
        self.btn2.clicked.connect(self.buttonClicked)

        self.setGeometry(300, 300, 290, 150)
        self.setWindowTitle('Event sender')
        self.show()

    @QtCore.pyqtSlot()
    def buttonClicked(self):
        sender = self.sender()
        self.status_bar.showMessage(sender.text() + ' was pressed')
        if self.sender() is self.btn1:
            print('btn1 was pressed')
        elif self.sender() is self.btn2:
            print('btn2 was pressed')

        # sender = self.sender()
        # self.statusBar().showMessage(sender.text() + ' was pressed')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
