import sys

from PyQt5 import QtGui, QtWidgets

from font_ui import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        QtGui.QFontDatabase.addApplicationFont("siyuan.ttf")
        self.label.setStyleSheet("font: 11pt \"Source Han Sans CN\";")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
