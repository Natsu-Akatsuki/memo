import sys

from PyQt5 import QtWidgets

from get_pixel_coordinate_ui import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.value = 0
        # self.setMouseTracking(True)

    def mousePressEvent(self, event):
        # mouseMoveEvent, mousePressEvent
        x = event.pos().x()
        y = event.pos().y()
        self.statusBar()
        self.statusBar().showMessage("x=" + str(x) + " " + "y=" + str(y))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
