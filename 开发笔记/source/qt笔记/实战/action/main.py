import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QAction
from PyQt5.QtCore import Qt

from btn_action_ui import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        temp_action = QAction("Exit", parent=self, shortcut=Qt.Key_Escape, enabled=True, triggered=self.close)
        self.pushButton.addAction(temp_action)

    def closeEvent(self, event):
        result = QtWidgets.QMessageBox.question(self,
                                                "退出",
                                                "确认退出？",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if result == QtWidgets.QMessageBox.Yes:
            self.showMinimized()
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
