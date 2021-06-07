import sys
from PyQt5.QtCore import Qt, QEventLoop
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout


class Demo(QWidget):
    def __init__(self):
        super(Demo, self).__init__()
        self.count = 0

        self.btn = QPushButton('计数', self)
        self.btn.clicked.connect(self.count_slot)
        self.label = QLabel('0', self)
        self.label.setAlignment(Qt.AlignCenter)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.label)
        v_layout.addWidget(self.btn)
        self.setLayout(v_layout)

    def count_slot(self):
        while True:
            self.count += 1
            self.label.setText(str(self.count))
            QApplication.processEvents()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = Demo()
    demo.show()
    sys.exit(app.exec_())