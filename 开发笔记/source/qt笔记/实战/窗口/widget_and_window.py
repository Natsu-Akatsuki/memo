import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QApplication


class Window(QWidget):

    def __init__(self):
        super().__init__()

        # This is the widget that we will experiment with: note that we don't
        # set a parenet, and don't add it to the layout.
        self.mywidget = QLabel("Hello")
        self.mywidget.show()  # We need to show it (or focus it) to make it visible.

        add_parent = QPushButton("Add parent")
        add_parent.clicked.connect(self.add_parent)

        remove_parent = QPushButton("Remove parent")
        remove_parent.clicked.connect(self.remove_parent)

        add_layout = QPushButton("Add layout")
        add_layout.clicked.connect(self.add_layout)

        remove_layout = QPushButton("Remove layout")
        remove_layout.clicked.connect(self.remove_layout)

        self.vlayout = QVBoxLayout()

        self.vlayout.addWidget(add_parent)
        self.vlayout.addWidget(remove_parent)
        self.vlayout.addWidget(add_layout)
        self.vlayout.addWidget(remove_layout)

        self.setLayout(self.vlayout)

    def add_parent(self):
        self.mywidget.setParent(self)
        self.mywidget.move(0, 0)
        self.mywidget.show()
        print("Added parent, parent is:", self.mywidget.parent())

    def remove_parent(self):
        self.mywidget.setParent(None)
        self.mywidget.show()
        print("Removed parent, parent is:", self.mywidget.parent())

    def add_layout(self):
        self.vlayout.addWidget(self.mywidget)
        print("Added layout, parent is:", self.mywidget.parent())

    def remove_layout(self):
        self.vlayout.removeWidget(self.mywidget)
        print("Removed layout, parent is:", self.mywidget.parent())


app = QApplication(sys.argv)

w = Window()
w.show()

app.exec()
