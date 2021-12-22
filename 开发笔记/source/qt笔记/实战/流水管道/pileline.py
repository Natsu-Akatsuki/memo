from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from PyQt5.QtGui import QBrush, QPen, QPainter, QPolygon, QPolygonF
from PyQt5.QtCore import QPoint, QPointF, QTimer, Qt


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "PyQt5 Drawing Polygon"
        self.InitWindow()

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.waterflow)
        self.timer.start(100)  # ms

        self.waterflow_offset = 0

    def waterflow(self):
        self.waterflow_offset += 0.1
        self.update()
    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.show()
        self.resize(1000, 1000)

    def paintEvent(self, event):
        points = [(500, 200), (600, 200), (600, 300)]

        # 管道
        painter = QPainter(self)
        pen = QPen(Qt.gray, 20, Qt.SolidLine)

        # MiterJoin, BevelJoin, RoundJoin
        pen.setJoinStyle(Qt.RoundJoin)
        # FlatCap, SquareCap, RoundCap
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)

        points = QPolygonF([QPointF(*point) for point in points])
        painter.drawPolyline(points)

        # 水流
        painter_fg = QPainter(self)
        # 虚线
        pen = QPen(Qt.yellow, 8, Qt.DashLine)

        # MiterJoin, BevelJoin, RoundJoin
        # pen.setJoinStyle(Qt.MiterJoin)
        # FlatCap, SquareCap, RoundCap
        # pen.setCapStyle(Qt.RoundCap)
        pen.setDashOffset(self.waterflow_offset)
        painter_fg.setPen(pen)

        painter_fg.drawPolyline(points)


App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
