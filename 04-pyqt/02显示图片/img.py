import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

import cv2
from img_ui import Ui_MainWindow


# 继承designer设计的界面类，进一步设计逻辑代码
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

    def show_img_on_label(self, img_np):
        """
        在一个label上显示图片
        """
        # 记录图片的形状信息
        img_h, img_w, img_d = img_np.shape
        # cv2.imshow("img", img_np)
        # 颜色通道的转换（qt使用的是RGB颜色空间）
        QIm = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        # 创建qt图像类
        # todo:img_w * img_d?
        QIm = QImage(QIm.data, img_w, img_h, img_w * img_d, QImage.Format_RGB888)
        # self.<标签名>.显示图片；将图片显示在label中
        self.cal_img_label.setPixmap(QPixmap.fromImage(QIm))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    image_path = "test01.jpg"
    image = cv2.imread(image_path)
    main_window.show_img_on_label(image)
    sys.exit(app.exec_())
