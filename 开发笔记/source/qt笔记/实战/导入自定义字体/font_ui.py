# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'font.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.cal_img_label = QtWidgets.QLabel(self.centralwidget)
        self.cal_img_label.setEnabled(True)
        self.cal_img_label.setGeometry(QtCore.QRect(90, 70, 261, 131))
        self.cal_img_label.setText("")
        self.cal_img_label.setObjectName("cal_img_label")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(180, 150, 67, 22))
        self.label.setStyleSheet("font: 11pt \"Noto Sans Adlam Unjoined\";")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(180, 180, 67, 22))
        self.label_2.setStyleSheet("font: 11pt \"Noto Sans Adlam Unjoined\";")
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 34))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p>思源黑体</p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p>宋体</p></body></html>"))
