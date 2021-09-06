# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qthread.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(526, 373)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.subreddits_input_layout = QtWidgets.QHBoxLayout()
        self.subreddits_input_layout.setObjectName("subreddits_input_layout")
        self.label_subreddits = QtWidgets.QLabel(self.centralwidget)
        self.label_subreddits.setObjectName("label_subreddits")
        self.subreddits_input_layout.addWidget(self.label_subreddits)
        self.edit_subreddits = QtWidgets.QLineEdit(self.centralwidget)
        self.edit_subreddits.setObjectName("edit_subreddits")
        self.subreddits_input_layout.addWidget(self.edit_subreddits)
        self.verticalLayout.addLayout(self.subreddits_input_layout)
        self.label_submissions_list = QtWidgets.QLabel(self.centralwidget)
        self.label_submissions_list.setObjectName("label_submissions_list")
        self.verticalLayout.addWidget(self.label_submissions_list)
        self.list_submissions = QtWidgets.QListWidget(self.centralwidget)
        self.list_submissions.setBatchSize(1)
        self.list_submissions.setObjectName("list_submissions")
        self.verticalLayout.addWidget(self.list_submissions)
        self.progress_bar = QtWidgets.QProgressBar(self.centralwidget)
        self.progress_bar.setProperty("value", 0)
        self.progress_bar.setObjectName("progress_bar")
        self.verticalLayout.addWidget(self.progress_bar)
        self.buttons_layout = QtWidgets.QHBoxLayout()
        self.buttons_layout.setObjectName("buttons_layout")
        self.btn_stop = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setObjectName("btn_stop")
        self.buttons_layout.addWidget(self.btn_stop)
        self.btn_start = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start.setObjectName("btn_start")
        self.buttons_layout.addWidget(self.btn_start)
        self.verticalLayout.addLayout(self.buttons_layout)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Threading Tutorial - nikolak.com"))
        self.label_subreddits.setText(_translate("MainWindow", "Subreddits:"))
        self.edit_subreddits.setPlaceholderText(_translate("MainWindow", "python,programming,linux,etc (comma separated)"))
        self.label_submissions_list.setText(_translate("MainWindow", "Submissions:"))
        self.btn_stop.setText(_translate("MainWindow", "Stop"))
        self.btn_start.setText(_translate("MainWindow", "Start"))

