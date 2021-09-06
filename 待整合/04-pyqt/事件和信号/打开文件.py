# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'open_file.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_open_file(object):
    def setupUi(self, open_file):
        open_file.setObjectName("open_file")
        open_file.resize(910, 545)
        self.verticalLayout = QtWidgets.QVBoxLayout(open_file)
        self.verticalLayout.setObjectName("verticalLayout")
        self.file_tree = QtWidgets.QTreeView(open_file)
        self.file_tree.setObjectName("file_tree")
        self.verticalLayout.addWidget(self.file_tree)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.file_path = QtWidgets.QLabel(open_file)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.file_path.setFont(font)
        self.file_path.setObjectName("file_path")
        self.horizontalLayout.addWidget(self.file_path)
        self.file_path_data = QtWidgets.QLabel(open_file)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.file_path_data.setFont(font)
        self.file_path_data.setText("")
        self.file_path_data.setObjectName("file_path_data")
        self.horizontalLayout.addWidget(self.file_path_data)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(open_file)
        QtCore.QMetaObject.connectSlotsByName(open_file)

    def retranslateUi(self, open_file):
        _translate = QtCore.QCoreApplication.translate
        open_file.setWindowTitle(_translate("open_file", "Form"))
        self.file_path.setText(_translate("open_file", "当前选中(双击打开)："))
