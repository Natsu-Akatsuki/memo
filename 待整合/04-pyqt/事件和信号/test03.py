"""
在Qthread中的事件无法被执行
"""

# import sys
# from PyQt5 import QtCore, QtWidgets
# import time
#
#
# def timer_func():
#     print("Timer works")
#
#
# class Thread(QtCore.QThread):
#     def __init__(self):
#         QtCore.QThread.__init__(self)
#
#     def run(self):
#         print("Thread works")
#         timer = QtCore.QTimer()
#         timer.timeout.connect(timer_func)
#         timer.start(1000)
#         print(timer.remainingTime())
#         print(timer.isActive())
#
# app = QtWidgets.QApplication(sys.argv)
# thread_instance = Thread()
# thread_instance.start()
# # 此时线程已经执行完毕
# thread_instance.exec_()
# sys.exit(app.exec_())

import sys
from PyQt5 import QtCore, QtWidgets
import threading


def timer_func():
    print("Timer works")


class Thread(QtCore.QObject):
    def __init__(self):
        # QtCore.QThread.__init__(self)
        QtCore.QObject.__init__(self)

    def run(self):
        print("the thread name is:", QtCore.QThread.currentThread().objectName())
        print("Thread works")
        timer = QtCore.QTimer()
        timer.timeout.connect(timer_func)
        timer.start(1000)
        print(timer.remainingTime())
        print(timer.isActive())
        self.exec_()


print("the thread name is:", threading.current_thread().name, "and the id is:",
      threading.current_thread().ident)
app = QtWidgets.QApplication(sys.argv)
thread_instance = Thread()
thread_instance.start()
sys.exit(app.exec_())
