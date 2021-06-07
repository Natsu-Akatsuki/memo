import threading
from PyQt5.QtCore import (QCoreApplication, QObject, QRunnable, QThread,
                          QThreadPool, pyqtSignal, pyqtSlot)
import sys

"""
result: 
-main- the thread name is: MainThread and the id is: 140393919043392
-f1- the thread name is: MainThread and the id is: 140393919043392
-f1- the thread name is: Dummy-1 and the id is: 140393602361088
-f2- the thread name is: Dummy-1 and the id is: 140393602361088
"""


class Worker(QObject):
    # 自定义信号
    call_f1 = pyqtSignal()
    call_f2 = pyqtSignal()

    def __init__(self):
        super(Worker, self).__init__()

    @pyqtSlot()
    def f1(self):
        print("-f1- the thread name is:", threading.current_thread().name, "and the id is:",
              threading.current_thread().ident)

    @pyqtSlot()
    def f2(self):
        print("-f2- the thread name is:", threading.current_thread().name, "and the id is:",
              threading.current_thread().ident)


if __name__ == '__main__':
    app = QCoreApplication([])
    print("-main- the thread name is:", threading.current_thread().name, "and the id is:",
          threading.current_thread().ident)
    # 启动一个线程
    my_thread = QThread()
    my_thread.start()

    my_worker = Worker()
    my_worker.call_f1.connect(my_worker.f1)
    my_worker.call_f1.emit()

    # 将QObject对象挪到该线程
    my_worker.moveToThread(my_thread)
    my_worker.call_f2.connect(my_worker.f2)
    my_worker.call_f1.emit()
    my_worker.call_f2.emit()
    sys.exit(app.exec_())
