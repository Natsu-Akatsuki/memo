import threading
from PyQt5.QtCore import (QCoreApplication, QObject, QRunnable, QThread,
                          QThreadPool, pyqtSignal, pyqtSlot)
import sys


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

    # 启动一个线程
    my_thread = QThread()
    my_thread.start()
    # 将QObject对象挪到该线程
    my_worker = Worker()
    my_worker.moveToThread(my_thread)
    my_worker.call_f2.connect(my_worker.f2)
    # 发送信号到该对象进行处理
    my_worker.call_f1.emit()
    my_worker.call_f2.emit()
    sys.exit(app.exec_())