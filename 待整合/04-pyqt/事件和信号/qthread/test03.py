import threading
from PyQt5.QtCore import (QCoreApplication, QObject, QRunnable, QThread,
                          QThreadPool, pyqtSignal, pyqtSlot)
import sys


class GenericWorker(QObject):
    def __init__(self, *args, **kwargs):
        super(GenericWorker, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.start.connect(self.run)

    start = pyqtSignal(str)

    @pyqtSlot()
    def run(self):
        print("-run- the thread name is:", threading.current_thread().name, "and the id is:",
              threading.current_thread().ident)
        print(QThread.currentThread().objectName())


print("-main- the thread name is:", threading.current_thread().name, "and the id is:",
      threading.current_thread().ident)
app = QCoreApplication([])
my_thread = QThread()
my_thread.start()

# This causes my_worker.run() to eventually execute in my_thread:
my_worker = GenericWorker(...)
my_worker.moveToThread(my_thread)
my_worker.start.emit("hello")
my_worker.start.connect(my_worker.run)  # <---- Like this instead
sys.exit(app.exec_())
