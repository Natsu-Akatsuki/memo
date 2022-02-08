# Thread-based parallelism
# example from https://zhuanlan.zhihu.com/p/267650598
import threading
import time


def job1():
    global n, lock
    with lock:
        for i in range(10):
            n += 1
            print('job1', n)
            time.sleep(2)


def job2():
    global n, lock

    for i in range(10):
        n += 10
        print('job2', n)
        time.sleep(1)


if __name__ == "__main__":
    n = 0
    lock = threading.Lock()
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    t1.start()
    t2.start()
