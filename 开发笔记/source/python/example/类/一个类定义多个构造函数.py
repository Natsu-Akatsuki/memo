import time


class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def today(cls):
        pass
        return cls
        # t = time.localtime()
        # return cls(t.tm_year, t.tm_mon, t.tm_mday)


a = Date(2021, 12, 21)
b = Date.today()
pass