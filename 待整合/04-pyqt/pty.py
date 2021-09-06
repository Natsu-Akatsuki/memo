import os
import select


def mkpty():
    master1, slave = os.openpty()
    slave_name1 = os.ttyname(slave)
    master2, slave = os.openpty()
    slave_name2 = os.ttyname(slave)
    print(f"slave device name is {slave_name1},{slave_name2}")
    return master1, master2


if __name__ == '__main__':
    master1, master2 = mkpty()
    while True:
        rl, wl, el = select.select([master1, master2], [], [], 1)
        for master in rl:
            data = os.read(master, 128)
            print(f"read {len(data)} data")
            if master == master1:
                os.write(master2, data)
            else:
                os.write(master1, data)
