from PyQt5.QtCore import QIODevice, pyqtSignal
from PyQt5.QtSerialPort import QSerialPort


class MyPort:
    dataSig = pyqtSignal(str)
    portOpenSig = pyqtSignal(bool)
    def __init__(self):
        super().__init__()
        self.is_port_open = False
        self.port = QSerialPort()
        self.paritySet = {'even': QSerialPort.EvenParity, 'odd': QSerialPort.OddParity, 'none': QSerialPort.NoParity}

    def close_port(self):
        self.is_port_open = False
        self.port.close()

    def open_port(self, port_cfg, port_name):
        self.port.setPortName(port_name)
        if not self.port.setBaudRate(int(port_cfg['baud_rate']), QSerialPort.AllDirections):
            return False
        if not self.port.setStopBits(int(port_cfg['stop_bit'])):
            return False
        if not self.port.setParity(self.paritySet[port_cfg['parity']]):
            return False
        if not self.port.open(QIODevice.ReadWrite):
            return False
        return True

    def read_port(self):
        if self.is_port_open:  # 如果串口打开
            try:
                if self.port.canReadLine():
                    data = str(self.port.readLine())
                    dataRec = data[2:]
                    self.dataSig.emit(dataRec)
            except:
                pass

    def write_port(self, data):
        if data:
            self.port.write(data)


if __name__ == '__main__':

    port_cfg = {'port_index': selectPortIndex,
                'baud_rate': baudrate,
                'stop_bit': stopbits,
                'parity': parity}

"gpsReceiveConfig": {
    "gpsPort": "/dev/ttyUSB0",
    "sendPort": "/dev/ttyUSB1",
    "automaticPort": false
},