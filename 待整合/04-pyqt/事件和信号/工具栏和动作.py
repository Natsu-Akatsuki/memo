from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, qApp, QMenu
from PyQt5.QtGui import QIcon, QStatusTipEvent
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.status_bar = self.statusBar()  # 创建状态栏
        self.initUI()

    def toggle_menu(self, state):  # 自定义事件函数
        if state:
            self.status_bar.show()
        else:
            self.status_bar.hide()

    def build_menu_bar(self):
        menu_bar = self.menuBar()  # 构建菜单栏
        file_menu = menu_bar.addMenu("&File")  # 增设菜单子栏(File)，有&时，显示会有下划线
        edit_menu = menu_bar.addMenu("&Edit")  # 增设菜单子栏(Edit)
        view_menu = menu_bar.addMenu("&View")  # 增设菜单子栏(View)

        import_menu = QMenu("Import", self)
        import_act = QAction("Import Email", self)
        import_act.setShortcut("Ctrl+S")
        import_menu.addAction(import_act)
        file_menu.addMenu(import_menu)

        new_act = QAction("New", self)
        file_menu.addAction(new_act)

        # view_menu（配置Action）
        view_status_action = QAction("是否显示状态栏", parent=self, checkable=True)
        view_status_action.setToolTip("view status bar")
        view_status_action.setChecked(False)
        view_status_action.setShortcut("Ctrl+A")
        view_status_action.triggered.connect(self.toggle_menu)

        view_menu.addAction(view_status_action)

    def initUI(self):

        self.status_bar.showMessage("ready!")  # 显示消息
        self.build_menu_bar()

        # 设置主窗口的位置和大小
        self.setGeometry(300, 300, 500, 600)
        self.setWindowTitle("主窗口的菜单栏和工具栏")
        self.show()

    # 此覆盖父类函数: 覆盖方法； 为了克服 将鼠标放置于菜单栏上 状态栏就消失的问题；
    def event(self, QEvent):
        if QEvent.type() == QEvent.StatusTip:
            if QEvent.tip() == "":
                QEvent = QStatusTipEvent("ready!")  # 此处为要始终显示的内容
        return super().event(QEvent)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
