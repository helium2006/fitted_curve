# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file '模式选择rfMYwB.ui'
##
## Created by: Qt User Interface Compiler version 6.10.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(600, 400)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.label_8 = QLabel(self.centralwidget)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(150, 5, 291, 71))
        font = QFont()
        font.setPointSize(36)
        self.label_8.setFont(font)
        self.pushButton_7 = QPushButton(self.centralwidget)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setGeometry(QRect(50, 130, 190, 150))
        font1 = QFont()
        font1.setPointSize(24)
        self.pushButton_7.setFont(font1)
        self.pushButton_7.setStyleSheet(u"QPushButton{\n"
"	background-color:rgb(0, 255, 255);\n"
"	color:rgb(0, 0, 0);\n"
"}\n"
"QPushButton:hover{\n"
"	background-color:rgb(170, 255, 255);\n"
"	color:rgb(0, 0, 0)\n"
"}\n"
"QPushButton:pressed{\n"
"	background-color: rgb(0, 234, 234);\n"
"	color:rgb(104, 104, 104);\n"
"}\n"
"\n"
"")
        self.pushButton_8 = QPushButton(self.centralwidget)
        self.pushButton_8.setObjectName(u"pushButton_8")
        self.pushButton_8.setGeometry(QRect(340, 130, 210, 150))
        font2 = QFont()
        font2.setPointSize(18)
        self.pushButton_8.setFont(font2)
        self.pushButton_8.setStyleSheet(u"QPushButton{\n"
"	background-color: rgb(0, 255, 0);\n"
"	color:rgb(0, 0, 0);\n"
"}\n"
"QPushButton:hover{\n"
"	background-color: rgb(0, 255, 127); \n"
"    color:rgb(0, 0, 0);\n"
"}\n"
"QPushButton:pressed{\n"
"	background-color:rgb(0, 220, 0);\n"
"	color:rgb(104, 104, 104);\n"
"}")
        self.pushButton_9 = QPushButton(self.centralwidget)
        self.pushButton_9.setObjectName(u"pushButton_9")
        self.pushButton_9.setGeometry(QRect(130, 310, 321, 51))
        font3 = QFont()
        font3.setPointSize(14)
        self.pushButton_9.setFont(font3)
        self.pushButton_9.setStyleSheet(u"QPushButton{\n"
"	background-color: rgb(255, 0, 0);\n"
"	color:rgb(255, 255, 255);\n"
"}\n"
"QPushButton:hover{\n"
"	background-color: rgb(255, 0, 127);\n"
"	color:rgb(255, 255, 255);\n"
"}\n"
"QPushButton:pressed{\n"
"	background-color: rgb(227, 0, 0);\n"
"	color:rgb(54,54,54);\n"
"}")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"曲线拟合程序 - 模式选择", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"曲线拟合程序", None))
        self.pushButton_7.setText(QCoreApplication.translate("MainWindow", u"实验数据模式", None))
        self.pushButton_8.setText(QCoreApplication.translate("MainWindow", u"数学回归模式", None))
        self.pushButton_9.setText(QCoreApplication.translate("MainWindow", u"退出程序", None))
    # retranslateUi


class ChooseModeWindow(QMainWindow):
    """模式选择窗口"""
    def __init__(self):
        super().__init__()
        
        # 设置UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 连接信号和槽
        self.ui.pushButton_7.clicked.connect(self.open_experiment_mode)
        self.ui.pushButton_8.clicked.connect(self.open_math_mode)
        self.ui.pushButton_9.clicked.connect(self.close)
        
        # 设置窗口属性
        self.setWindowTitle("曲线拟合程序 - 模式选择")
        self.setFixedSize(600, 400)
    
    def open_experiment_mode(self):
        """打开实验数据模式窗口"""
        # 延迟导入以避免循环导入
        from gui.experiment_mode_window import ExperimentModeWindow
        
        # 关闭当前窗口
        self.close()
        
        # 创建并显示实验模式窗口
        self.experiment_window = ExperimentModeWindow()
        self.experiment_window.show()
    
    def open_math_mode(self):
        """打开数学回归模式窗口"""
        # 延迟导入以避免循环导入
        from gui.ui_math_mode import MathModeWindow
        
        # 关闭当前窗口
        self.close()
        
        # 创建并显示数学模式窗口
        self.math_window = MathModeWindow()
        self.math_window.show()

