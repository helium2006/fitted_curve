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
from PySide6.QtGui import QIcon

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
        self.label_8.setStyleSheet(u"QLabel{\n"
"	color:rgb(26, 217, 255);\n"
"}")
        font = QFont()
        font.setPointSize(36)
        self.label_8.setFont(font)
        self.pushButton_7 = QPushButton(self.centralwidget)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setGeometry(QRect(50, 130, 190, 150))
        font1 = QFont()
        font1.setPointSize(24)
        self.pushButton_7.setFont(font1)
        self.pushButton_7.setStyleSheet(u"color:rgb(10,150,255);")
        '''self.pushButton_7.setStyleSheet(u"QPushButton{\n"
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
"")'''
        self.pushButton_8 = QPushButton(self.centralwidget)
        self.pushButton_8.setObjectName(u"pushButton_8")
        self.pushButton_8.setGeometry(QRect(340, 130, 210, 150))
        font2 = QFont()
        font2.setPointSize(18)
        self.pushButton_8.setFont(font2)
        self.pushButton_8.setStyleSheet(u"color:rgb(10,150,255);")
        '''self.pushButton_8.setStyleSheet(u"QPushButton{\n"
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
"}")'''
        self.pushButton_9 = QPushButton(self.centralwidget)
        self.pushButton_9.setObjectName(u"pushButton_9")
        self.pushButton_9.setGeometry(QRect(130, 310, 321, 51))
        font3 = QFont()
        font3.setPointSize(14)
        self.pushButton_9.setFont(font3)
        self.pushButton_9.setStyleSheet(u"color:rgb(225, 0, 0);")
        '''self.pushButton_9.setStyleSheet(u"QPushButton{\n"
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
"}")'''
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
        self.pushButton_7.setText(QCoreApplication.translate("MainWindow", u"实验数据\n分析模式", None))
        self.pushButton_8.setText(QCoreApplication.translate("MainWindow", u"数学\n解题\n模式", None))
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
        
        # 设置窗口图标
        icon_path = r"./resources/icon.ico"
        self.setWindowIcon(QIcon(icon_path))
        
        # 保存背景图片路径
        self.background_image_path = r"./resources/background.png"
        
        # 设置背景图片
        self.set_background_image()
        
        # 确保所有控件在背景图片之上显示
        self.raise_controls()
    
    def set_background_image(self):
        """设置窗口背景图片"""
        try:
            # 加载背景图片
            background_pixmap = QPixmap(self.background_image_path)
            
            # 创建调色板
            palette = QPalette()
            
            # 设置背景图片，并使其拉伸以填充整个窗口
            palette.setBrush(QPalette.Window, QBrush(background_pixmap.scaled(
                self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))
            
            # 应用调色板到窗口
            self.setPalette(palette)
            
            # 确保窗口自动填充背景
            self.setAttribute(Qt.WA_StyledBackground, True)
        except Exception as e:
            print(f"设置背景图片失败: {str(e)}")
    
    def resizeEvent(self, event):
        """窗口大小变化时重新设置背景图片"""
        # 调用父类的resizeEvent
        super().resizeEvent(event)
        
        # 重新设置背景图片以适应新的窗口大小
        self.set_background_image()
    
    def raise_controls(self):
        """确保所有控件在背景图片之上显示"""
        # 将所有控件提升到前面
        self.ui.label_8.raise_()
        self.ui.pushButton_7.raise_()
        self.ui.pushButton_8.raise_()
        self.ui.pushButton_9.raise_()
    
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

