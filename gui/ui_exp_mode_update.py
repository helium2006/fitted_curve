# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file '曲线拟合-实验数据模式QYeyzt.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QCheckBox, QDoubleSpinBox, QFrame,
    QGraphicsView, QLabel, QLineEdit, QMainWindow,
    QPushButton, QRadioButton, QSizePolicy, QSpinBox,
    QStatusBar, QTextBrowser, QTextEdit, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1000, 800)
        MainWindow.setStyleSheet(u"QMianWindow{\n"
"	background-color: rgb(0, 255, 255);\n"
"}")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.frame_5 = QFrame(self.centralwidget)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setGeometry(QRect(5, 5, 260, 300))
        self.frame_5.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Shadow.Raised)
        self.textEdit = QTextEdit(self.frame_5)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(5, 35, 120, 260))
        font4=QFont()
        font4.setPointSize(10)
        self.textEdit.setFont(font4)
        self.textEdit.setStyleSheet(u"QTextEdit{\n"
"background : rgb(85, 255, 127)\n"
"}")
        self.label_8 = QLabel(self.frame_5)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(50, 10, 53, 20))
        font = QFont()
        font.setPointSize(11)
        self.label_8.setFont(font)
        self.textEdit_2 = QTextEdit(self.frame_5)
        self.textEdit_2.setObjectName(u"textEdit_2")
        self.textEdit_2.setGeometry(QRect(135, 35, 120, 260))
        self.textEdit_2.setFont(font4)
        self.textEdit_2.setStyleSheet(u"QTextEdit{\n"
"background : rgb(85, 255, 127)\n"
"}")
        self.label_9 = QLabel(self.frame_5)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(175, 10, 53, 20))
        self.label_9.setFont(font)
        self.pushButton_7 = QPushButton(self.centralwidget)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setGeometry(QRect(25, 310, 85, 30))
        font1 = QFont()
        font1.setPointSize(10)
        self.pushButton_7.setFont(font1)
        self.pushButton_7.setStyleSheet(u"QPushButton{\n"
"	background-color: rgb(0, 170, 0);\n"
"    color: rgb(255, 255, 255)\n"
"}\n"
"QPushButton:hover{\n"
"	background-color: rgb(85, 255, 127);\n"
"	color: rgb(0, 0, 0)\n"
"}\n"
"QPushButton:pressed{\n"
"	background-color: rgb(85, 170, 0);\n"
"	color: rgb(0, 0, 0)\n"
"}")
        self.frame_6 = QFrame(self.centralwidget)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setGeometry(QRect(5, 345, 260, 130))
        self.frame_6.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Shadow.Raised)
        self.label_10 = QLabel(self.frame_6)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(5, 5, 95, 25))
        self.label_10.setFont(font)
        self.label_11 = QLabel(self.frame_6)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(10, 35, 70, 20))
        self.label_11.setFont(font1)
        self.comboBox = QComboBox(self.frame_6)
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(80, 35, 120, 23))
        # 修改为迭代过滤控件
        self.radioButton_13 = QRadioButton(self.frame_6)
        self.radioButton_13.setObjectName(u"radioButton_13")
        self.radioButton_13.setGeometry(QRect(10, 70, 120, 20))
        self.label_16 = QLabel(self.frame_6)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(130, 70, 70, 20))
        self.spinBox_iterations = QSpinBox(self.frame_6)
        self.spinBox_iterations.setObjectName(u"spinBox_iterations")
        self.spinBox_iterations.setGeometry(QRect(200, 70, 51, 23))
        self.spinBox_iterations.setMinimum(1)
        self.spinBox_iterations.setMaximum(10)
        self.spinBox_iterations.setValue(3)
        self.label_12 = QLabel(self.frame_6)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(20, 100, 90, 20))
        self.doubleSpinBox = QDoubleSpinBox(self.frame_6)
        self.doubleSpinBox.setObjectName(u"doubleSpinBox")
        self.doubleSpinBox.setGeometry(QRect(110, 100, 87, 23))
        self.doubleSpinBox.setMinimum(0.01)
        self.doubleSpinBox.setMaximum(10.0)
        self.doubleSpinBox.setValue(1.00)
        self.doubleSpinBox.setSingleStep(0.01)
        self.frame_7 = QFrame(self.centralwidget)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setGeometry(QRect(5, 480, 260, 100))
        self.frame_7.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Shadow.Raised)
        self.label_13 = QLabel(self.frame_7)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(5, 5, 110, 25))
        self.label_13.setFont(font)
        self.label_14 = QLabel(self.frame_7)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(20, 40, 53, 20))
        self.label_14.setFont(font1)
        self.label_15 = QLabel(self.frame_7)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(20, 70, 53, 20))
        self.label_15.setFont(font1)
        self.lineEdit_4 = QLineEdit(self.frame_7)
        self.lineEdit_4.setObjectName(u"lineEdit_4")
        self.lineEdit_4.setGeometry(QRect(60, 40, 113, 21))
        self.lineEdit_4.setStyleSheet(u"QLineEdit{\n"
"background : rgb(85, 255, 127)\n"
"}")
        self.lineEdit_5 = QLineEdit(self.frame_7)
        self.lineEdit_5.setObjectName(u"lineEdit_5")
        self.lineEdit_5.setGeometry(QRect(60, 70, 113, 21))
        self.lineEdit_5.setStyleSheet(u"QLineEdit{\n"
"background-color: rgb(85, 255, 127);\n"
"}")
        self.graphicsView = QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName(u"graphicsView")
        self.graphicsView.setGeometry(QRect(270, 5, 720, 580))
        self.textBrowser = QTextBrowser(self.centralwidget)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setGeometry(QRect(270, 590, 720, 192))
        font = QFont()
        font.setPointSize(12)  # 增大字体大小
        self.textBrowser.setFont(font)
        self.pushButton_8 = QPushButton(self.centralwidget)
        self.pushButton_8.setObjectName(u"pushButton_8")
        self.pushButton_8.setGeometry(QRect(60, 590, 140, 50))
        font2 = QFont()
        font2.setPointSize(14)
        self.pushButton_8.setFont(font2)
        self.pushButton_8.setStyleSheet(u"QPushButton{\n"
"	background-color: rgb(255, 150, 150);\n"
"	color: rgb(0, 0, 0);\n"
"}\n"
"QPushButton:hover{\n"
"	background-color: rgb(255, 0, 127);\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"QPushButton:pressed{\n"
"	background-color: rgb(200, 0, 0);\n"
"	color: rgb(255, 255, 255);\n"
"}")
        self.radioButton_14 = QCheckBox(self.centralwidget)
        self.radioButton_14.setObjectName(u"radioButton_14")
        self.radioButton_14.setGeometry(QRect(140, 660, 140, 30))
        self.pushButton_9 = QPushButton(self.centralwidget)
        self.pushButton_9.setObjectName(u"pushButton_9")
        self.pushButton_9.setGeometry(QRect(5, 650, 130, 45))
        self.pushButton_9.setFont(font)
        self.pushButton_9.setStyleSheet(u"QPushButton{\n"
"	background-color: rgb(255, 150, 150);\n"
"	color: rgb(0, 0, 0);\n"
"}\n"
"QPushButton:hover{\n"
"	background-color: rgb(255, 0, 127);\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"QPushButton:pressed{\n"
"	background-color: rgb(200, 0, 0);\n"
"	color: rgb(255, 255, 255);\n"
"}")
        self.pushButton_10 = QPushButton(self.centralwidget)
        self.pushButton_10.setObjectName(u"pushButton_10")
        self.pushButton_10.setGeometry(QRect(5, 710, 130, 45))
        self.pushButton_10.setFont(font)
        self.pushButton_10.setStyleSheet(u"QPushButton{\n"
"	background-color: rgb(255, 150, 150);\n"
"	color: rgb(0, 0, 0);\n"
"}\n"
"QPushButton:hover{\n"
"	background-color: rgb(255, 0, 127);\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"QPushButton:pressed{\n"
"	background-color: rgb(200, 0, 0);\n"
"	color: rgb(255, 255, 255);\n"
"}")
        self.pushButton_11 = QPushButton(self.centralwidget)
        self.pushButton_11.setObjectName(u"pushButton_11")
        self.pushButton_11.setGeometry(QRect(140, 710, 130, 45))
        self.pushButton_11.setFont(font)
        self.pushButton_11.setStyleSheet(u"QPushButton{\n"
"	background-color: rgb(255, 0, 0);\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"QPushButton:hover{\n"
"	background-color: rgb(255, 60, 150);\n"
"	color: rgb(0, 0, 0);\n"
"}\n"
"QPushButton:pressed{\n"
"	background-color: rgb(214, 50, 127);\n"
"	color: rgb(0, 0, 0);\n"
"}")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u66f2\u7ebf\u62df\u5408-\u5b9e\u9a8c\u6570\u636e\u6a21\u5f0f", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"X\u5750\u6807", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Y\u5750\u6807", None))
        self.pushButton_7.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u793a\u4f8b\u6570\u636e", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"\u914d\u7f6e\u62df\u5408\u4fe1\u606f", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"\u62df\u5408\u65b9\u5f0f\uff1a", None))
        self.radioButton_13.setText(QCoreApplication.translate("MainWindow", u"\u8fed\u4ee3\u8fc7\u6ee4\u5f02\u5e38\u503c", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"\u8fed\u4ee3\u9608\u503c\uff1a", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"\u8fed\u4ee3\u6b65\u6570\uff1a", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"\u914d\u7f6e\u5750\u6807\u8f74\u4fe1\u606f", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"X\u8f74\uff1a", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Y\u8f74\uff1a", None))
        self.pushButton_8.setText(QCoreApplication.translate("MainWindow", u"\u7ed8\u5236\u62df\u5408\u66f2\u7ebf", None))
        self.radioButton_14.setText(QCoreApplication.translate("MainWindow", u"\u540c\u65f6\u8f93\u51fa\u5408\u7406\u6027\u5206\u6790", None))
        self.pushButton_9.setText(QCoreApplication.translate("MainWindow", u"\u8f93\u51fa\u7edf\u8ba1\u5b66\u6570\u636e", None))
        self.pushButton_10.setText(QCoreApplication.translate("MainWindow", u"\u5bfc\u51fa\u4e3a\u56fe\u7247", None))
        self.pushButton_11.setText(QCoreApplication.translate("MainWindow", u"\u8fd4\u56de\u6a21\u5f0f\u9009\u62e9", None))
    # retranslateUi

