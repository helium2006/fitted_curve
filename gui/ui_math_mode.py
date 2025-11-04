# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file '曲线拟合-数学模式fnfiEj.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFrame,
    QGraphicsView, QLabel, QMainWindow, QPushButton,
    QRadioButton, QSizePolicy, QStatusBar, QTextBrowser,
    QTextEdit, QWidget)
from utils.math_mode_utils import iterative_filter_outliers

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
        font1 = QFont()
        font1.setPointSize(10)
        self.textEdit.setFont(font1)
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
        font4 = QFont()
        font4.setPointSize(12)
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
        self.label_10.setGeometry(QRect(5, 5, 140, 25))
        self.label_10.setFont(font)
        self.label_11 = QLabel(self.frame_6)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(10, 35, 70, 20))
        self.label_11.setFont(font1)
        self.comboBox = QComboBox(self.frame_6)
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(80, 35, 120, 23))
        self.radioButton_13 = QRadioButton(self.frame_6)
        self.radioButton_13.setObjectName(u"radioButton_13")
        self.radioButton_13.setGeometry(QRect(10, 70, 120, 20))
        self.label_12 = QLabel(self.frame_6)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(20, 95, 90, 20))
        self.doubleSpinBox = QDoubleSpinBox(self.frame_6)
        self.doubleSpinBox.setObjectName(u"doubleSpinBox")
        self.doubleSpinBox.setGeometry(QRect(110, 95, 87, 23))
        self.doubleSpinBox.setMinimum(0.0)
        self.doubleSpinBox.setMaximum(10.0)
        self.doubleSpinBox.setValue(1.00)
        self.doubleSpinBox.setSingleStep(0.01)
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
        font3 = QFont()
        font3.setPointSize(9)
        self.pushButton_10.setFont(font3)
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
        self.pushButton_11.setGeometry(QRect(140, 654, 130, 101))
        self.pushButton_11.setFont(font2)
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
        self.frame_7 = QFrame(self.centralwidget)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setGeometry(QRect(5, 480, 260, 100))
        self.frame_7.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Shadow.Raised)
        self.label_13 = QLabel(self.frame_7)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(5, 5, 80, 20))
        self.textBrowser_2 = QTextBrowser(self.frame_7)
        self.textBrowser_2.setObjectName(u"textBrowser_2")
        self.textBrowser_2.setGeometry(QRect(5, 30, 240, 60))
        self.textBrowser_2.setFont(font4)
        self.textBrowser_2.setStyleSheet(u"background-color: rgb(235, 235, 235);")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u66f2\u7ebf\u62df\u5408-\u6570\u5b66\u6a21\u5f0f\uff08\u56de\u5f52\u8ba1\u7b97\uff09", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"X\u5750\u6807", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Y\u5750\u6807", None))
        self.pushButton_7.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u793a\u4f8b\u6570\u636e", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"\u914d\u7f6e\u56de\u5f52\u8ba1\u7b97\u65b9\u5f0f", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"\u56de\u5f52\u65b9\u5f0f\uff1a", None))
        self.radioButton_13.setText(QCoreApplication.translate("MainWindow", u"\u662f\u5426\u8fc7\u6ee4\u5f02\u5e38\u503c", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"\u5f02\u5e38\u503c\u9608\u503c\uff1a", None))
        self.pushButton_8.setText(QCoreApplication.translate("MainWindow", u"\u7ed8\u5236\u56de\u5f52\u66f2\u7ebf", None))
        self.pushButton_9.setText(QCoreApplication.translate("MainWindow", u"\u8f93\u51fa\u7edf\u8ba1\u5b66\u6570\u636e", None))
        self.pushButton_10.setText(QCoreApplication.translate("MainWindow", u"\u8f93\u51fa\u56de\u5f52\u65b9\u7a0b\u5bf9\u5e94\u7cfb\u6570", None))
        self.pushButton_11.setText(QCoreApplication.translate("MainWindow", u"\u8fd4\u56de\n"
"\u6a21\u5f0f\n"
"\u9009\u62e9", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"\u56de\u5f52\u65b9\u7a0b\u683c\u5f0f", None))
    # retranslateUi

from PySide6.QtWidgets import QMainWindow, QGraphicsScene, QMessageBox
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from PIL import Image

# 导入工具函数
from utils.math_mode_utils import math_mode_analysis, calculate_detailed_statistics, \
    format_math_results, assess_fitting_quality, generate_math_plot_data, \
    get_available_function_types, validate_function_type, get_function_type_from_display, get_function_display_name
from utils.fitting_functions import filter_outliers, get_function_info, fit_data

class MathModeWindow(QMainWindow):
    """数学回归模式窗口"""
    def __init__(self):
        super().__init__()
        
        # 设置UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 初始化数据
        self.x_data = []
        self.y_data = []
        self.current_results = None
        
        # 初始化绘图
        self.init_plot()
        
        # 设置回归方法下拉框
        function_types = get_available_function_types()
        self.ui.comboBox.addItems(function_types)
        
        # 连接信号和槽
        self.ui.pushButton_7.clicked.connect(self.show_example_data)
        self.ui.pushButton_8.clicked.connect(self.draw_regression_curve)
        self.ui.pushButton_9.clicked.connect(self.output_statistics)
        self.ui.pushButton_10.clicked.connect(self.output_regression_coefficients)
        self.ui.pushButton_11.clicked.connect(self.return_to_mode_select)
        self.ui.comboBox.currentIndexChanged.connect(self.on_function_type_changed)
        
        # 添加清除数据按钮
        self.clear_data_button = QPushButton("清除数据", self)
        self.clear_data_button.setGeometry(QRect(155, 310, 85, 30))
        fontx=QFont()
        fontx.setPointSize(10)
        self.clear_data_button.setFont(fontx)
        self.clear_data_button.setStyleSheet("""
            QPushButton{
                background-color: rgb(255, 165, 0);
                color: rgb(255, 255, 255)
            }
            QPushButton:hover{
                background-color: rgb(255, 215, 0);
                color: rgb(0, 0, 0)
            }
            QPushButton:pressed{
                background-color: rgb(255, 140, 0);
                color: rgb(0, 0, 0)
            }
        """)
        self.clear_data_button.clicked.connect(self.clear_data)
        
        # 设置窗口属性
        self.setWindowTitle("曲线拟合-数学模式（回归计算）")
        self.setFixedSize(1000, 800)
        
        # 设置窗口图标
        icon_path = r"./resources/icon.ico"
        self.setWindowIcon(QIcon(icon_path))
        
        # 显示初始回归方程格式
        self.on_function_type_changed(0)
    
    def init_plot(self):
        """初始化绘图区域"""
        self.scene = QGraphicsScene()
        self.ui.graphicsView.setScene(self.scene)
        
        # 创建matplotlib图形
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.scene.addWidget(self.canvas)
        
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei']  # 优先使用支持特殊字符的字体
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    def on_function_type_changed(self, index):
        """当回归方法改变时更新方程格式显示"""
        # 获取当前选择的函数类型
        function_type = self.ui.comboBox.currentText()
        
        try:
            # 从函数类型中提取方程格式（括号内的部分）
            if '(' in function_type and ')' in function_type:
                equation_format = function_type.split('(')[1].strip(')')
                self.ui.textBrowser_2.setText(f"方程格式: {equation_format}")
            else:
                # 尝试通过函数类型键名获取显示名称
                func_key = get_function_type_from_display(function_type)
                display_name = get_function_display_name(func_key)
                if '(' in display_name and ')' in display_name:
                    equation_format = display_name.split('(')[1].strip(')')
                    self.ui.textBrowser_2.setText(f"方程格式: {equation_format}")
                else:
                    # 如果没有找到，显示默认信息
                    self.ui.textBrowser_2.setText("方程格式: y = f(x)")
        except Exception as e:
            self.ui.textBrowser_2.setText("方程格式: y = f(x)")
    
    def show_example_data(self):
        """显示示例数据"""
        try:
            # 生成示例数据（线性关系）
            x = np.linspace(0, 10, 30)
            y = 2.5 * x + 1.3 + np.random.normal(0, 2, 30)
            
            # 填充到文本框
            self.ui.textEdit.setPlainText('\n'.join([f'{val:.4f}' for val in x]))
            self.ui.textEdit_2.setPlainText('\n'.join([f'{val:.4f}' for val in y]))
            
            # 更新内部数据
            self.x_data = x.tolist()
            self.y_data = y.tolist()
            
            # 显示提示
            QMessageBox.information(self, "成功", "示例数据已加载")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"生成示例数据时出错: {str(e)}")
    
    def parse_input_data(self):
        """解析输入数据"""
        try:
            # 从文本框获取数据
            x_text = self.ui.textEdit.toPlainText()
            y_text = self.ui.textEdit_2.toPlainText()
            
            # 分割并转换为浮点数
            x_data = [float(line.strip()) for line in x_text.split('\n') if line.strip()]
            y_data = [float(line.strip()) for line in y_text.split('\n') if line.strip()]
            
            # 检查数据长度是否一致
            if len(x_data) != len(y_data):
                QMessageBox.warning(self, "数据错误", "X和Y数据长度不一致")
                return None, None
            
            # 检查数据是否为空
            if not x_data or not y_data:
                QMessageBox.warning(self, "数据错误", "请输入数据")
                return None, None
            
            return x_data, y_data
        except Exception as e:
            QMessageBox.warning(self, "数据错误", f"数据格式错误: {str(e)}")
            return None, None
    
    def draw_regression_curve(self):
        """绘制回归曲线"""
        # 解析输入数据
        x_data, y_data = self.parse_input_data()
        if x_data is None or y_data is None:
            return
        
        # 更新内部数据
        self.x_data = x_data
        self.y_data = y_data
        
        # 获取回归参数
        function_type = self.ui.comboBox.currentText()
        filter_outlier = self.ui.radioButton_13.isChecked()
        outlier_threshold = self.ui.doubleSpinBox.value() if filter_outlier else 3.0
        
        # 准备数据用于分析
        x_array = np.array(x_data)
        y_array = np.array(y_data)
        
        # 验证函数类型是否适用于当前数据
        if not validate_function_type(function_type):
            QMessageBox.warning(self, "数据错误", f"选择的函数类型无效")
            return
        
        # 过滤异常值（如果需要）
        filtered_x, filtered_y = x_array, y_array
        filtered_indices = []
        iteration_history = []
        
        if filter_outlier:
            try:
                # 使用迭代过滤异常值，固定最大迭代次数为5
                filtered_x, filtered_y, filtered_indices, iteration_history = iterative_filter_outliers(
                    x_array, y_array, function_type, threshold=outlier_threshold, max_iterations=5
                )
                
                if len(filtered_x) < 3:
                    QMessageBox.warning(self, "警告", "过滤异常值后数据点过少，将使用原始数据")
                    filtered_x, filtered_y = x_array, y_array
                    filtered_indices = []
                    iteration_history = []
                else:
                    # 显示迭代过滤信息
                    self.statusBar().showMessage(
                        f"迭代过滤完成 - 过滤掉 {len(filtered_indices)} 个异常点, "
                        f"迭代次数: {len(iteration_history)}, "
                        f"剩余数据点: {len(filtered_x)}", 
                        5000
                    )
            except Exception as e:
                QMessageBox.warning(self, "警告", f"过滤异常值时出错: {str(e)}，将使用原始数据")
                filtered_x, filtered_y = x_array, y_array
        
        # 执行回归分析
        try:
            # 使用数学模式分析函数
            analysis_results = math_mode_analysis(filtered_x, filtered_y, function_type)
            
            # 检查分析结果是否成功
            if not analysis_results.get('success', False):
                error_msg = analysis_results.get('error', '未知错误')
                QMessageBox.critical(self, "分析失败", f"回归分析失败: {error_msg}")
                return
            
            # 保存当前结果，包含过滤信息
            self.current_results = analysis_results
            self.current_results['filtered_indices'] = filtered_indices
            self.current_results['iteration_history'] = iteration_history
            
            # 安全获取绘图数据
            curve_data = analysis_results.get('curve_data', ([], []))
            if not curve_data or len(curve_data) < 2 or len(curve_data[0]) == 0:
                QMessageBox.critical(self, "错误", "无法获取有效的绘图数据")
                return
            
            x_smooth, y_fit = curve_data
            
            # 评估拟合质量
            detailed_stats = analysis_results.get('detailed_stats', {})
            quality = assess_fitting_quality(detailed_stats)
            
            # 格式化结果
            results_text = format_math_results(analysis_results)
            
            # 绘制图形
            self.ax.clear()
            self.ax.scatter(filtered_x, filtered_y, color='blue', label='数据点')
            self.ax.plot(x_smooth, y_fit, color='red', label='回归曲线')
            
            # 添加标题和图例
            func_display_name = get_function_display_name(analysis_results.get('func_type', function_type))
            self.ax.set_title(f'{func_display_name}回归分析')
            self.ax.set_xlabel('X轴')
            self.ax.set_ylabel('Y轴')
            self.ax.legend()
            self.ax.grid(True)
            
            # 更新画布
            self.fig.tight_layout()
            self.canvas.draw()
            
            # 显示结果
            self.ui.textBrowser.setText(results_text)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"回归计算出错: {str(e)}")
    
    def output_statistics(self):
        """输出详细的统计学数据"""
        # 确保有计算结果
        if self.current_results is None:
            QMessageBox.warning(self, "警告", "请先进行回归计算")
            return
        
        try:
            # 安全获取详细统计数据
            stats = self.current_results.get('detailed_stats', {})
            fitting_result = self.current_results.get('fitting_result', {})
            
            # 格式化统计数据
            stats_text = "=== 详细统计学数据 ===\n\n"
            stats_text += f"数据点数量: {stats.get('n_points', 0)}\n"
            stats_text += f"\n--- 描述性统计 ---\n"
            stats_text += f"X平均值: {stats.get('mean_x', 0):.6f}\n"
            stats_text += f"X标准差: {stats.get('std_x', 0):.6f}\n"
            stats_text += f"Y平均值: {stats.get('mean_y', 0):.6f}\n"
            stats_text += f"Y标准差: {stats.get('std_y', 0):.6f}\n"
            stats_text += f"\n--- 回归统计 ---\n"
            
            # 从stats或fitting_result中获取回归统计量，优先使用stats
            corr_coef = stats.get('correlation', fitting_result.get('corr_coef', 0))
            r_squared = stats.get('r_squared', fitting_result.get('r_squared', 0))
            mse = fitting_result.get('mse', 0)
            rmse = stats.get('rmse', fitting_result.get('rmse', 0))
            
            stats_text += f"相关系数 (r): {corr_coef:.6f}\n"
            stats_text += f"决定系数 (R²): {r_squared:.6f}\n"
            stats_text += f"调整决定系数: {stats.get('adjusted_r_squared', 0):.6f}\n"
            stats_text += f"均方误差 (MSE): {mse:.6f}\n"
            stats_text += f"均方根误差 (RMSE): {rmse:.6f}\n"
            stats_text += f"平均绝对误差 (MAE): {stats.get('abs_residual_mean', 0):.6f}\n"
            
            # 添加额外的统计信息
            stats_text += f"\n--- 模型统计 ---\n"
            stats_text += f"参数数量: {stats.get('n_params', 0)}\n"
            stats_text += f"自由度: {stats.get('degrees_of_freedom', 0)}\n"
            stats_text += f"标准误差: {stats.get('standard_error', 0):.6f}\n"
            
            # 如果是线性回归，显示F统计量
            if fitting_result.get('func_type') == 'linear' and stats.get('f_statistic', 0) > 0:
                stats_text += f"F统计量: {stats['f_statistic']:.6f}\n"
            
            # 显示统计数据
            self.ui.textBrowser.setText(stats_text)
        except Exception as e:
            self.ui.textBrowser.setText(f"无法获取或显示统计数据: {str(e)}")
    
    def output_regression_coefficients(self):
        """输出回归方程对应系数"""
        # 确保有计算结果
        if self.current_results is None:
            QMessageBox.warning(self, "警告", "请先进行回归计算")
            return
        
        try:
            # 获取回归参数
            function_type = self.ui.comboBox.currentText()
            fitting_result = self.current_results.get('fitting_result', {})
            params = fitting_result.get('params', [])
            param_names = fitting_result.get('param_names', [])
            
            # 转换NumPy数组为Python列表
            if isinstance(params, np.ndarray):
                params = params.tolist()
            
            # 确保params是列表格式
            if not isinstance(params, list):
                params = [params] if params is not None else []
            
            # 获取函数键名
            func_key = get_function_type_from_display(function_type)
            
            # 格式化系数输出
            coeff_text = f"=== {function_type}回归系数 ===\n\n"
            
            # 检查参数是否存在
            if not params or len(params) == 0:
                coeff_text += "未找到回归参数\n"
                self.ui.textBrowser.setText(coeff_text)
                return
            
            # 安全获取标量参数的辅助函数
            def get_scalar_param(param):
                """确保参数是标量值"""
                if isinstance(param, (np.ndarray, list)) and len(param) == 1:
                    return param[0]
                elif isinstance(param, (np.ndarray, list)):
                    raise ValueError("参数应为标量值")
                return param
            
            # 根据不同的函数类型格式化系数
            try:
                if '线性' in function_type or func_key == 'linear':
                    if len(params) >= 2:
                        p0 = get_scalar_param(params[0])
                        p1 = get_scalar_param(params[1])
                        coeff_text += f"斜率 (a): {p0:.6f}\n"
                        coeff_text += f"截距 (b): {p1:.6f}\n"
                        coeff_text += f"\n回归方程: y = {p0:.4f}x + {p1:.4f}\n"
                    else:
                        coeff_text += f"参数不足，无法生成完整方程\n"
                elif '二次' in function_type or func_key == 'quadratic':
                    if len(params) >= 3:
                        p0 = get_scalar_param(params[0])
                        p1 = get_scalar_param(params[1])
                        p2 = get_scalar_param(params[2])
                        coeff_text += f"二次项系数 (a): {p0:.6f}\n"
                        coeff_text += f"一次项系数 (b): {p1:.6f}\n"
                        coeff_text += f"常数项 (c): {p2:.6f}\n"
                        coeff_text += f"\n回归方程: y = {p0:.4f}x² + {p1:.4f}x + {p2:.4f}\n"
                    else:
                        coeff_text += f"参数不足，无法生成完整方程\n"
                elif '三次' in function_type or func_key == 'cubic':
                    if len(params) >= 4:
                        p0 = get_scalar_param(params[0])
                        p1 = get_scalar_param(params[1])
                        p2 = get_scalar_param(params[2])
                        p3 = get_scalar_param(params[3])
                        coeff_text += f"三次项系数 (a): {p0:.6f}\n"
                        coeff_text += f"二次项系数 (b): {p1:.6f}\n"
                        coeff_text += f"一次项系数 (c): {p2:.6f}\n"
                        coeff_text += f"常数项 (d): {p3:.6f}\n"
                        coeff_text += f"\n回归方程: y = {p0:.4f}x³ + {p1:.4f}x² + {p2:.4f}x + {p3:.4f}\n"
                    else:
                        coeff_text += f"参数不足，无法生成完整方程\n"
                elif '指数' in function_type or func_key == 'exponential':
                    if len(params) >= 2:
                        p0 = get_scalar_param(params[0])
                        p1 = get_scalar_param(params[1])
                        coeff_text += f"系数 (a): {p0:.6f}\n"
                        coeff_text += f"指数系数 (b): {p1:.6f}\n"
                        coeff_text += f"\n回归方程: y = {p0:.4f}e^({p1:.4f}x)\n"
                    else:
                        coeff_text += f"参数不足，无法生成完整方程\n"
                elif '对数' in function_type or func_key == 'logarithmic':
                    if len(params) >= 2:
                        p0 = get_scalar_param(params[0])
                        p1 = get_scalar_param(params[1])
                        coeff_text += f"系数 (a): {p0:.6f}\n"
                        coeff_text += f"系数 (b): {p1:.6f}\n"
                        coeff_text += f"\n回归方程: y = {p0:.4f}ln(x) + {p1:.4f}\n"
                    else:
                        coeff_text += f"参数不足，无法生成完整方程\n"
                elif '幂' in function_type or func_key == 'power':
                    if len(params) >= 2:
                        p0 = get_scalar_param(params[0])
                        p1 = get_scalar_param(params[1])
                        coeff_text += f"系数 (a): {p0:.6f}\n"
                        coeff_text += f"指数 (b): {p1:.6f}\n"
                        coeff_text += f"\n回归方程: y = {p0:.4f}x^{p1:.4f}\n"
                    else:
                        coeff_text += f"参数不足，无法生成完整方程\n"
                elif '正弦' in function_type or func_key == 'sine':
                    if len(params) >= 4:
                        p0 = get_scalar_param(params[0])
                        p1 = get_scalar_param(params[1])
                        p2 = get_scalar_param(params[2])
                        p3 = get_scalar_param(params[3])
                        coeff_text += f"振幅 (A): {p0:.6f}\n"
                        coeff_text += f"角频率 (ω): {p1:.6f}\n"
                        coeff_text += f"相位 (φ): {p2:.6f}\n"
                        coeff_text += f"偏移 (C): {p3:.6f}\n"
                        coeff_text += f"\n回归方程: y = {p0:.4f}sin({p1:.4f}x + {p2:.4f}) + {p3:.4f}\n"
                    else:
                        coeff_text += f"参数不足，无法生成完整方程\n"
                else:
                    # 通用参数显示
                    coeff_text += "参数:\n"
                    min_len = min(len(params), len(param_names)) if param_names else len(params)
                    
                    # 确保param_names是列表格式
                    if not isinstance(param_names, list):
                        param_names = [param_names] if param_names is not None else []
                    
                    for i in range(min_len):
                        param = get_scalar_param(params[i])
                        name = param_names[i] if i < len(param_names) else f"参数 {i+1}"
                        coeff_text += f"{name}: {param:.6f}\n"
                    
                    # 如果有param_names，尝试构建方程
                    if param_names:
                        coeff_text += "\n回归方程格式可能需要根据具体模型调整\n"
            except Exception as e:
                # 如果特定格式输出失败，使用通用格式
                coeff_text += "参数详情:\n"
                
                # 确保param_names是列表格式
                if not isinstance(param_names, list):
                    param_names = [param_names] if param_names is not None else []
                
                if param_names and len(param_names) >= len(params):
                    for i, param in enumerate(params):
                        try:
                            scalar_param = get_scalar_param(param)
                            coeff_text += f"{param_names[i]}: {scalar_param:.6f}\n"
                        except:
                            coeff_text += f"{param_names[i]}: [数组值]\n"
                else:
                    for i, param in enumerate(params):
                        try:
                            scalar_param = get_scalar_param(param)
                            coeff_text += f"参数 {i+1}: {scalar_param:.6f}\n"
                        except:
                            coeff_text += f"参数 {i+1}: [数组值]\n"
            
            # 显示系数
            self.ui.textBrowser.setText(coeff_text)
        except Exception as e:
            self.ui.textBrowser.setText(f"显示回归系数时出错: {str(e)}")
    
    def clear_data(self):
        """清除所有数据"""
        try:
            # 清除文本框内容
            self.ui.textEdit.clear()
            self.ui.textEdit_2.clear()
            
            # 清空内部数据
            self.x_data = []
            self.y_data = []
            self.current_results = None
            
            # 清空绘图区域
            self.ax.clear()
            self.ax.set_title('数据可视化')
            self.ax.set_xlabel('X轴')
            self.ax.set_ylabel('Y轴')
            self.ax.grid(True)
            self.canvas.draw()
            
            # 清空结果显示
            self.ui.textBrowser.clear()
            
            # 更新状态栏
            self.statusBar().showMessage("数据已清除", 3000)
            
            # 显示提示
            QMessageBox.information(self, "成功", "所有数据已清除")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"清除数据时出错: {str(e)}")
    
    def return_to_mode_select(self):
        """返回模式选择界面"""
        try:
            # 延迟导入以避免循环导入
            from gui.ui_choose_mode import ChooseModeWindow
            
            # 关闭当前窗口
            self.close()
            
            # 创建并显示模式选择窗口
            self.mode_window = ChooseModeWindow()
            self.mode_window.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"返回模式选择界面时出错: {str(e)}")
            self.close()  # 即使出错也尝试关闭当前窗口

