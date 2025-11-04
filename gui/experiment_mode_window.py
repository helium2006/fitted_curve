# -*- coding: utf-8 -*-

from PySide6.QtWidgets import QMainWindow, QGraphicsScene, QMessageBox
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from PIL import Image

# 导入修改后的UI
from .ui_exp_mode_update import Ui_MainWindow

# 导入工具函数
from utils.experiment_mode_utils import experiment_mode_analysis, find_best_polynomial_fit, \
    evaluate_curve_quality, format_experiment_results, generate_experiment_plot_data
from utils.fitting_functions import filter_outliers, get_function_info, fit_data

class ExperimentModeWindow(QMainWindow):
    """实验数据模式窗口"""
    def __init__(self):
        super().__init__()
        
        # 设置UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 初始化数据
        self.x_data = []
        self.y_data = []
        
        # 初始化绘图
        self.init_plot()
        
        # 设置拟合方法下拉框
        self.ui.comboBox.addItems(['多项式拟合', '平滑样条拟合'])
        
        # 连接信号和槽
        self.ui.pushButton_7.clicked.connect(self.show_example_data)
        self.ui.pushButton_8.clicked.connect(self.draw_fitting_curve)
        self.ui.pushButton_9.clicked.connect(self.output_statistics)
        self.ui.pushButton_10.clicked.connect(self.export_as_image)
        self.ui.pushButton_11.clicked.connect(self.return_to_mode_select)
        
        # 设置窗口属性
        self.setWindowTitle("曲线拟合-实验数据模式")
        self.setFixedSize(1000, 800)
    
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
    
    def show_example_data(self):
        """显示示例数据"""
        # 生成示例数据
        x = np.linspace(0, 10, 20)
        y = 2 * x**2 + 3 * x + 1 + np.random.normal(0, 5, 20)
        
        # 添加一些异常值
        y[3] += 50
        y[15] -= 40
        
        # 填充到文本框
        self.ui.textEdit.setPlainText('\n'.join([f'{val:.4f}' for val in x]))
        self.ui.textEdit_2.setPlainText('\n'.join([f'{val:.4f}' for val in y]))
        
        # 更新内部数据
        self.x_data = x.tolist()
        self.y_data = y.tolist()
        
        # 显示提示
        QMessageBox.information(self, "成功", "示例数据已加载")
    
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
    
    def draw_fitting_curve(self):
        """绘制拟合曲线并显示详细分析结果，支持同时输出合理性分析选项"""
        # 显示加载状态
        self.statusBar().showMessage("正在处理数据并进行拟合分析...")
        
        # 解析输入数据
        x_data, y_data = self.parse_input_data()
        if x_data is None or y_data is None:
            self.statusBar().showMessage("数据解析失败", 3000)
            QMessageBox.warning(self, "警告", "请先输入数据")
            return
        
        # 更新内部数据
        self.x_data = x_data
        self.y_data = y_data
        
        # 获取拟合参数
        fit_method = self.ui.comboBox.currentText()
        # 根据要求，过滤异常值只使用迭代过滤
        enable_iterative_filter = self.ui.radioButton_13.isChecked()
        iteration_count = self.ui.spinBox_iterations.value() if enable_iterative_filter else 0
        # 异常值阈值就是迭代过滤阈值
        iteration_threshold = self.ui.doubleSpinBox.value() if enable_iterative_filter else 0.1
        show_rationality = self.ui.radioButton_14.isChecked()
        
        # 准备数据用于分析
        x_array = np.array(x_data)
        y_array = np.array(y_data)
        
        # 执行完整的实验模式分析
        try:
            # 调用实验模式分析函数，获取完整的分析结果
            analysis_results = experiment_mode_analysis(
                x_array, 
                y_array, 
                # 根据要求，不再使用enable_outlier_filter，只使用迭代过滤
                enable_outlier_filter=False,
                fit_method=fit_method,
                enable_iterative_filter=enable_iterative_filter,
                iteration_count=iteration_count,
                iteration_threshold=iteration_threshold
            )
            
            # 从分析结果中获取数据
            filtered_x, filtered_y = analysis_results.get('filtered_data', (x_array, y_array))
            filtered_indices = analysis_results.get('filtered_indices', [])
            
            # 确定拟合曲线数据
            if 'best_poly_fit' in analysis_results and analysis_results['best_poly_fit']:
                # 多项式拟合
                best_fit = analysis_results['best_poly_fit']
                x_smooth = np.linspace(min(filtered_x) * 0.9, max(filtered_x) * 1.1, 1000)
                y_fit = np.polyval(best_fit['coeffs'], x_smooth)
            elif 'smooth_curve' in analysis_results and analysis_results['smooth_curve']:
                # 平滑样条拟合
                x_smooth, y_fit = analysis_results['smooth_curve']
            else:
                # 如果没有拟合曲线，生成默认的
                x_smooth = np.linspace(min(filtered_x) * 0.9, max(filtered_x) * 1.1, 1000)
                y_fit = np.zeros_like(x_smooth)  # 占位符
            
            # 绘制图形
            self.ax.clear()
            
            # 绘制原始数据点
            self.ax.scatter(x_array, y_array, color='blue', alpha=0.6, label='原始数据')
            
            # 如果过滤了异常值，只显示用红叉标注的异常点
            if filtered_indices:
                # 标记被过滤的异常点
                outlier_x = x_array[filtered_indices]
                outlier_y = y_array[filtered_indices]
                self.ax.scatter(outlier_x, outlier_y, color='red', s=100, alpha=0.7, marker='x', label='异常点')
            elif len(filtered_x) < len(x_array):
                # 兼容旧版本的过滤方式，同样只标记异常点
                # 找出被过滤的异常点索引
                kept_set = set(zip(filtered_x, filtered_y))
                outlier_x = []
                outlier_y = []
                for i, (x, y) in enumerate(zip(x_array, y_array)):
                    if (x, y) not in kept_set:
                        outlier_x.append(x)
                        outlier_y.append(y)
                if outlier_x:
                    self.ax.scatter(outlier_x, outlier_y, color='red', s=100, alpha=0.7, marker='x', label='异常点')
            
            # 绘制拟合曲线
            self.ax.plot(x_smooth, y_fit, color='red', linewidth=2, label='拟合曲线')
            
            # 设置坐标轴标签
            x_label = self.ui.lineEdit_4.text() if self.ui.lineEdit_4.text() else 'X轴'
            y_label = self.ui.lineEdit_5.text() if self.ui.lineEdit_5.text() else 'Y轴'
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(y_label)
            
            # 添加标题和图例
            self.ax.set_title(f'实验数据{"多项式" if fit_method == "多项式拟合" else "平滑样条"}拟合曲线')
            self.ax.legend()
            self.ax.grid(True, linestyle='--', alpha=0.7)
            
            # 更新画布
            self.fig.tight_layout()
            self.canvas.draw()
            
            # 根据选项决定输出内容
            if show_rationality:
                # 使用增强的格式化函数生成详细结果
                detailed_results = format_experiment_results(analysis_results)
            else:
                # 充分利用_generate_simplified_results函数，提供更直观的结果摘要
                detailed_results = self._generate_simplified_results(analysis_results)
            
            # 显示结果
            self.ui.textBrowser.setText(detailed_results)
            
            # 滚动到顶部
            self.ui.textBrowser.moveCursor(QTextCursor.Start)
            
            # 更新状态栏
            data_points = len(filtered_x)
            outliers = len(filtered_indices)
            status_msg = f"拟合完成 - 数据点: {data_points}, 异常点: {outliers}"
            if 'best_poly_fit' in analysis_results and analysis_results['best_poly_fit'] and 'r_squared' in analysis_results['best_poly_fit']:
                status_msg += f", R²: {analysis_results['best_poly_fit']['r_squared']:.4f}"
            if 'iteration_history' in analysis_results:
                status_msg += f", 迭代次数: {len(analysis_results['iteration_history'])}"
            self.statusBar().showMessage(status_msg, 5000)
            
        except Exception as e:
            import traceback
            self.statusBar().showMessage("拟合过程出错", 3000)
            QMessageBox.critical(self, "错误", f"拟合过程出错: {str(e)}")
            # 添加详细的错误信息到文本浏览器以便调试
            self.ui.textBrowser.setText(f"拟合过程出错: {str(e)}")
            self.ui.textBrowser.append(f"\n详细错误信息:\n{traceback.format_exc()}")
            
    def _generate_simplified_results(self, analysis_results: dict) -> str:
        """生成简化版的结果输出，聚焦关键信息"""
        output = []
        
        # 基本信息
        output.append("=== 拟合结果概览 ===")
        output.append("=" * 30)
        output.append(f"原始数据点数量: {len(analysis_results['original_data'][0])}")
        output.append(f"过滤后数据点数量: {len(analysis_results['filtered_data'][0])}")
        
        if len(analysis_results['filtered_indices']) > 0:
            outlier_percent = len(analysis_results['filtered_indices']) / len(analysis_results['original_data'][0]) * 100
            output.append(f"过滤的异常点: {len(analysis_results['filtered_indices'])} ({outlier_percent:.1f}%)")
        
        # 拟合信息
        quality = analysis_results['curve_quality']
        output.append("\n=== 拟合质量 ===")
        output.append(f"拟合优度等级: {quality['goodness_of_fit']}")
        output.append(f"数据代表性: {quality['data_representativeness']}")
        
        # 新增：残差正态分布合理性判断
        if 'normality_analysis' in quality and quality['normality_analysis']:
            normality_analysis = quality['normality_analysis']
            normality_icon = "✅" if normality_analysis['normality_assessment'] == "良好" else "⚠️" if normality_analysis['normality_assessment'] == "一般" else "❌"
            output.append(f"\n=== 合理性判断 ===")
            output.append(f"{normality_icon} 正态性评估: {normality_analysis['normality_assessment']}")
            # 只显示最核心的解释，避免输出过长
            interpretation = normality_analysis['normality_interpretation']
            if len(interpretation) > 60:
                interpretation = interpretation[:57] + "..."
            output.append(f"解释: {interpretation}")
        
        # 关键误差指标
        if quality['error_analysis']:
            error_analysis = quality['error_analysis']
            output.append(f"均方根误差(RMSE): {error_analysis.get('std_error', 0):.6f}")
            if 'mean_relative_error' in error_analysis:
                output.append(f"平均相对误差: {error_analysis['mean_relative_error']:.2f}%")
        
        # 简要建议
            if quality['recommendations']:
                output.append("\n=== 主要建议 ===")
                # 只显示前3条最重要的建议
                for i, rec in enumerate(quality['recommendations'][:3], 1):
                    output.append(f"{i}. {rec}")
                if len(quality['recommendations']) > 3:
                    output.append(f"... 还有 {len(quality['recommendations']) - 3} 条建议，请点击'输出统计学数据'查看完整分析")
            
            # 迭代过滤信息
            if 'iteration_history' in analysis_results and analysis_results['iteration_history']:
                output.append("\n=== 迭代过滤信息 ===")
                output.append(f"迭代次数: {len(analysis_results['iteration_history'])}")
                for iter_info in analysis_results['iteration_history']:
                    output.append(f"第{iter_info['iteration']}次迭代: 移除{iter_info['removed_count']}点, "
                                 f"剩余{iter_info['remaining_count']}点")
        
        return "\n".join(output)
    
    def output_statistics(self):
        """输出数据的统计信息，优化用户体验"""
        try:
            # 确保有数据
            if not self.x_data or not self.y_data:
                QMessageBox.warning(self, "警告", "请先绘制拟合曲线")
                return
            
            # 显示加载状态
            self.statusBar().showMessage("正在生成详细统计分析...")
            
            # 准备数据用于分析
            x_array = np.array(self.x_data)
            y_array = np.array(self.y_data)
            
            # 获取当前设置的参数
            # 根据要求，过滤异常值只使用迭代过滤
            enable_iterative_filter = self.ui.radioButton_13.isChecked()
            iteration_count = self.ui.spinBox_iterations.value() if enable_iterative_filter else 0
            iteration_threshold = self.ui.doubleSpinBox.value() if enable_iterative_filter else 0.1
            fit_method = self.ui.comboBox.currentText()
            
            # 调用实验模式分析函数，获取完整的分析结果，避免重复计算
            analysis_results = experiment_mode_analysis(
                x_array, 
                y_array, 
                enable_outlier_filter=False,  # 不再使用单次过滤
                fit_method=fit_method,
                enable_iterative_filter=enable_iterative_filter,
                iteration_count=iteration_count,
                iteration_threshold=iteration_threshold
            )
            
            # 使用format_experiment_results生成详细的统计和分析结果
            detailed_results = format_experiment_results(analysis_results)
            
            # 显示结果
            self.ui.textBrowser.setText(detailed_results)
            
            # 滚动到顶部
            self.ui.textBrowser.moveCursor(QTextCursor.Start)
            
            # 完成后更新状态栏
            data_points = len(analysis_results['filtered_data'][0])
            outliers = len(analysis_results['filtered_indices'])
            status_msg = f"统计分析完成 - 数据点: {data_points}, 异常点: {outliers}"
            if 'iteration_history' in analysis_results:
                status_msg += f", 迭代次数: {len(analysis_results['iteration_history'])}"
            self.statusBar().showMessage(status_msg, 5000)
            
        except Exception as e:
            import traceback
            self.statusBar().showMessage("统计分析过程出错", 3000)
            QMessageBox.critical(self, "错误", f"统计分析出错: {str(e)}")
            self.ui.textBrowser.setText(f"统计分析出错: {str(e)}")
            self.ui.textBrowser.append(f"\n详细错误信息:\n{traceback.format_exc()}")
    
    def export_as_image(self):
        """导出图表为图片"""
        try:
            # 创建一个临时缓冲区
            buf = io.BytesIO()
            
            # 保存图像到缓冲区
            self.fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            
            # 重置缓冲区指针
            buf.seek(0)
            
            # 使用PIL打开图像
            img = Image.open(buf)
            
            # 显示保存对话框
            from PySide6.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
            )
            
            if file_path:
                # 保存图像
                img.save(file_path)
                QMessageBox.information(self, "成功", f"图像已保存至: {file_path}")
            
            # 关闭缓冲区
            buf.close()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出图像失败: {str(e)}")
    
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
