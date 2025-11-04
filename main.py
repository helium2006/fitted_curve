#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
曲线拟合程序 - 主入口
支持实验数据模式和数学模式两种工作方式
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox
from PySide6.QtCore import Qt
from gui.ui_choose_mode import ChooseModeWindow

def main():
    """主函数"""
    # 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("曲线拟合程序")
    app.setOrganizationName("Curve Fitting Tools")
    
    # 确保中文正常显示
    font = app.font()
    font.setFamily("SimHei")
    app.setFont(font)
    
    # 创建并显示模式选择窗口
    window = ChooseModeWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())

if __name__ == "__main__":
    main()