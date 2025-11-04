'''数学解题模式，主要是进行数据点(x,y)的回归分析，根据从界面上选择的回归类型
显示回归公式，计算回归公式中各项的系数，并且显示相关的统计学数据，如
方差，标准差，相关系数，均方误差等
'''

import numpy as np
from typing import Dict, Any, Tuple, List
from .fitting_functions import perform_fitting, generate_curve_points, validate_data, calculate_statistics

def iterative_filter_outliers(x_data: np.ndarray, y_data: np.ndarray, func_type: str, threshold: float = 3.0, max_iterations: int = 5) -> Tuple[np.ndarray, np.ndarray, List[int], List[Dict[str, Any]]]:
    """迭代过滤异常值的函数
    
    Args:
        x_data: x坐标数据数组
        y_data: y坐标数据数组
        func_type: 拟合函数类型
        threshold: 异常值过滤阈值，从界面传入
        max_iterations: 最大迭代次数，默认为5
        
    Returns:
        过滤后的x数据、过滤后的y数据、被过滤的异常点索引列表、迭代历史记录
    """
    # 记录迭代历史
    iteration_history = []
    
    # 初始化当前数据
    current_x = np.array(x_data, dtype=float).copy()
    current_y = np.array(y_data, dtype=float).copy()
    current_indices = np.arange(len(current_x))
    original_total_indices = len(x_data)
    
    for iteration in range(max_iterations):
        # 执行拟合以获取曲线
        fitting_result = perform_fitting(current_x, current_y, func_type)
        
        if not fitting_result.get('success', False):
            break  # 如果拟合失败，停止迭代
        
        # 获取拟合结果
        y_pred = fitting_result.get('y_fit', np.zeros_like(current_y))
        
        # 计算残差
        residuals = current_y - y_pred
        
        # 计算残差统计量
        median_residual = np.median(residuals)
        mad = np.median(np.abs(residuals - median_residual))
        
        # 如果MAD为0，使用标准差
        if mad == 0:
            std_residual = np.std(residuals)
            # 使用相对阈值而非固定阈值
            actual_threshold = threshold * std_residual if std_residual > 0 else threshold
        else:
            # 使用MAD和传入的阈值参数
            actual_threshold = threshold * mad / 0.6745
        
        # 创建掩码，保留绝对值残差小于阈值的点
        mask = np.abs(residuals - median_residual) <= actual_threshold
        
        # 计算移除的点数量
        removed_count = len(current_x) - np.sum(mask)
        
        # 记录本次迭代信息
        iteration_history.append({
            'iteration': iteration + 1,
            'removed_count': removed_count,
            'remaining_count': int(np.sum(mask)),
            'threshold_used': float(actual_threshold)
        })
        
        # 如果没有过滤掉任何点，停止迭代
        if removed_count == 0:
            break
        
        # 更新数据和索引
        current_x = current_x[mask]
        current_y = current_y[mask]
        current_indices = current_indices[mask]
        
        # 如果数据点少于5个，停止迭代
        if len(current_x) < 5:
            break
    
    # 确定原始数据中的索引
    final_filtered_indices = []
    
    # 计算被过滤掉的索引
    all_indices = np.arange(original_total_indices)
    kept_indices = [all_indices[i] for i in current_indices]
    final_filtered_indices = [i for i in all_indices if i not in kept_indices]
    
    return current_x, current_y, final_filtered_indices, iteration_history

def math_mode_analysis(x_data: np.ndarray, y_data: np.ndarray, func_type: str) -> Dict[str, Any]:
    """数学模式的回归分析"""
    try:
        # 确保输入数据是numpy数组
        valid_x = np.array(x_data, dtype=float)
        valid_y = np.array(y_data, dtype=float)
        
        # 直接验证数据有效性，不经过validate_data函数（它需要字符串输入）
        # 检查数据是否有效
        if len(valid_x) != len(valid_y):
            return {
                'success': False,
                'error': 'x和y数据点数量不匹配'
            }
        
        if len(valid_x) < 2:
            return {
                'success': False,
                'error': '数据点数量过少，至少需要2个数据点'
            }
        
        if not (np.isfinite(valid_x).all() and np.isfinite(valid_y).all()):
            return {
                'success': False,
                'error': '数据包含无效值（NaN或无穷大）'
            }
        
        # 获取函数类型键名（支持显示名称或键名）
        func_key = get_function_type_from_display(func_type)
        
        # 执行拟合
        fitting_result = perform_fitting(valid_x, valid_y, func_key)
        
        if not fitting_result.get('success', False):
            return {
                'success': False,
                'error': fitting_result.get('error', '拟合失败')
            }
        
        # 计算更详细的统计信息
        detailed_stats = calculate_detailed_statistics(valid_x, valid_y, fitting_result)
        
        # 生成拟合曲线数据
        x_curve, y_curve = generate_curve_points(valid_x, func_key, fitting_result['params'])
        
        # 整理结果，确保包含UI所需的所有字段
        result = {
            'success': True,
            'input_data': (valid_x, valid_y),
            'fitting_result': fitting_result,
            'detailed_stats': detailed_stats,
            'curve_data': (x_curve, y_curve),
            # 确保包含UI直接访问的字段
            'y_pred': fitting_result.get('y_fit', np.array([])),
            'params': fitting_result.get('params', []),
            'func_type': func_key
        }
        
        return result
    except Exception as e:
        return {
            'success': False,
            'error': f'分析过程中发生错误: {str(e)}'
        }

def calculate_detailed_statistics(x_data: np.ndarray, y_data: np.ndarray, fitting_result: Dict[str, Any]) -> Dict[str, float]:
    """计算详细的统计信息"""
    try:
        # 获取基本统计信息
        basic_stats = calculate_statistics(x_data, y_data)
        
        # 确保fitting_result包含必要的字段
        if not isinstance(fitting_result, dict):
            # 如果拟合结果不是字典，返回基本统计
            return basic_stats
        
        # 安全获取拟合结果中的数据
        y_fit = fitting_result.get('y_fit', np.zeros_like(y_data))
        residuals = fitting_result.get('residuals', y_data - y_fit)
        
        # 计算额外的统计量
        n = len(y_data)
        k = len(fitting_result.get('params', []))  # 参数数量
        
        # 计算调整后的R²
        r_squared = fitting_result.get('r_squared', 0)
        adj_r_squared = 0
        if n > k + 1:
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        
        # 计算标准误差
        std_error = 0
        if n > k:
            std_error = np.sqrt(np.sum(residuals**2) / (n - k))
        
        # 计算F统计量（用于线性回归的显著性检验）
        f_statistic = 0
        if fitting_result.get('func_type') == 'linear' and n > k and 'ss_tot' in fitting_result:
            ss_res = np.sum(residuals**2)
            ss_reg = fitting_result['ss_tot'] - ss_res
            if k > 1 and ss_res > 0:
                f_statistic = (ss_reg / (k - 1)) / (ss_res / (n - k))
        
        # 计算残差的统计信息
        residual_stats = {
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals)),
            'residual_min': float(np.min(residuals)),
            'residual_max': float(np.max(residuals)),
            'abs_residual_mean': float(np.mean(np.abs(residuals)))
        }
        
        # 组合所有统计信息
        detailed_stats = {
            **basic_stats,
            'adjusted_r_squared': adj_r_squared,
            'standard_error': std_error,
            'f_statistic': f_statistic,
            'n_params': k,
            'degrees_of_freedom': n - k,
            **residual_stats,
            # 添加从fitting_result中获取的关键指标
            'r_squared': r_squared,
            'correlation': fitting_result.get('corr_coef', 0),
            'rmse': fitting_result.get('rmse', 0)
        }
        
        return detailed_stats
    except Exception as e:
        # 返回基本统计信息，确保函数不会中断
        try:
            return calculate_statistics(x_data, y_data)
        except:
            return {
                'n_points': len(x_data),
                'mean_x': float(np.mean(x_data)) if len(x_data) > 0 else 0,
                'std_x': float(np.std(x_data)) if len(x_data) > 0 else 0,
                'mean_y': float(np.mean(y_data)) if len(y_data) > 0 else 0,
                'std_y': float(np.std(y_data)) if len(y_data) > 0 else 0
            }

def format_math_results(analysis_results: Dict[str, Any]) -> str:
    """格式化数学模式的结果输出"""
    try:
        if not analysis_results.get('success', False):
            return f"错误: {analysis_results.get('error', '未知错误')}"
        
        output = []
        fitting_result = analysis_results.get('fitting_result', {})
        detailed_stats = analysis_results.get('detailed_stats', {})
        
        # 检查是否有迭代过滤信息
        filtered_indices = analysis_results.get('filtered_indices', [])
        iteration_history = analysis_results.get('iteration_history', [])
        
        if filtered_indices or iteration_history:
            output.append("=== 异常值过滤信息 ===")
            output.append(f"过滤的异常点数量: {len(filtered_indices)}")
            
            # 显示迭代历史
            if iteration_history:
                output.append(f"迭代次数: {len(iteration_history)}")
                for history in iteration_history:
                    output.append(f"  迭代 {history.get('iteration', 0)}: 移除 {history.get('removed_count', 0)} 个点, 剩余 {history.get('remaining_count', 0)} 个点")
            
            output.append("")
        
        # 拟合函数信息
        output.append("=== 回归分析结果 ===")
        output.append(f"拟合函数: {fitting_result.get('func_name', '未知')}")
        output.append(f"函数公式: {fitting_result.get('formula', '未知')}")
        
        # 参数估计
        output.append("\n=== 参数估计 ===")
        param_names = fitting_result.get('param_names', [])
        params = fitting_result.get('params', [])
        param_errors = fitting_result.get('param_errors', [])
        
        # 确保参数格式正确
        if isinstance(params, np.ndarray):
            params = params.tolist()
        if not isinstance(params, list):
            params = [params] if params is not None else []
            
        if isinstance(param_names, np.ndarray):
            param_names = param_names.tolist()
        if not isinstance(param_names, list):
            param_names = [param_names] if param_names is not None else []
            
        if isinstance(param_errors, np.ndarray):
            param_errors = param_errors.tolist()
        if not isinstance(param_errors, list):
            param_errors = [param_errors] if param_errors is not None else []
        
        # 确保参数长度一致
        min_len = min(len(param_names), len(params))
        for i in range(min_len):
            # 安全获取标量参数
            param_value = params[i]
            if isinstance(param_value, (np.ndarray, list)) and len(param_value) == 1:
                param_value = param_value[0]
            
            error_text = ""
            if i < len(param_errors):
                error_value = param_errors[i]
                if isinstance(error_value, (np.ndarray, list)) and len(error_value) == 1:
                    error_value = error_value[0]
                if isinstance(error_value, (int, float, np.number)):
                    error_text = f" ± {float(error_value):.6f}"
            
            # 确保能正确格式化
            if isinstance(param_value, (int, float, np.number)):
                output.append(f"{param_names[i]} = {float(param_value):.6f}{error_text}")
            else:
                output.append(f"{param_names[i]} = {str(param_value)}{error_text}")
        
        # 拟合优度
        output.append("\n=== 拟合优度 ===")
        # 安全获取标量统计值
        r_squared = detailed_stats.get('r_squared', 0)
        if isinstance(r_squared, (np.ndarray, list)) and len(r_squared) == 1:
            r_squared = r_squared[0]
            
        adj_r_squared = detailed_stats.get('adjusted_r_squared', 0)
        if isinstance(adj_r_squared, (np.ndarray, list)) and len(adj_r_squared) == 1:
            adj_r_squared = adj_r_squared[0]
            
        mse = fitting_result.get('mse', 0)
        if isinstance(mse, (np.ndarray, list)) and len(mse) == 1:
            mse = mse[0]
            
        rmse = detailed_stats.get('rmse', 0)
        if isinstance(rmse, (np.ndarray, list)) and len(rmse) == 1:
            rmse = rmse[0]
            
        std_error = detailed_stats.get('standard_error', 0)
        if isinstance(std_error, (np.ndarray, list)) and len(std_error) == 1:
            std_error = std_error[0]
        
        output.append(f"决定系数 (R²): {float(r_squared):.6f}")
        output.append(f"调整决定系数 (Adj R²): {float(adj_r_squared):.6f}")
        output.append(f"均方误差 (MSE): {float(mse):.6f}")
        output.append(f"均方根误差 (RMSE): {float(rmse):.6f}")
        output.append(f"标准误差: {float(std_error):.6f}")
        
        # 相关系数
        output.append("\n=== 相关性分析 ===")
        correlation = detailed_stats.get('correlation', 0)
        if isinstance(correlation, (np.ndarray, list)) and len(correlation) == 1:
            correlation = correlation[0]
        output.append(f"相关系数: {float(correlation):.6f}")
        
        # 数据统计
        output.append("\n=== 数据统计 ===")
        output.append(f"数据点数量: {detailed_stats.get('n_points', 0)}")
        
        # 安全获取统计值
        mean_x = detailed_stats.get('mean_x', 0)
        if isinstance(mean_x, (np.ndarray, list)) and len(mean_x) == 1:
            mean_x = mean_x[0]
            
        std_x = detailed_stats.get('std_x', 0)
        if isinstance(std_x, (np.ndarray, list)) and len(std_x) == 1:
            std_x = std_x[0]
            
        mean_y = detailed_stats.get('mean_y', 0)
        if isinstance(mean_y, (np.ndarray, list)) and len(mean_y) == 1:
            mean_y = mean_y[0]
            
        std_y = detailed_stats.get('std_y', 0)
        if isinstance(std_y, (np.ndarray, list)) and len(std_y) == 1:
            std_y = std_y[0]
        
        output.append(f"X均值: {float(mean_x):.4f}, X标准差: {float(std_x):.4f}")
        output.append(f"Y均值: {float(mean_y):.4f}, Y标准差: {float(std_y):.4f}")
        
        # 残差分析
        output.append("\n=== 残差分析 ===")
        
        # 安全获取残差统计值
        residual_mean = detailed_stats.get('residual_mean', 0)
        if isinstance(residual_mean, (np.ndarray, list)) and len(residual_mean) == 1:
            residual_mean = residual_mean[0]
            
        residual_std = detailed_stats.get('residual_std', 0)
        if isinstance(residual_std, (np.ndarray, list)) and len(residual_std) == 1:
            residual_std = residual_std[0]
            
        residual_min = detailed_stats.get('residual_min', 0)
        if isinstance(residual_min, (np.ndarray, list)) and len(residual_min) == 1:
            residual_min = residual_min[0]
            
        residual_max = detailed_stats.get('residual_max', 0)
        if isinstance(residual_max, (np.ndarray, list)) and len(residual_max) == 1:
            residual_max = residual_max[0]
            
        abs_residual_mean = detailed_stats.get('abs_residual_mean', 0)
        if isinstance(abs_residual_mean, (np.ndarray, list)) and len(abs_residual_mean) == 1:
            abs_residual_mean = abs_residual_mean[0]
        
        output.append(f"残差均值: {float(residual_mean):.6f}")
        output.append(f"残差标准差: {float(residual_std):.6f}")
        output.append(f"残差范围: [{float(residual_min):.6f}, {float(residual_max):.6f}]")
        output.append(f"平均绝对残差: {float(abs_residual_mean):.6f}")
        
        # 模型信息
        output.append("\n=== 模型信息 ===")
        output.append(f"参数数量: {detailed_stats.get('n_params', 0)}")
        output.append(f"自由度: {detailed_stats.get('degrees_of_freedom', 0)}")
        
        # 如果是线性回归，显示F统计量
        if fitting_result.get('func_type') == 'linear' and detailed_stats.get('f_statistic', 0) > 0:
            f_statistic = detailed_stats['f_statistic']
            if isinstance(f_statistic, (np.ndarray, list)) and len(f_statistic) == 1:
                f_statistic = f_statistic[0]
            output.append(f"F统计量: {float(f_statistic):.6f}")
        
        # 添加拟合质量评估
        output.append("\n=== 拟合质量评估 ===")
        quality_assessment = assess_fitting_quality(detailed_stats)
        for key, value in quality_assessment.items():
            output.append(f"{key}: {value}")
        
        return "\n".join(output)
    except Exception as e:
        return f"格式化结果时发生错误: {str(e)}"

def assess_fitting_quality(detailed_stats: Dict[str, float]) -> Dict[str, str]:
    """评估拟合质量，仅使用detailed_stats参数"""
    try:
        # 从detailed_stats中获取必要的指标
        r_squared = detailed_stats.get('r_squared', 0)
        rmse = detailed_stats.get('rmse', 0)
        corr_coef = detailed_stats.get('correlation', 0)
        
        assessment = {}
        
        # 基于R²的评估
        if r_squared >= 0.95:
            assessment['整体拟合效果'] = '优秀'
        elif r_squared >= 0.85:
            assessment['整体拟合效果'] = '良好'
        elif r_squared >= 0.70:
            assessment['整体拟合效果'] = '一般'
        elif r_squared >= 0.50:
            assessment['整体拟合效果'] = '较差'
        else:
            assessment['整体拟合效果'] = '很差'
        
        # 基于相关系数的评估
        abs_corr = abs(corr_coef)
        if abs_corr >= 0.9:
            assessment['变量相关性'] = '强相关'
        elif abs_corr >= 0.7:
            assessment['变量相关性'] = '中度相关'
        elif abs_corr >= 0.4:
            assessment['变量相关性'] = '弱相关'
        else:
            assessment['变量相关性'] = '极弱相关或无相关'
        
        # 相关性方向
        if corr_coef > 0:
            assessment['相关性方向'] = '正相关'
        elif corr_coef < 0:
            assessment['相关性方向'] = '负相关'
        else:
            assessment['相关性方向'] = '无相关'
        
        # 模型适用性建议
        if r_squared < 0.7:
            assessment['模型建议'] = '考虑尝试其他类型的拟合函数'
        else:
            assessment['模型建议'] = '当前模型拟合效果良好'
        
        return assessment
    except Exception as e:
        # 返回默认评估结果
        return {
            '整体拟合效果': '无法评估',
            '变量相关性': '无法评估',
            '相关性方向': '无法确定',
            '模型建议': '请检查数据质量'
        }

def generate_math_plot_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """生成用于绘图的数据"""
    try:
        if not analysis_results.get('success', False):
            return {}
        
        # 安全获取数据
        input_data = analysis_results.get('input_data', ([], []))
        curve_data = analysis_results.get('curve_data', ([], []))
        fitting_result = analysis_results.get('fitting_result', {})
        
        plot_data = {
            'x_data': input_data[0].tolist() if hasattr(input_data[0], 'tolist') else list(input_data[0]),
            'y_data': input_data[1].tolist() if hasattr(input_data[1], 'tolist') else list(input_data[1]),
            'x_curve': curve_data[0].tolist() if hasattr(curve_data[0], 'tolist') else list(curve_data[0]),
            'y_curve': curve_data[1].tolist() if hasattr(curve_data[1], 'tolist') else list(curve_data[1]),
            'func_type': fitting_result.get('func_type', ''),
            'func_name': fitting_result.get('func_name', '')
        }
        
        return plot_data
    except Exception as e:
        return {}

def get_available_function_types() -> List[str]:
    """获取所有可用的函数类型"""
    return [
        '线性函数 (y = a*x + b)',
        '二次函数 (y = a*x² + b*x + c)',
        '三次函数 (y = a*x³ + b*x² + c*x + d)',
        '指数函数 (y = a*e^(b*x))',
        '对数函数 (y = a*ln(x) + b)',
        '幂函数 (y = a*x^b)',
        '正弦函数 (y = a*sin(b*x + c) + d)'
    ]

def validate_function_type(func_type: str) -> bool:
    """验证函数类型是否有效"""
    try:
        # 检查是否为有效的显示名称
        function_types = get_available_function_types()
        if func_type in function_types:
            return True
        
        # 检查是否为显示名称的前缀（如'线性'而不是'线性函数'）
        for display_name in function_types:
            if display_name.startswith(func_type):
                return True
        
        return False
    except:
        return False

def get_function_type_from_display(display_name: str) -> str:
    """从显示名称获取函数类型键名"""
    try:
        # 定义显示名称到函数类型的映射
        name_map = {
            '线性函数 (y = a*x + b)': 'linear',
            '二次函数 (y = a*x² + b*x + c)': 'quadratic',
            '三次函数 (y = a*x³ + b*x² + c*x + d)': 'cubic',
            '指数函数 (y = a*e^(b*x))': 'exponential',
            '对数函数 (y = a*ln(x) + b)': 'logarithmic',
            '幂函数 (y = a*x^b)': 'power',
            '正弦函数 (y = a*sin(b*x + c) + d)': 'sine',
            # 简化名称映射
            '线性': 'linear',
            '二次': 'quadratic',
            '三次': 'cubic',
            '指数': 'exponential',
            '对数': 'logarithmic',
            '幂': 'power',
            '正弦': 'sine'
        }
        
        # 直接查找完整显示名称
        if display_name in name_map:
            return name_map[display_name]
        
        # 检查是否部分匹配显示名称
        for full_name, key in name_map.items():
            if full_name.startswith(display_name):
                return key
        
        # 尝试从显示名称中提取关键字段
        for keyword, key in {
            '线性': 'linear',
            '二次': 'quadratic',
            '三次': 'cubic',
            '指数': 'exponential',
            '对数': 'logarithmic',
            '幂': 'power',
            '正弦': 'sine'
        }.items():
            if keyword in display_name:
                return key
        
        return 'linear'  # 默认返回线性函数
    except:
        return 'linear'  # 默认返回线性函数

def get_function_display_name(func_type: str) -> str:
    """获取函数的显示名称"""
    try:
        # 定义函数类型到显示名称的映射
        type_map = {
            'linear': '线性函数 (y = a*x + b)',
            'quadratic': '二次函数 (y = a*x² + b*x + c)',
            'cubic': '三次函数 (y = a*x³ + b*x² + c*x + d)',
            'exponential': '指数函数 (y = a*e^(b*x))',
            'logarithmic': '对数函数 (y = a*ln(x) + b)',
            'power': '幂函数 (y = a*x^b)',
            'sine': '正弦函数 (y = a*sin(b*x + c) + d)'
        }
        
        if func_type in type_map:
            return type_map[func_type]
        
        # 检查func_type是否已经是显示名称格式
        function_types = get_available_function_types()
        if func_type in function_types:
            return func_type
        
        return f"未知函数 ({func_type})"
    except:
        return "未知函数"