import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Callable

# 定义各种拟合函数
def linear_func(x, a, b):
    """线性函数: y = a*x + b"""
    return a * x + b

def quadratic_func(x, a, b, c):
    """二次函数: y = a*x^2 + b*x + c"""
    return a * x**2 + b * x + c

def cubic_func(x, a, b, c, d):
    """三次函数: y = a*x^3 + b*x^2 + c*x + d"""
    return a * x**3 + b * x**2 + c * x + d

def exponential_func(x, a, b):
    """指数函数: y = a*e^(b*x)"""
    return a * np.exp(b * x)

def logarithmic_func(x, a, b):
    """对数函数: y = a*ln(x) + b"""
    return a * np.log(x) + b

def power_func(x, a, b):
    """幂函数: y = a*x^b"""
    return a * x**b

def sine_func(x, a, b, c, d):
    """正弦函数: y = a*sin(b*x + c) + d"""
    return a * np.sin(b * x + c) + d

# 获取函数信息和初始参数
def get_function_info(func_type: str) -> Dict[str, Any]:
    """获取拟合函数的相关信息"""
    func_dict = {
        'linear': {
            'function': linear_func,
            'name': '线性函数',
            'formula': 'y = a*x + b',
            'params': ['a', 'b'],
            'initial_guess': [1, 0]
        },
        'quadratic': {
            'function': quadratic_func,
            'name': '二次函数',
            'formula': 'y = a*x^2 + b*x + c',
            'params': ['a', 'b', 'c'],
            'initial_guess': [1, 1, 0]
        },
        'cubic': {
            'function': cubic_func,
            'name': '三次函数',
            'formula': 'y = a*x^3 + b*x^2 + c*x + d',
            'params': ['a', 'b', 'c', 'd'],
            'initial_guess': [1, 1, 1, 0]
        },
        'exponential': {
            'function': exponential_func,
            'name': '指数函数',
            'formula': 'y = a*e^(b*x)',
            'params': ['a', 'b'],
            'initial_guess': [1, 0.1]
        },
        'logarithmic': {
            'function': logarithmic_func,
            'name': '对数函数',
            'formula': 'y = a*ln(x) + b',
            'params': ['a', 'b'],
            'initial_guess': [1, 0]
        },
        'power': {
            'function': power_func,
            'name': '幂函数',
            'formula': 'y = a*x^b',
            'params': ['a', 'b'],
            'initial_guess': [1, 1]
        },
        'sine': {
            'function': sine_func,
            'name': '正弦函数',
            'formula': 'y = a*sin(b*x + c) + d',
            'params': ['a', 'b', 'c', 'd'],
            'initial_guess': [1, 1, 0, 0]
        }
    }
    return func_dict.get(func_type, func_dict['linear'])

# 执行曲线拟合
def perform_fitting(x_data: np.ndarray, y_data: np.ndarray, func_type: str) -> Dict[str, Any]:
    """执行曲线拟合，返回拟合结果"""
    try:
        # 获取函数信息
        func_info = get_function_info(func_type)
        func = func_info['function']
        initial_guess = func_info['initial_guess']
        
        # 为正弦函数设置更高的最大迭代次数
        if func_type == 'sine':
            # 正弦拟合通常需要更多迭代次数才能收敛
            popt, pcov = curve_fit(func, x_data, y_data, p0=initial_guess, maxfev=5000)
        else:
            popt, pcov = curve_fit(func, x_data, y_data, p0=initial_guess)
        
        # 计算拟合值
        y_fit = func(x_data, *popt)
        
        # 计算统计学数据
        residuals = y_data - y_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 计算标准误差
        std_errors = np.sqrt(np.diag(pcov))
        
        # 计算均方误差
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        
        # 计算相关系数
        if len(x_data) > 1:
            corr_coef = np.corrcoef(y_data, y_fit)[0, 1]
        else:
            corr_coef = 0
        
        # 整理结果
        result = {
            'success': True,
            'params': popt,
            'param_names': func_info['params'],
            'param_errors': std_errors,
            'func_type': func_type,
            'func_name': func_info['name'],
            'formula': func_info['formula'],
            'y_fit': y_fit,
            'r_squared': r_squared,
            'mse': mse,
            'rmse': rmse,
            'corr_coef': corr_coef,
            'residuals': residuals,
            'ss_res': ss_res,
            'ss_tot': ss_tot
        }
        
    except Exception as e:
        result = {
            'success': False,
            'error': str(e)
        }
    
    return result

# 过滤异常点
def filter_outliers(x_data: np.ndarray, y_data: np.ndarray, threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """使用改进的异常值检测方法过滤异常点，主要检测y值异常，返回过滤后的数据和被过滤的索引
    
    Args:
        x_data: x坐标数据数组
        y_data: y坐标数据数组
        threshold: 异常值检测阈值（倍数）
        
    Returns:
        过滤后的x数据、过滤后的y数据、被过滤的异常点索引列表
    """
    if threshold <= 0 or len(y_data) < 3:
        return x_data, y_data, []
    
    # 使用稳健的统计量：中位数绝对偏差(MAD)代替标准差
    # MAD对异常值的抵抗力更强，不会被异常值本身严重影响
    median_y = np.median(y_data)
    mad = np.median(np.abs(y_data - median_y))
    
    # 如果所有值都相同，MAD会为0，这时我们退回到标准差方法
    if mad == 0:
        mean_y = np.mean(y_data)
        std_y = np.std(y_data)
        # 如果标准差也为0（所有值相同），则没有异常值
        if std_y == 0:
            return x_data, y_data, []
        mask = np.abs(y_data - mean_y) <= threshold * std_y
    else:
        # 使用MAD计算Z-score
        # 对于正态分布数据，MAD ≈ 0.6745 * σ，所以需要乘以这个常数进行标准化
        z_scores = 0.6745 * (y_data - median_y) / mad
        mask = np.abs(z_scores) <= threshold
    
    # 只检测y值异常，根据用户反馈，一般实验情况下x轴取值是合理的
    
    # 获取被过滤掉的索引（异常点的索引）
    filtered_indices = np.where(~mask)[0].tolist()
    
    return x_data[mask], y_data[mask], filtered_indices

# 计算统计数据
def calculate_statistics(x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, float]:
    """计算数据的统计信息"""
    if len(y_data) == 0:
        return {}
    
    stats_dict = {
        'n_points': len(y_data),
        'mean_x': np.mean(x_data),
        'mean_y': np.mean(y_data),
        'std_x': np.std(x_data),
        'std_y': np.std(y_data),
        'min_x': np.min(x_data),
        'max_x': np.max(x_data),
        'min_y': np.min(y_data),
        'max_y': np.max(y_data),
        'median_x': np.median(x_data),
        'median_y': np.median(y_data)
    }
    
    # 如果有足够的数据点，计算相关系数
    if len(x_data) > 1:
        stats_dict['correlation'] = np.corrcoef(x_data, y_data)[0, 1]
    
    return stats_dict

# 生成拟合曲线的坐标点（用于绘图）
def generate_curve_points(x_data: np.ndarray, func_type: str, params: np.ndarray, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """生成拟合曲线的坐标点"""
    func_info = get_function_info(func_type)
    func = func_info['function']
    
    # 生成更密集的x坐标
    x_min, x_max = np.min(x_data), np.max(x_data)
    # 扩展范围10%，使曲线更完整
    x_range = x_max - x_min
    x_curve = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, num_points)
    
    # 计算对应的y值
    y_curve = func(x_curve, *params)
    
    return x_curve, y_curve

# 验证数据有效性
def validate_data(x_data: List[str], y_data: List[str]) -> Tuple[bool, str, np.ndarray, np.ndarray]:
    """验证并转换输入的数据"""
    try:
        # 转换为浮点数
        x_array = np.array([float(x.strip()) for x in x_data if x.strip()], dtype=float)
        y_array = np.array([float(y.strip()) for y in y_data if y.strip()], dtype=float)
        
        # 检查数据长度是否一致
        if len(x_array) != len(y_array):
            return False, "x和y数据点数量不匹配", np.array([]), np.array([])
        
        # 检查数据点数量
        if len(x_array) < 2:
            return False, "数据点数量过少，至少需要2个数据点", np.array([]), np.array([])
        
        # 检查数据是否有效（非无穷大，非NaN）
        if not (np.isfinite(x_array).all() and np.isfinite(y_array).all()):
            return False, "数据包含无效值（NaN或无穷大）", np.array([]), np.array([])
        
        return True, "", x_array, y_array
        
    except ValueError as e:
        return False, f"数据格式错误：{str(e)}", np.array([]), np.array([])
    except Exception as e:
        return False, f"数据处理错误：{str(e)}", np.array([]), np.array([])

# 主拟合函数
def fit_data(x_data: List[str], y_data: List[str], func_type: str, filter_outliers_flag: bool = False, outlier_threshold: float = 2.0) -> Dict[str, Any]:
    """整合数据验证、异常值过滤和曲线拟合的主函数
    
    Args:
        x_data: x坐标数据列表
        y_data: y坐标数据列表
        func_type: 拟合函数类型
        filter_outliers_flag: 是否过滤异常值
        outlier_threshold: 异常值过滤阈值
    
    Returns:
        包含拟合结果的字典
    """
    # 验证数据
    is_valid, error_msg, x_array, y_array = validate_data(x_data, y_data)
    
    if not is_valid:
        return {
            'success': False,
            'error': error_msg
        }
    
    # 过滤异常值（如果需要）
    if filter_outliers_flag:
        original_len = len(x_array)
        x_array, y_array, _ = filter_outliers(x_array, y_array, outlier_threshold)
        filtered_count = original_len - len(x_array)
    else:
        filtered_count = 0
    
    # 执行拟合
    result = perform_fitting(x_array, y_array, func_type)
    
    # 添加原始数据信息
    result['original_data'] = {
        'x': x_array,
        'y': y_array
    }
    
    # 添加过滤信息
    result['filtered_count'] = filtered_count
    
    # 计算统计数据
    result['statistics'] = calculate_statistics(x_array, y_array)
    
    return result