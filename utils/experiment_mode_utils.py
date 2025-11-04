'''实验模式工具,与数学模式不完全相同，主要是绘制尽可能多的穿过数据点的光滑曲线
需要过滤超出设定阈值的数据异常点
如果阈值为0，则不进行过滤
实验模式会对输入的数据进行分析，主要是对绘制出的曲线进行合理性分析，判断
根据已知的数据点判断在合理的误差允许范围内曲线是否具有代表性
显示实验数据处理时的统计学数据，用于辅助计算不确定度和误差值
'''

import numpy as np
from scipy import interpolate
from typing import Tuple, List, Dict, Any
from .fitting_functions import filter_outliers, calculate_statistics, generate_curve_points, perform_fitting

def experiment_mode_analysis(x_data: np.ndarray, y_data: np.ndarray, enable_outlier_filter: bool = False, 
                           outlier_threshold: float = 3.0, fit_method: str = '多项式拟合', 
                           enable_iterative_filter: bool = False, iteration_count: int = 3, 
                           iteration_threshold: float = 0.1) -> Dict[str, Any]:
    """实验模式的数据处理和分析主函数，增强版，支持多种拟合方法"""
    results = {
        'original_data': (x_data.copy(), y_data.copy()),
        'filtered_data': (x_data.copy(), y_data.copy()),  # 默认使用原始数据
        'filtered_indices': [],
        'threshold_used': 0.0,
        'filtered_stats': None,
        'best_poly_fit': None,
        'smooth_curve': None,
        'curve_quality': None
    }
    
    # 数据验证
    if len(x_data) != len(y_data) or len(x_data) < 2:
        results['curve_quality'] = {
            'goodness_of_fit': '未知',
            'error_analysis': {},
            'data_representativeness': '未知',
            'recommendations': ['数据点数量不足或X-Y数据长度不匹配'],
            'uncertainty_estimates': {},
            'residual_analysis': {},
            'confidence_measures': {}
        }
        return results
    
    # 过滤异常点（如果需要）
    if enable_outlier_filter and outlier_threshold > 0:
        filtered_x, filtered_y, filtered_indices = filter_outliers(x_data, y_data, threshold=outlier_threshold)
        results['filtered_data'] = (filtered_x, filtered_y)
        results['filtered_indices'] = filtered_indices
        results['threshold_used'] = outlier_threshold
    else:
        # 不进行过滤，使用原始数据
        results['filtered_data'] = (x_data.copy(), y_data.copy())
        results['filtered_indices'] = []
        results['threshold_used'] = 0.0
    
    # 执行迭代过滤
    if enable_iterative_filter and iteration_count > 0 and iteration_threshold > 0:
        # 记录迭代过滤的历史
        iteration_history = []
        current_x, current_y = results['filtered_data']
        current_indices = np.arange(len(current_x))
        original_total_indices = len(x_data)
        
        for iteration in range(iteration_count):
            # 执行拟合以获取曲线
            temp_results = {}
            if fit_method == '多项式拟合':
                # 找到最佳多项式拟合
                poly_fit = find_best_polynomial_fit(current_x, current_y)
                temp_results['best_poly_fit'] = poly_fit
                
                if poly_fit:
                    # 计算当前数据点在拟合曲线上的预测值
                    y_pred = np.polyval(poly_fit['coeffs'], current_x)
                else:
                    break  # 如果拟合失败，停止迭代
            else:  # 平滑样条拟合
                try:
                    # 对x数据进行排序以确保样条插值的正确性
                    sorted_indices = np.argsort(current_x)
                    sorted_x = current_x[sorted_indices]
                    sorted_y = current_y[sorted_indices]
                    
                    # 使用合适的插值方法
                    if len(sorted_x) >= 4:
                        spl = interpolate.CubicSpline(sorted_x, sorted_y)
                    else:
                        spl = interpolate.interp1d(sorted_x, sorted_y, kind='linear')
                    
                    # 计算当前数据点在拟合曲线上的预测值
                    y_pred = spl(current_x)
                except Exception:
                    break  # 如果拟合失败，停止迭代
            
            # 计算残差
            residuals = np.abs(current_y - y_pred)
            
            # 计算相对误差阈值（如果指定了相对误差）
            if iteration_threshold < 1.0:  # 假设小于1.0表示相对误差
                # 使用当前y值的范围或均值作为基准
                y_range = np.max(current_y) - np.min(current_y)
                if y_range > 0:
                    absolute_threshold = iteration_threshold * y_range
                else:
                    absolute_threshold = iteration_threshold * np.mean(current_y) if np.mean(current_y) > 0 else iteration_threshold
            else:
                absolute_threshold = iteration_threshold  # 绝对值阈值
            
            # 过滤超出阈值的点
            mask = residuals <= absolute_threshold
            
            # 如果没有过滤掉任何点，停止迭代
            if np.all(mask):
                break
            
            # 更新数据和索引
            iteration_history.append({
                'iteration': iteration + 1,
                'removed_count': len(current_x) - np.sum(mask),
                'remaining_count': np.sum(mask),
                'threshold_used': absolute_threshold
            })
            
            # 更新当前数据
            current_x = current_x[mask]
            current_y = current_y[mask]
            current_indices = current_indices[mask]
            
            # 如果数据点少于5个，停止迭代
            if len(current_x) < 5:
                break
        
        # 确定原始数据中的索引（如果需要）
        final_filtered_indices = []
        if len(iteration_history) > 0:
            # 计算被过滤掉的索引
            all_indices = np.arange(original_total_indices)
            if results['filtered_indices']:  # 如果之前有过滤
                # 获取未被初始过滤的索引
                initial_kept_indices = [i for i in all_indices if i not in results['filtered_indices']]
                # 在初始保留的索引中找到被迭代过滤掉的
                kept_in_iteration = [initial_kept_indices[i] for i in current_indices]
                # 所有被过滤的索引 = 初始过滤的 + 迭代过滤的
                final_filtered_indices = results['filtered_indices'] + [i for i in initial_kept_indices if i not in kept_in_iteration]
            else:
                # 直接计算被迭代过滤掉的索引
                kept_in_iteration = [all_indices[i] for i in current_indices]
                final_filtered_indices = [i for i in all_indices if i not in kept_in_iteration]
            
            # 更新结果
            results['filtered_data'] = (current_x, current_y)
            results['filtered_indices'] = final_filtered_indices
            results['iteration_history'] = iteration_history
    
    # 获取过滤后的数据
    filtered_x, filtered_y = results['filtered_data']
    
    # 如果过滤后数据点太少，返回错误结果
    if len(filtered_x) < 2:
        results['curve_quality'] = {
            'goodness_of_fit': '未知',
            'error_analysis': {},
            'data_representativeness': '未知',
            'recommendations': ['过滤后数据点太少，无法进行有效拟合'],
            'uncertainty_estimates': {},
            'residual_analysis': {},
            'confidence_measures': {}
        }
        return results
    
    # 计算过滤后数据的增强统计信息
    # 基础统计量
    mean_x = np.mean(filtered_x)
    std_x = np.std(filtered_x)
    mean_y = np.mean(filtered_y)
    std_y = np.std(filtered_y)
    
    # 计算中位数
    median_x = np.median(filtered_x)
    median_y = np.median(filtered_y)
    
    # 计算四分位数
    q1_x = np.percentile(filtered_x, 25)
    q3_x = np.percentile(filtered_x, 75)
    q1_y = np.percentile(filtered_y, 25)
    q3_y = np.percentile(filtered_y, 75)
    
    # 计算四分位距
    iqr_x = q3_x - q1_x
    iqr_y = q3_y - q1_y
    
    # 计算变异系数
    cv_x = (std_x / mean_x * 100) if mean_x != 0 else 0
    cv_y = (std_y / mean_y * 100) if mean_y != 0 else 0
    
    # 计算相关系数
    correlation = np.corrcoef(filtered_x, filtered_y)[0, 1] if len(filtered_x) > 1 else 0
    
    # 组合增强的统计信息
    results['filtered_stats'] = {
        'n_points': len(filtered_x),
        'min_x': np.min(filtered_x),
        'max_x': np.max(filtered_x),
        'mean_x': mean_x,
        'median_x': median_x,
        'std_x': std_x,
        'q1_x': q1_x,
        'q3_x': q3_x,
        'iqr_x': iqr_x,
        'cv_x': cv_x,
        'min_y': np.min(filtered_y),
        'max_y': np.max(filtered_y),
        'mean_y': mean_y,
        'median_y': median_y,
        'std_y': std_y,
        'q1_y': q1_y,
        'q3_y': q3_y,
        'iqr_y': iqr_y,
        'cv_y': cv_y,
        'correlation': correlation
    }
    
    # 根据选择的拟合方法执行拟合
    if fit_method == '多项式拟合':
        # 找到最佳多项式拟合
        results['best_poly_fit'] = find_best_polynomial_fit(filtered_x, filtered_y)
        
        # 评估拟合质量
        if results['best_poly_fit']:
            results['curve_quality'] = evaluate_curve_quality(filtered_x, filtered_y, results['best_poly_fit'])
        else:
            # 如果多项式拟合失败，设置默认质量
            results['curve_quality'] = {
                'goodness_of_fit': '较差',
                'error_analysis': {},
                'data_representativeness': '一般',
                'recommendations': ['多项式拟合失败，数据可能不适合多项式模型'],
                'uncertainty_estimates': {},
                'residual_analysis': {},
                'confidence_measures': {}
            }
    else:  # 平滑样条拟合
        try:
            # 对x数据进行排序以确保样条插值的正确性
            sorted_indices = np.argsort(filtered_x)
            sorted_x = filtered_x[sorted_indices]
            sorted_y = filtered_y[sorted_indices]
            
            # 使用不同的插值方法，根据数据量选择合适的方法
            if len(sorted_x) >= 4:
                # 对于足够的数据点，使用三次样条插值
                spl = interpolate.CubicSpline(sorted_x, sorted_y)
            else:
                # 对于较少的数据点，使用线性插值
                spl = interpolate.interp1d(sorted_x, sorted_y, kind='linear')
            
            # 生成平滑曲线的数据点
            x_min, x_max = min(sorted_x), max(sorted_x)
            # 扩展范围以更好地显示曲线
            x_min_extended = x_min - 0.1 * (x_max - x_min)
            x_max_extended = x_max + 0.1 * (x_max - x_min)
            x_smooth = np.linspace(x_min_extended, x_max_extended, 1000)
            
            # 计算插值后的y值
            y_smooth = spl(x_smooth)
            
            # 存储平滑曲线
            results['smooth_curve'] = (x_smooth, y_smooth)
            
            # 计算平滑曲线的拟合质量
            # 由于样条曲线是通过所有数据点的，计算插值时的误差
            residuals = []
            # 计算原始数据点上的拟合质量
            for x, y in zip(filtered_x, filtered_y):
                # 找到x在排序后的数组中的位置，计算对应的y值
                if x >= sorted_x[0] and x <= sorted_x[-1]:
                    y_fit = spl(x)
                    residuals.append(y - y_fit)
            
            residuals = np.array(residuals)
            mse = np.mean(residuals**2) if len(residuals) > 0 else 0
            rmse = np.sqrt(mse)
            r_squared = 1.0  # 样条插值在数据点上应该完全匹配
            
            # 创建一个模拟的最佳拟合结果用于质量评估
            mock_best_fit = {
                'degree': '样条',
                'coeffs': [],
                'mse': mse,
                'rmse': rmse,
                'r_squared': r_squared
            }
            
            # 评估拟合质量
            results['curve_quality'] = evaluate_curve_quality(filtered_x, filtered_y, mock_best_fit)
        except Exception as e:
            # 如果样条拟合失败，设置默认质量
            results['curve_quality'] = {
                'goodness_of_fit': '较差',
                'error_analysis': {},
                'data_representativeness': '一般',
                'recommendations': [f'平滑样条拟合失败: {str(e)}'],
                'uncertainty_estimates': {},
                'residual_analysis': {},
                'confidence_measures': {}
            }
    
    # 确保curve_quality始终存在
    if not results['curve_quality']:
        results['curve_quality'] = {
            'goodness_of_fit': '未知',
            'error_analysis': {},
            'data_representativeness': '未知',
            'recommendations': ['拟合过程中出现未知错误'],
            'uncertainty_estimates': {},
            'residual_analysis': {},
            'confidence_measures': {}
        }
    
    return results

def find_best_polynomial_fit(x_data: np.ndarray, y_data: np.ndarray, max_degree: int = 7) -> Dict[str, Any]:
    """基于AIC准则选择最优多项式阶数，增强版"""
    best_fit = None
    best_aic = float('inf')
    all_fits = []  # 存储所有尝试的拟合结果，用于比较
    
    # 计算不同阶数的多项式拟合
    for degree in range(1, max_degree + 1):
        try:
            # 拟合多项式，使用更多参数以提高数值稳定性
            coeffs = np.polyfit(x_data, y_data, degree, full=False)
            
            # 计算拟合值和残差
            poly_func = np.poly1d(coeffs)
            y_fit = poly_func(x_data)
            residuals = y_data - y_fit
            
            # 计算误差指标
            n = len(x_data)
            k = degree + 1  # 参数数量
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(residuals))
            
            # 计算R²
            ss_total = np.sum((y_data - np.mean(y_data))**2)
            ss_residual = np.sum(residuals**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
            
            # 计算调整后的R²
            adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1)) if n > k + 1 else 0
            
            # 计算AIC
            aic = n * np.log(mse) + 2 * k
            
            # 计算BIC (贝叶斯信息准则)
            bic = n * np.log(mse) + k * np.log(n)
            
            # 存储当前阶数的拟合结果
            fit_result = {
                'degree': degree,
                'coeffs': coeffs,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r_squared': r_squared,
                'adjusted_r_squared': adj_r_squared,
                'aic': aic,
                'bic': bic
            }
            
            all_fits.append(fit_result)
            
            # 如果AIC更小，则更新最佳拟合
            if aic < best_aic:
                best_aic = aic
                best_fit = fit_result
                
        except np.linalg.LinAlgError:
            # 如果拟合失败（可能由于数值问题），跳过该阶数
            continue
    
    # 如果没有找到合适的拟合，返回一个简单的线性拟合
    if best_fit is None and len(x_data) >= 2:
        try:
            coeffs = np.polyfit(x_data, y_data, 1)
            poly_func = np.poly1d(coeffs)
            y_fit = poly_func(x_data)
            residuals = y_data - y_fit
            n = len(x_data)
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            ss_total = np.sum((y_data - np.mean(y_data))**2)
            ss_residual = np.sum(residuals**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
            
            best_fit = {
                'degree': 1,
                'coeffs': coeffs,
                'mse': mse,
                'rmse': rmse,
                'mae': np.mean(np.abs(residuals)),
                'r_squared': r_squared,
                'adjusted_r_squared': 1 - ((1 - r_squared) * (n - 1) / (n - 2)) if n > 2 else 0,
                'aic': n * np.log(mse) + 2,
                'bic': n * np.log(mse) + np.log(n)
            }
        except:
            # 最后的备选方案
            best_fit = {
                'degree': 1,
                'coeffs': np.array([0, np.mean(y_data)]),  # 水平线
                'mse': np.var(y_data),
                'rmse': np.std(y_data),
                'mae': np.mean(np.abs(y_data - np.mean(y_data))),
                'r_squared': 0,
                'adjusted_r_squared': 0,
                'aic': float('inf'),
                'bic': float('inf')
            }
    
    # 如果有多个拟合结果，考虑过拟合风险
    if len(all_fits) > 1 and best_fit:
        # 检查最佳拟合是否为最高阶数，如果是，考虑是否真的必要
        if best_fit['degree'] == max_degree:
            # 比较与次高阶的性能差异
            for fit in all_fits:
                if fit['degree'] == max_degree - 1:
                    # 如果次高阶的AIC与最佳拟合相差不大，但阶数更低
                    if (best_fit['aic'] - fit['aic']) < 2:  # AIC差值小于2，认为没有显著差异
                        # 同时检查R²的差异
                        if (best_fit['r_squared'] - fit['r_squared']) < 0.05:  # R²提升不明显
                            # 选择较低阶数的拟合以避免过拟合
                            best_fit = fit
                        break
        
        # 找出AIC值与最佳拟合相差小于2的所有模型
        similar_fits = [fit for fit in all_fits if abs(fit['aic'] - best_fit['aic']) < 2]
        
        # 如果有多个AIC相近的模型，选择阶数最低的
        if len(similar_fits) > 1:
            # 按阶数排序
            similar_fits.sort(key=lambda x: x['degree'])
            # 选择阶数最低的模型
            best_fit = similar_fits[0]
    
    return best_fit

def evaluate_curve_quality(x_data: np.ndarray, y_data: np.ndarray, poly_fit: Dict[str, Any]) -> Dict[str, Any]:
    """评估拟合曲线的质量和合理性，增强版，基于残差正态分布分析"""
    quality = {
        'goodness_of_fit': '未知',
        'error_analysis': {},
        'data_representativeness': '未知',
        'recommendations': [],
        'uncertainty_estimates': {},  # 新增：不确定度估计
        'residual_analysis': {},      # 新增：残差分析
        'confidence_measures': {},     # 新增：置信度指标
        'normality_analysis': {}       # 新增：正态分布分析
    }
    
    if poly_fit is None or len(x_data) < 2:
        quality['recommendations'].append('数据点不足，无法进行有效评估')
        return quality
    
    # 计算拟合值和残差（delta = y_data - y0，其中y0是拟合曲线上的y值）
    poly_func = np.poly1d(poly_fit['coeffs'])
    y0 = poly_func(x_data)  # 由输入的x值计算得曲线上的y值
    delta = y_data - y0     # 输入的y值与y0的差值
    
    # 计算误差统计
    abs_delta = np.abs(delta)
    mean_y = np.mean(y_data)
    
    # 基础误差分析
    error_analysis = {
        'mean_absolute_error': np.mean(abs_delta),
        'max_absolute_error': np.max(abs_delta),
        'std_error': np.std(delta),
        'percent_points_within_1std': np.mean(abs_delta <= np.std(delta)) * 100,
        'percent_points_within_2std': np.mean(abs_delta <= 2 * np.std(delta)) * 100
    }
    
    # 新增：相对误差计算
    if np.mean(np.abs(y_data)) > 0:
        error_analysis['mean_relative_error'] = np.mean(abs_delta / np.abs(y_data)) * 100
        error_analysis['max_relative_error'] = np.max(abs_delta / np.abs(y_data)) * 100
    
    quality['error_analysis'] = error_analysis
    
    # 新增：残差分析（基于delta）
    residual_analysis = {
        'residual_mean': np.mean(delta),
        'residual_std': np.std(delta),
        'residual_skewness': np.mean(delta**3) / (np.std(delta)**3) if np.std(delta) > 0 else 0,
        'residual_kurtosis': np.mean(delta**4) / (np.std(delta)**4) - 3 if np.std(delta) > 0 else 0,
        'residual_min': np.min(delta),
        'residual_max': np.max(delta),
        'residual_range': np.max(delta) - np.min(delta),
        'abs_residual_mean': np.mean(abs_delta)  # 平均绝对残差
    }
    
    # 残差分布特征与解释
    if abs(residual_analysis['residual_skewness']) < 0.5:
        residual_analysis['distribution_shape'] = '近似对称'
        residual_analysis['shape_interpretation'] = '残差分布接近正态分布，拟合模型假设较为合理'
    elif residual_analysis['residual_skewness'] > 0:
        residual_analysis['distribution_shape'] = '右偏'
        residual_analysis['shape_interpretation'] = '存在较多正残差，模型可能低估了实际值'
    else:
        residual_analysis['distribution_shape'] = '左偏'
        residual_analysis['shape_interpretation'] = '存在较多负残差，模型可能高估了实际值'
    
    # 峰度解释
    if residual_analysis['residual_kurtosis'] > 1:
        residual_analysis['kurtosis_interpretation'] = '残差分布陡峭，存在离群值'
    elif residual_analysis['residual_kurtosis'] < -1:
        residual_analysis['kurtosis_interpretation'] = '残差分布平坦，数据变异性大'
    else:
        residual_analysis['kurtosis_interpretation'] = '残差分布正常'
    
    quality['residual_analysis'] = residual_analysis
    
    # 新增：残差正态分布分析（核心合理性判断标准）
    normality_analysis = {
        'jarque_bera_stat': None,
        'jb_p_value': None,
        'ks_stat': None,
        'ks_p_value': None,
        'normal_qq_correlation': None,
        'normality_assessment': '未知',
        'normality_interpretation': ''
    }
    
    # 计算正态分布检验统计量
    n = len(delta)
    
    # Jarque-Bera正态性检验
    if n >= 20:  # JB检验在小样本时不准确
        try:
            from scipy import stats
            # 计算JB统计量和p值
            skewness = residual_analysis['residual_skewness']
            kurtosis = residual_analysis['residual_kurtosis']
            jb_stat = (n / 6) * (skewness**2 + (kurtosis**2) / 4)
            # 自由度为2的卡方分布
            jb_p_value = 1 - stats.chi2.cdf(jb_stat, df=2)
            normality_analysis['jarque_bera_stat'] = jb_stat
            normality_analysis['jb_p_value'] = jb_p_value
        except:
            pass
    
    # Kolmogorov-Smirnov检验（与正态分布比较）
    try:
        from scipy import stats
        # 标准化残差
        if np.std(delta) > 0:
            z_scores = (delta - np.mean(delta)) / np.std(delta)
            ks_stat, ks_p_value = stats.kstest(z_scores, 'norm')
            normality_analysis['ks_stat'] = ks_stat
            normality_analysis['ks_p_value'] = ks_p_value
    except:
        pass
    
    # Q-Q图相关性检验（简单实现）
    try:
        # 计算理论分位数和样本分位数
        sorted_delta = np.sort(delta)
        n = len(sorted_delta)
        if n > 1:
            # 计算理论正态分位数
            theoretical_quantiles = np.arange(1, n + 1) / (n + 1)
            theoretical_norm = stats.norm.ppf(theoretical_quantiles, loc=np.mean(delta), scale=np.std(delta))
            # 计算Q-Q图上点的相关性
            qq_corr = np.corrcoef(sorted_delta, theoretical_norm)[0, 1]
            normality_analysis['normal_qq_correlation'] = qq_corr
    except:
        pass
    
    # 基于正态分布分析评估拟合合理性
    normality_assessment = '良好'
    normality_interpretations = []
    
    # 使用多种指标综合评估
    # 1. 偏度和峰度
    if abs(residual_analysis['residual_skewness']) > 0.5 or abs(residual_analysis['residual_kurtosis']) > 1:
        normality_assessment = '一般'
        normality_interpretations.append('残差分布的偏度或峰度偏离正态分布特征')
    
    # 2. Q-Q图相关性
    if 'normal_qq_correlation' in normality_analysis and normality_analysis['normal_qq_correlation'] is not None:
        qq_corr = normality_analysis['normal_qq_correlation']
        if qq_corr < 0.95:
            normality_assessment = '较差'
            normality_interpretations.append(f'Q-Q图相关系数为{qq_corr:.3f}，残差分布与正态分布有明显差异')
    
    # 3. 正态性检验p值
    p_value_significant = False
    if 'jb_p_value' in normality_analysis and normality_analysis['jb_p_value'] is not None:
        if normality_analysis['jb_p_value'] < 0.05:
            p_value_significant = True
            normality_assessment = '较差'
            normality_interpretations.append('Jarque-Bera检验表明残差显著偏离正态分布')
    
    if 'ks_p_value' in normality_analysis and normality_analysis['ks_p_value'] is not None:
        if normality_analysis['ks_p_value'] < 0.05:
            p_value_significant = True
            normality_assessment = '较差'
            normality_interpretations.append('Kolmogorov-Smirnov检验表明残差显著偏离正态分布')
    
    # 4. 数据点在标准差范围内的比例（经验法则）
    if error_analysis['percent_points_within_1std'] < 68 or error_analysis['percent_points_within_2std'] < 95:
        if normality_assessment == '良好':
            normality_assessment = '一般'
        elif normality_assessment == '一般':
            normality_assessment = '较差'
        normality_interpretations.append('残差分布不符合正态分布的经验法则')
    
    # 设置最终评估和解释
    normality_analysis['normality_assessment'] = normality_assessment
    
    if not normality_interpretations:
        normality_interpretations.append('残差分布符合正态分布特征，拟合模型合理')
    
    normality_analysis['normality_interpretation'] = '；'.join(normality_interpretations)
    quality['normality_analysis'] = normality_analysis
    
    # 基于残差正态分布分析重新定义拟合优度评估
    r_squared = poly_fit['r_squared']
    
    # 综合考虑R²和残差正态性
    if normality_assessment == '良好' and r_squared >= 0.9:
        quality['goodness_of_fit'] = '优秀'
        quality['goodness_of_fit_interpretation'] = '残差分布符合正态性假设，且模型解释了90%以上的数据变异性，拟合效果理想'
    elif normality_assessment == '良好' and r_squared >= 0.75:
        quality['goodness_of_fit'] = '良好'
        quality['goodness_of_fit_interpretation'] = '残差分布符合正态性假设，模型解释了75%以上的数据变异性，拟合效果较好'
    elif normality_assessment == '一般' and r_squared >= 0.6:
        quality['goodness_of_fit'] = '一般'
        quality['goodness_of_fit_interpretation'] = '残差分布基本符合正态性假设，模型解释了60%以上的数据变异性，拟合效果可接受'
    elif r_squared >= 0.5:
        quality['goodness_of_fit'] = '较差'
        quality['goodness_of_fit_interpretation'] = '残差分布偏离正态性假设或模型解释力不足，拟合效果较差'
    else:
        quality['goodness_of_fit'] = '很差'
        quality['goodness_of_fit_interpretation'] = '残差分布严重偏离正态性假设且模型解释力极低，拟合效果很差'
    
    # 新增：调整后的R²
    k = poly_fit['degree'] + 1  # 参数数量
    if n > k + 1 and r_squared > 0:
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        quality['confidence_measures']['adjusted_r_squared'] = adj_r_squared
        # 添加调整后R²的解释
        if adj_r_squared > 0.9:
            quality['confidence_measures']['adjusted_r_squared_interpretation'] = '即使考虑模型复杂度，拟合效果仍然优秀'
        elif adj_r_squared < r_squared - 0.1:
            quality['confidence_measures']['adjusted_r_squared_interpretation'] = '模型可能过于复杂，存在过拟合风险'
    
    # 评估数据代表性 - 基于残差分析
    std_delta = np.std(delta)
    mean_abs_delta = np.mean(abs_delta)
    
    # 使用残差统计量评估数据代表性
    if normality_assessment == '良好' and mean_abs_delta < 0.5 * std_delta and r_squared >= 0.85:
        quality['data_representativeness'] = '非常好'
        quality['data_representativeness_interpretation'] = '残差分布正态，且平均绝对残差较小，模型对数据的代表性极佳'
    elif (normality_assessment == '良好' or normality_assessment == '一般') and r_squared >= 0.7:
        quality['data_representativeness'] = '良好'
        quality['data_representativeness_interpretation'] = '残差分布基本符合正态，模型对数据的代表性较好'
    elif r_squared >= 0.5:
        quality['data_representativeness'] = '一般'
        quality['data_representativeness_interpretation'] = '残差分布或模型解释力存在不足，模型对数据的代表性一般'
    else:
        quality['data_representativeness'] = '较差'
        quality['data_representativeness_interpretation'] = '残差分布严重偏离正态或模型解释力低，模型对数据的代表性较差'
    
    # 新增：不确定度估计与解释
    std_uncertainty = std_delta / np.sqrt(n) if n > 1 else 0
    quality['uncertainty_estimates'] = {
        'standard_uncertainty': std_uncertainty,
        'expanded_uncertainty_95': 1.96 * std_uncertainty,  # 95%置信水平
        'relative_uncertainty': (std_uncertainty / mean_y * 100) if mean_y != 0 else 0
    }
    
    # 不确定度等级与解释
    rel_uncertainty = quality['uncertainty_estimates']['relative_uncertainty']
    if rel_uncertainty < 5:
        quality['uncertainty_estimates']['uncertainty_level'] = '低'
        quality['uncertainty_estimates']['uncertainty_interpretation'] = '测量结果非常可靠'
    elif rel_uncertainty < 10:
        quality['uncertainty_estimates']['uncertainty_level'] = '中低'
        quality['uncertainty_estimates']['uncertainty_interpretation'] = '测量结果较为可靠'
    elif rel_uncertainty < 20:
        quality['uncertainty_estimates']['uncertainty_level'] = '中'
        quality['uncertainty_estimates']['uncertainty_interpretation'] = '测量结果有一定可靠性，但需谨慎使用'
    else:
        quality['uncertainty_estimates']['uncertainty_level'] = '高'
        quality['uncertainty_estimates']['uncertainty_interpretation'] = '测量结果可靠性低，建议改进实验'
    
    # 增强版建议生成逻辑 - 更具针对性和实用性
    recommendations = []
    detailed_suggestions = []
    
    # 1. 基于残差正态分布的建议
    if normality_assessment == '较差':
        recommendations.append('残差分布显著偏离正态，拟合模型假设可能不成立')
        detailed_suggestions.append('建议尝试数据变换(如对数、平方根变换)，或考虑使用稳健回归方法')
    
    # 2. 模型选择建议
    if poly_fit['degree'] > 5 and r_squared < 0.9:
        recommendations.append(f'多项式阶数({poly_fit["degree"]})较高但拟合效果一般，可能存在过拟合风险')
        detailed_suggestions.append('建议尝试使用3-4阶多项式，或考虑平滑样条拟合以获得更稳健的结果')
    elif poly_fit['degree'] == 1 and r_squared < 0.7:
        recommendations.append('线性拟合效果不佳，数据可能存在非线性关系')
        detailed_suggestions.append('建议尝试2-3阶多项式或指数/对数等非线性模型')
    elif r_squared < 0.5:
        recommendations.append('拟合效果很差，模型选择可能不合适')
        detailed_suggestions.append('建议重新考虑数据的物理/数学模型，或检查数据采集过程')
    
    # 3. 数据质量分析与建议
    if error_analysis['percent_points_within_1std'] < 70:
        recommendations.append('误差分布较广，数据质量存在问题')
        detailed_suggestions.append('建议检查实验环境稳定性，或考虑使用鲁棒拟合方法(如RANSAC)')
    elif error_analysis['percent_points_within_2std'] < 95:
        recommendations.append('存在较多误差较大的数据点')
        detailed_suggestions.append('建议使用箱线图方法识别异常值，或采用局部加权回归方法')
    
    # 4. 数据量与分布建议
    if len(x_data) < 10:
        recommendations.append(f'数据点数量较少({len(x_data)}个)，统计可靠性不足')
        detailed_suggestions.append('建议至少增加到15-20个数据点，确保覆盖整个变量范围')
    elif len(x_data) < 5 and poly_fit['degree'] > 2:
        recommendations.append('数据点过少而模型过于复杂')
        detailed_suggestions.append('建议增加3倍于多项式阶数的数据点，或降低模型复杂度')
    
    # 5. 系统性误差检测
    if abs(residual_analysis['residual_mean']) > 0.5 * residual_analysis['residual_std']:
        recommendations.append('残差均值明显偏离零，存在系统性误差')
        detailed_suggestions.append('建议检查测量仪器校准状态，或考虑模型中加入常数项')
    elif abs(residual_analysis['residual_skewness']) > 1:
        recommendations.append('残差分布明显偏斜，模型假设可能不成立')
        detailed_suggestions.append('建议尝试数据变换(如对数变换)或使用非参数回归方法')
    
    # 6. 实验改进建议
    if rel_uncertainty > 20:
        recommendations.append('相对不确定度较大，实验可靠性低')
        detailed_suggestions.append('建议改进测量方法，增加重复测量次数，控制环境变量')
    elif rel_uncertainty > 10:
        recommendations.append('相对不确定度中等，实验可靠性一般')
        detailed_suggestions.append('建议在关键数据点增加重复测量，提高仪器精度')
    
    # 7. 数据处理优化建议
    if quality['data_representativeness'] in ['一般', '较差']:
        recommendations.append('模型对数据的代表性不足')
        detailed_suggestions.append('建议考虑分段拟合，或使用自适应拟合方法')
    
    # 8. 特殊情况详细分析
    if 'mean_relative_error' in error_analysis and error_analysis['mean_relative_error'] > 30:
        recommendations.append('平均相对误差过大，预测精度低')
        detailed_suggestions.append('建议重新检查数据采集过程，或考虑使用更适合的数学模型')
    
    # 9. 数据分布均匀性建议
    x_range = max(x_data) - min(x_data)
    x_spacing = np.diff(np.sort(x_data))
    if x_range > 0 and np.max(x_spacing) > 3 * np.mean(x_spacing):
        recommendations.append('数据点分布不均匀，可能影响拟合质量')
        detailed_suggestions.append('建议在数据密集区域适当减少点，稀疏区域增加点，使数据分布更均匀')
    
    # 10. 残差自相关性检查（简单实现）
    if len(delta) > 5:
        # 计算一阶自相关
        lag1_residuals = delta[:-1]
        lag1_next = delta[1:]
        corr_coef = np.corrcoef(lag1_residuals, lag1_next)[0, 1]
        if abs(corr_coef) > 0.5:
            recommendations.append('残差存在明显自相关性')
            detailed_suggestions.append('数据可能存在时间序列特性，建议考虑时间序列模型或调整实验顺序')
    
    # 整合建议
    quality['recommendations'] = recommendations
    quality['detailed_suggestions'] = detailed_suggestions
    
    return quality

def format_experiment_results(results: Dict[str, Any]) -> str:
    """优化版格式化实验模式的结果输出，提供更实用的数据分析和可视化信息"""
    output = []
    
    # 数据概览（优化版）- 突出关键信息
    output.append("📊 实验数据处理结果 📊")
    output.append("=" * 45)
    output.append(f"原始数据点数量: {len(results['original_data'][0])}")
    output.append(f"过滤后数据点数量: {len(results['filtered_data'][0])}")
    
    if len(results['filtered_indices']) > 0:
        output.append(f"过滤的异常点数量: {len(results['filtered_indices'])}")
        # 计算异常点占比
        outlier_percent = len(results['filtered_indices']) / len(results['original_data'][0]) * 100
        output.append(f"异常点占比: {outlier_percent:.1f}%")
        # 添加异常点评价
        if outlier_percent < 5:
            output.append(f"✅ 异常点评价: 数据质量良好，异常点较少")
        elif outlier_percent < 10:
            output.append(f"⚠️ 异常点评价: 存在少量异常点")
        elif outlier_percent < 20:
            output.append(f"⚠️ 异常点评价: 异常点比例中等，建议检查原始数据")
        else:
            output.append(f"❌ 异常点评价: 异常点比例较高，数据可靠性可能受影响")
    else:
        output.append("✅ 异常点评价: 未检测到异常点")
    
    if results['threshold_used'] > 0:
        output.append(f"使用的异常点过滤阈值: {results['threshold_used']}倍标准差")
    
    # 增强的统计信息 - 精简但保留关键指标
    output.append("\n📈 过滤后数据统计信息 📈")
    output.append("=" * 45)
    stats = results['filtered_stats']
    if stats:
        # 基础统计量
        output.append(f"数据点数量: {stats['n_points']}")
        
        # X数据统计 - 保留核心指标
        output.append("\nX轴数据统计:")
        output.append(f"  范围: [{stats['min_x']:.4f}, {stats['max_x']:.4f}]")
        output.append(f"  中位数: {stats['median_x']:.4f}")
        output.append(f"  平均值: {stats['mean_x']:.4f}")
        output.append(f"  标准差: {stats['std_x']:.4f}")
        
        # 计算X的变异系数
        cv_x = stats.get('cv_x', (stats['std_x'] / stats['mean_x'] * 100) if stats['mean_x'] != 0 else 0)
        output.append(f"  变异系数: {cv_x:.2f}%")
        # 变异程度评价已移除
        
        # Y数据统计 - 保留核心指标
        output.append("\nY轴数据统计:")
        output.append(f"  范围: [{stats['min_y']:.4f}, {stats['max_y']:.4f}]")
        output.append(f"  中位数: {stats['median_y']:.4f}")
        output.append(f"  平均值: {stats['mean_y']:.4f}")
        output.append(f"  标准差: {stats['std_y']:.4f}")
        
        # 计算Y的变异系数
        cv_y = stats.get('cv_y', (stats['std_y'] / stats['mean_y'] * 100) if stats['mean_y'] != 0 else 0)
        output.append(f"  变异系数: {cv_y:.2f}%")
        # 变异程度评价已移除
        
        # 相关性分析
        if 'correlation' in stats:
            corr = stats['correlation']
            output.append("\n🔗 相关性分析:")
            output.append(f"  X-Y相关系数: {corr:.4f}")
            
            # 相关性解释
            if abs(corr) >= 0.9:
                corr_interpretation = "强相关"
                icon = "✅"
            elif abs(corr) >= 0.7:
                corr_interpretation = "中度强相关"
                icon = "✅"
            elif abs(corr) >= 0.5:
                corr_interpretation = "中度相关"
                icon = "⚠️"
            elif abs(corr) >= 0.3:
                corr_interpretation = "弱相关"
                icon = "⚠️"
            else:
                corr_interpretation = "极弱相关或无相关"
                icon = "❌"
            
            direction = "正相关" if corr > 0 else "负相关" if corr < 0 else "无相关"
            output.append(f"  {icon} 相关性解释: {corr_interpretation} ({direction})")
            
            # 相关性强度评价
            if abs(corr) >= 0.7:
                output.append(f"  💡 相关性强度评价: 两变量关系密切，适合使用多项式拟合")
            elif abs(corr) >= 0.3:
                output.append(f"  💡 相关性强度评价: 两变量存在一定关系，拟合可能存在中等误差")
            else:
                output.append(f"  💡 相关性强度评价: 两变量关系较弱，建议考虑其他建模方法或增加数据量")
    
    # 最佳多项式拟合（优化展示）
    if results['best_poly_fit']:
        output.append("\n🔍 最佳多项式拟合结果 🔍")
        output.append("=" * 45)
        poly = results['best_poly_fit']
        output.append(f"多项式阶数: {poly['degree']}")
        
        # 格式化多项式表达式（更易读的格式）
        coeffs = poly['coeffs']
        terms = []
        for i, coef in enumerate(reversed(coeffs)):
            if abs(coef) < 1e-10:  # 跳过接近零的系数
                continue
            
            if i == 0:
                terms.append(f"{coef:+.4f}")
            elif i == 1:
                terms.append(f"{coef:+.4f}x")
            else:
                terms.append(f"{coef:+.4f}x^{i}")
        
        # 移除第一个项的+号
        if terms and terms[0].startswith('+'):
            terms[0] = terms[0][1:]
        
        polynomial_str = " ".join(terms)
        output.append(f"多项式方程: y = {polynomial_str}")
        
        # 拟合质量指标 - 突出关键指标
        output.append("\n📊 拟合质量指标:")
        output.append(f"  均方根误差(RMSE): {poly['rmse']:.6f}")
        # 增加MAE如果可用
        if 'mae' in poly:
            output.append(f"  平均绝对误差(MAE): {poly['mae']:.6f}")
        output.append(f"  决定系数(R²): {poly['r_squared']:.4f}")
        
        if 'adjusted_r_squared' in poly:
            output.append(f"  调整后的决定系数(Adj-R²): {poly['adjusted_r_squared']:.4f}")
        
        # 模型复杂度评估
        if poly['degree'] <= 2:
            output.append(f"  ✅ 模型复杂度: 低 - 模型简单，泛化能力强")
        elif poly['degree'] <= 4:
            output.append(f"  ⚠️ 模型复杂度: 中等 - 平衡拟合精度和泛化能力")
        else:
            output.append(f"  ❌ 模型复杂度: 高 - 拟合精度高但可能过拟合")
        
        # AIC和BIC信息准则 - 仅保留关键信息
        if 'aic' in poly:
            output.append(f"  AIC信息准则: {poly['aic']:.4f}")
        if 'bic' in poly:
            output.append(f"  BIC信息准则: {poly['bic']:.4f}")
        
        # 过拟合风险评估
        if 'adjusted_r_squared' in poly and 'r_squared' in poly:
            r_diff = poly['r_squared'] - poly['adjusted_r_squared']
            if r_diff > 0.1:
                output.append(f"  ❌ 过拟合风险: 高 (R²与Adj-R²差异较大)，建议降低多项式阶数")
            elif r_diff > 0.05:
                output.append(f"  ⚠️ 过拟合风险: 中等，可考虑验证集测试")
            else:
                output.append(f"  ✅ 过拟合风险: 低，模型泛化能力良好")
    
    # 平滑样条拟合结果（如果存在）- 增强展示
    if 'smooth_curve' in results and results['smooth_curve']:
        output.append("\n📈 平滑样条拟合结果 📈")
        output.append("=" * 45)
        output.append("✅ 平滑样条拟合已完成，适合展示数据趋势变化")
        
        # 计算平滑曲线与原始数据的误差
        if results['best_poly_fit']:
            # 基于多项式拟合的误差进行参考比较
            poly_mse = results['best_poly_fit']['mse']
            # 计算平滑曲线的MSE
            x_data, y_data = results['filtered_data']
            smooth_x, smooth_y = results['smooth_curve']
            # 对于每个过滤后的数据点，找到对应的平滑曲线上的y值（使用最近的x值）
            smooth_y_for_data = []
            for x in x_data:
                # 找到最接近x的smooth_x值的索引
                idx = np.argmin(np.abs(smooth_x - x))
                smooth_y_for_data.append(smooth_y[idx])
            
            # 计算平滑曲线的MSE
            smooth_mse = np.mean((np.array(y_data) - np.array(smooth_y_for_data))**2)
            output.append(f"  💡 多项式拟合MSE: {poly_mse:.6f}")
            output.append(f"  💡 平滑样条拟合MSE: {smooth_mse:.6f}")
            
            # 比较两种拟合方法
            if smooth_mse < poly_mse:
                output.append(f"  ✅ 平滑样条拟合效果更好，提供了更灵活的数据趋势表示")
            else:
                output.append(f"  ⚠️ 多项式拟合精度更高，但平滑样条可能更好地捕捉非线性趋势")
    
    # 增强的曲线质量评估
    output.append("\n🌟 曲线质量综合评估 🌟")
    output.append("=" * 45)
    quality = results['curve_quality']
    output.append(f"拟合优度等级: {quality['goodness_of_fit']}")
    output.append(f"数据代表性: {quality['data_representativeness']}")
    
    # 详细误差分析 - 精简但保留关键指标
    error_analysis = quality['error_analysis']
    if error_analysis:
        output.append("\n⚠️ 详细误差分析 ⚠️")
        output.append("=" * 45)
        output.append(f"平均绝对误差(MAE): {error_analysis['mean_absolute_error']:.6f}")
        output.append(f"最大绝对误差: {error_analysis['max_absolute_error']:.6f}")
        output.append(f"标准误差: {error_analysis['std_error']:.6f}")
        
        # 相对误差信息（如果存在）
        if 'mean_relative_error' in error_analysis:
            output.append(f"平均相对误差: {error_analysis['mean_relative_error']:.2f}%")
            output.append(f"最大相对误差: {error_analysis['max_relative_error']:.2f}%")
            
            # 相对误差评价
            mean_rel_error = error_analysis['mean_relative_error']
            if mean_rel_error < 5:
                output.append(f"✅ 相对误差评价: 优秀 - 误差在可接受范围内")
            elif mean_rel_error < 10:
                output.append(f"✅ 相对误差评价: 良好 - 误差较小")
            elif mean_rel_error < 20:
                output.append(f"⚠️ 相对误差评价: 一般 - 误差中等，可接受")
            elif mean_rel_error < 30:
                output.append(f"❌ 相对误差评价: 较差 - 误差较大，建议改进")
            else:
                output.append(f"❌ 相对误差评价: 很差 - 误差过大，需重新建模")
        
        # 误差分布统计
        output.append("\n误差分布情况:")
        output.append(f"  1倍标准差内的数据点: {error_analysis['percent_points_within_1std']:.1f}%")
        output.append(f"  2倍标准差内的数据点: {error_analysis['percent_points_within_2std']:.1f}%")
        
        # 误差分布评价
        percent_within_1std = error_analysis['percent_points_within_1std']
        percent_within_2std = error_analysis['percent_points_within_2std']
        
        if percent_within_1std >= 90:
            output.append(f"  ✅ 误差分布评价: 非常集中，拟合稳定性高")
        elif percent_within_1std >= 70:
            output.append(f"  ⚠️ 误差分布评价: 相对集中，拟合较稳定")
        elif percent_within_2std >= 90:
            output.append(f"  ⚠️ 误差分布评价: 一般，拟合稳定性一般")
        else:
            output.append(f"  ❌ 误差分布评价: 分散，拟合不稳定，建议检查异常点或增加数据量")
    
    # 残差分析 - 优化版，更简洁易读
    if 'residual_analysis' in quality and quality['residual_analysis']:
        residual_analysis = quality['residual_analysis']
        output.append("\n📊 残差分析 📊")
        output.append("=" * 45)
        output.append(f"残差均值: {residual_analysis['residual_mean']:.6f}")
        output.append(f"残差标准差: {residual_analysis['residual_std']:.6f}")
        output.append(f"残差偏度: {residual_analysis['residual_skewness']:.3f}")
        output.append(f"残差峰度: {residual_analysis['residual_kurtosis']:.3f}")
        
        if 'distribution_shape' in residual_analysis:
            output.append(f"残差分布形状: {residual_analysis['distribution_shape']}")
        
        # 残差正态性判断和建议
        if abs(residual_analysis['residual_mean']) < 1e-6:
            output.append("✅ 残差均值接近零，符合模型假设，拟合无系统性偏差")
        else:
            output.append("❌ 残差均值偏离零，存在系统性偏差，建议检查模型假设或考虑其他模型")
        
        # 残差偏度解释
        skewness = residual_analysis['residual_skewness']
        if abs(skewness) < 0.5:
            output.append(f"✅ 残差偏度适中，分布接近对称，拟合效果良好")
        elif skewness > 0:
            output.append(f"⚠️ 残差显著右偏，模型对低值估计较好，高值估计偏低")
        else:
            output.append(f"⚠️ 残差显著左偏，模型对高值估计较好，低值估计偏高")
    
    # 新增：残差正态分布分析（基于delta的合理性判断）
    if 'normality_analysis' in quality and quality['normality_analysis']:
        normality_analysis = quality['normality_analysis']
        output.append("\n🔬 残差正态分布分析 🔬")
        output.append("=" * 45)
        
        # 显示正态性评估结果（核心合理性判断）
        normality_icon = "✅" if normality_analysis['normality_assessment'] == "良好" else "⚠️" if normality_analysis['normality_assessment'] == "一般" else "❌"
        output.append(f"{normality_icon} 正态性评估等级: {normality_analysis['normality_assessment']}")
        output.append(f"📝 评估解释: {normality_analysis['normality_interpretation']}")
        
        # 显示统计检验结果（如果有）
        test_results = []
        if normality_analysis.get('jarque_bera_stat') is not None:
            jb_stat = normality_analysis['jarque_bera_stat']
            jb_p_value = normality_analysis['jb_p_value']
            jb_icon = "✅" if jb_p_value >= 0.05 else "❌"
            test_results.append(f"{jb_icon} Jarque-Bera统计量: {jb_stat:.4f}, p值: {jb_p_value:.4f}")
        
        if normality_analysis.get('ks_stat') is not None:
            ks_stat = normality_analysis['ks_stat']
            ks_p_value = normality_analysis['ks_p_value']
            ks_icon = "✅" if ks_p_value >= 0.05 else "❌"
            test_results.append(f"{ks_icon} Kolmogorov-Smirnov统计量: {ks_stat:.4f}, p值: {ks_p_value:.4f}")
        
        if normality_analysis.get('normal_qq_correlation') is not None:
            qq_corr = normality_analysis['normal_qq_correlation']
            qq_icon = "✅" if qq_corr >= 0.95 else "⚠️" if qq_corr >= 0.9 else "❌"
            test_results.append(f"{qq_icon} Q-Q图相关系数: {qq_corr:.4f}")
        
        # 如果有统计检验结果，输出它们
        if test_results:
            output.append("\n📊 正态性检验结果:")
            for result in test_results:
                output.append(f"  {result}")
        
        # 根据正态性评估提供特定建议
        if normality_analysis['normality_assessment'] == "良好":
            output.append("✅ 合理性判断: 残差分布符合正态性假设，拟合模型非常合理")
        elif normality_analysis['normality_assessment'] == "一般":
            output.append("⚠️ 合理性判断: 残差分布基本符合正态性假设，拟合模型合理但存在改进空间")
        else:
            output.append("❌ 合理性判断: 残差分布显著偏离正态性假设，拟合模型合理性较差，建议改进")
    
    # 不确定度分析 - 优化版，更简洁易读
    if 'uncertainty_estimates' in quality and quality['uncertainty_estimates']:
        uncertainty = quality['uncertainty_estimates']
        output.append("\n🔍 不确定度分析 🔍")
        output.append("=" * 45)
        output.append(f"标准不确定度: {uncertainty['standard_uncertainty']:.6f}")
        output.append(f"95%置信水平扩展不确定度: {uncertainty['expanded_uncertainty_95']:.6f}")
        
        if 'relative_uncertainty' in uncertainty:
            output.append(f"相对不确定度: {uncertainty['relative_uncertainty']:.2f}%")
            
            # 不确定度等级和解释 - 更简洁的解释
            rel_uncertainty = uncertainty['relative_uncertainty']
            if rel_uncertainty < 5:
                output.append("✅ 不确定度等级: 低")
                output.append("💡 数据处理建议: 当前数据处理方法可靠，可保持现有参数")
            elif rel_uncertainty < 10:
                output.append("✅ 不确定度等级: 中低")
                output.append("💡 数据处理建议: 可考虑轻微优化参数以进一步降低不确定度")
            elif rel_uncertainty < 20:
                output.append("⚠️ 不确定度等级: 中等")
                output.append("💡 数据处理建议: 建议重新评估数据采集方法，增加关键数据点测量次数")
            elif rel_uncertainty < 30:
                output.append("❌ 不确定度等级: 中高")
                output.append("💡 数据处理建议: 建议改进实验方法，增加样本量，重新采集数据")
            else:
                output.append("❌ 不确定度等级: 高")
                output.append("💡 数据处理建议: 必须重新设计实验，改进测量方法，收集新数据")
    
    # 置信度指标 - 优化版，更简洁易读
    if 'confidence_measures' in quality and quality['confidence_measures']:
        confidence = quality['confidence_measures']
        output.append("\n🔐 模型置信度指标 🔐")
        output.append("=" * 45)
        if 'adjusted_r_squared' in confidence:
            adj_r2 = confidence['adjusted_r_squared']
            output.append(f"调整后的决定系数(R²): {adj_r2:.4f}")
            
            # 调整R²评价 - 更简洁的评价
            if adj_r2 >= 0.95:
                output.append(f"✅ 模型解释力: 极高")
                output.append(f"💡 置信度评估: 模型非常可靠，可用于高精度预测")
            elif adj_r2 >= 0.9:
                output.append(f"✅ 模型解释力: 很高")
                output.append(f"💡 置信度评估: 模型可靠，适合大多数应用场景")
            elif adj_r2 >= 0.8:
                output.append(f"⚠️ 模型解释力: 高")
                output.append(f"💡 置信度评估: 模型较可靠，可用于常规分析")
            elif adj_r2 >= 0.7:
                output.append(f"⚠️ 模型解释力: 中等")
                output.append(f"💡 置信度评估: 模型基本可靠，结果仅供参考")
            elif adj_r2 >= 0.5:
                output.append(f"❌ 模型解释力: 一般")
                output.append(f"💡 置信度评估: 模型解释力有限，需谨慎使用")
                output.append(f"💡 改进建议: 考虑其他类型的模型或增加数据量")
            else:
                output.append(f"❌ 模型解释力: 低")
                output.append(f"💡 置信度评估: 模型解释力不足，不建议用于预测")
                output.append(f"💡 改进建议: 必须重新选择模型或考虑数据预处理方法")
    
    # 优化版实用建议 - 更具针对性和可操作性
    output.append("\n💡 数据处理优化建议 💡")
    output.append("=" * 45)
    
    # 初始化建议列表
    recommendations = []
    n_points = len(results['filtered_data'][0])
    outlier_percent = len(results['filtered_indices']) / len(results['original_data'][0]) * 100 if len(results['original_data'][0]) > 0 else 0
    
    # 样本量建议
    if n_points < 10:
        recommendations.append(f"📈 增加样本量至少至10个数据点，当前样本量({n_points})不足以支撑可靠的统计分析")
    elif n_points < 20:
        recommendations.append(f"📈 考虑增加样本量至20个以上，以提高模型的稳定性和泛化能力")
    
    # 异常点建议
    if outlier_percent > 20:
        recommendations.append(f"🔍 异常点比例过高({outlier_percent:.1f}%)，建议检查实验条件，重新采集数据或使用稳健估计方法")
    elif outlier_percent > 10:
        recommendations.append(f"🔍 存在较多异常点({outlier_percent:.1f}%)，建议验证这些数据点的可靠性")
    
    # 模型拟合建议
    if results['best_poly_fit']:
        poly = results['best_poly_fit']
        if poly['degree'] > 4:
            recommendations.append(f"🧮 当前多项式阶数({poly['degree']})较高，可能存在过拟合风险，建议尝试阶数≤3的模型")
        elif poly['r_squared'] < 0.7 and results['filtered_stats'] and results['filtered_stats'].get('correlation', 0) > 0.5:
            recommendations.append("🧮 虽然数据相关性较好，但多项式拟合效果一般，建议尝试非线性模型")
    
    # 残差分析建议
    if 'residual_analysis' in quality and quality['residual_analysis']:
        residual_analysis = quality['residual_analysis']
        if abs(residual_analysis['residual_mean']) > 1e-6:
            recommendations.append("📊 残差存在系统性偏差，建议检查数据采集过程中的系统误差")
        if abs(residual_analysis['residual_skewness']) > 1.0:
            recommendations.append("📊 残差分布明显偏斜，建议考虑数据变换或其他模型类型")
    
    # 相关性建议
    if results['filtered_stats'] and results['filtered_stats'].get('correlation', 0) and abs(results['filtered_stats']['correlation']) < 0.3:
        recommendations.append("🔗 变量相关性很弱，传统拟合方法可能不适用，建议重新考虑数据模型或实验设计")
    
    # 不确定性建议
    if 'uncertainty_estimates' in quality and quality['uncertainty_estimates'] and 'relative_uncertainty' in quality['uncertainty_estimates']:
        if quality['uncertainty_estimates']['relative_uncertainty'] > 20:
            recommendations.append("⚠️ 数据不确定性较高，建议改进测量精度或增加重复测量次数")
    
    # 添加质量评估中的建议
    if quality['recommendations']:
        for rec in quality['recommendations']:
            if rec not in recommendations:  # 避免重复
                recommendations.append(rec)
    
    # 如果没有具体建议，添加通用建议
    if not recommendations:
        recommendations.append("✅ 当前数据和模型拟合情况良好，无需特殊调整")
    
    # 添加最终建议列表 - 限制数量，避免信息过载
    max_recommendations = 5  # 限制最多显示5条建议
    for i, rec in enumerate(recommendations[:max_recommendations], 1):
        output.append(f"{i}. {rec}")
    
    # 如果有更多建议，提示用户
    if len(recommendations) > max_recommendations:
        output.append(f"... 还有 {len(recommendations) - max_recommendations} 条详细建议，请参考完整分析")
    
    # 总结性评价 - 更加突出和简洁
    output.append("\n🏆 分析总结 🏆")
    output.append("=" * 45)
    
    # 计算综合评分（0-100分）- 优化权重
    # 拟合优度得分 (0-35分)
    goodness_score = 0
    if quality['goodness_of_fit'] == '优秀':
        goodness_score = 35
    elif quality['goodness_of_fit'] == '良好':
        goodness_score = 28
    elif quality['goodness_of_fit'] == '一般':
        goodness_score = 20
    elif quality['goodness_of_fit'] == '较差':
        goodness_score = 10
    else:  # 很差
        goodness_score = 5
    
    # 数据代表性得分 (0-30分)
    representativeness_score = 0
    if quality['data_representativeness'] == '非常好':
        representativeness_score = 30
    elif quality['data_representativeness'] == '良好':
        representativeness_score = 25
    elif quality['data_representativeness'] == '一般':
        representativeness_score = 15
    else:  # 较差
        representativeness_score = 10
    
    # 数据质量得分 (0-35分) - 增加对统计特性的考虑
    data_quality_score = 35
    
    # 异常点扣分
    if len(results['filtered_indices']) > 0:
        outlier_percent = len(results['filtered_indices']) / len(results['original_data'][0]) * 100
        if outlier_percent > 20:
            data_quality_score -= 15
        elif outlier_percent > 10:
            data_quality_score -= 10
        elif outlier_percent > 5:
            data_quality_score -= 5
    
    # 样本量扣分
    n_points = len(results['filtered_data'][0])
    if n_points < 3:
        data_quality_score -= 15
    elif n_points < 5:
        data_quality_score -= 10
    elif n_points < 10:
        data_quality_score -= 5
    
    # 相关性加分（如果相关系数良好）
    if results['filtered_stats'] and 'correlation' in results['filtered_stats']:
        corr = results['filtered_stats']['correlation']
        if abs(corr) >= 0.9:
            data_quality_score += 5
        elif abs(corr) >= 0.7:
            data_quality_score += 3
    
    # 确保分数不小于0
    data_quality_score = max(0, data_quality_score)
    
    # 计算总分
    total_score = goodness_score + representativeness_score + data_quality_score
    
    # 总结评级
    if total_score >= 90:
        grade = "优秀"
        icon = "🏆"
    elif total_score >= 80:
        grade = "良好"
        icon = "👍"
    elif total_score >= 70:
        grade = "一般"
        icon = "💡"
    elif total_score >= 60:
        grade = "及格"
        icon = "⚠️"
    else:
        grade = "不及格"
        icon = "❌"
    
    # 添加总结评价
    output.append(f"综合评分: {total_score}/100 {icon}")
    output.append(f"评级: {grade}")
    
    # 详细总结 - 更简洁有力的指导
    if grade == "优秀":
        output.append("✅ 总体评价: 数据质量优秀，拟合效果极佳，结果可靠性高。")
        output.append("💡 此数据分析结果可用于发表研究论文或重要决策。")
    elif grade == "良好":
        output.append("✅ 总体评价: 数据质量良好，拟合结果可靠，可用于常规分析和决策。")
        output.append("💡 建议定期验证结果以确保稳定性，并考虑上述优化建议进一步提升质量。")
    elif grade == "一般":
        output.append("⚠️ 总体评价: 数据质量一般，拟合结果有一定参考价值，但存在明显改进空间。")
        output.append("💡 请务必按照上述建议进行优化，特别是增加样本量和处理异常点。")
    elif grade == "及格":
        output.append("⚠️ 总体评价: 数据质量和拟合效果基本合格，但可靠性有限。")
        output.append("💡 不建议将结果用于重要决策，必须增加数据量并改进实验方法。")
    else:  # 不及格
        output.append("❌ 总体评价: 数据质量较差，拟合结果可靠性低，无法用于专业分析。")
        output.append("💡 强烈建议重新设计实验，改进测量方法，收集新数据。")
    
    # 附加建议 - 更有针对性和可操作性
    output.append("\n🔮 后续工作建议 🔮")
    
    # 基于不同分数段给出不同级别的建议
    if grade in ['优秀', '良好']:
        output.append("1. 🧪 考虑对模型进行交叉验证，进一步验证其泛化能力")
        output.append("2. 📊 尝试不同的模型类型，比较效果差异")
        output.append("3. 🔍 进行敏感性分析，评估关键参数变化对结果的影响")
    elif grade in ['一般', '及格']:
        output.append("1. 📈 必须增加样本量，确保数据覆盖所有关键区域")
        output.append("2. 🔧 重新设计实验流程，减少测量误差")
        output.append("3. 🧮 尝试不同的数据预处理方法和模型类型")
    else:  # 不及格
        output.append("1. ⚠️ 立即停止使用当前数据进行决策，重新规划实验")
        output.append("2. 📚 咨询统计学专家，设计更合理的实验方案")
        output.append("3. 🔧 改进测量设备和方法，提高数据质量")
    
    # 通用建议 - 更加简洁实用
    output.append("\n📋 数据处理最佳实践:")
    output.append("✅ 始终保持原始数据记录，避免数据丢失")
    output.append("✅ 记录实验条件和环境参数，便于后续分析")
    output.append("✅ 对异常数据进行验证而非直接删除")
    output.append("✅ 结合专业知识解释统计结果，避免机械解读")
    
    return "\n".join(output)

def generate_experiment_plot_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """生成用于绘图的数据"""
    plot_data = {
        'original_x': results['original_data'][0].tolist(),
        'original_y': results['original_data'][1].tolist(),
        'filtered_x': results['filtered_data'][0].tolist(),
        'filtered_y': results['filtered_data'][1].tolist(),
        'filtered_indices': results['filtered_indices']
    }
    
    # 添加多项式拟合曲线
    if results['best_poly_fit']:
        x_min, x_max = min(results['filtered_data'][0]), max(results['filtered_data'][0])
        x_curve = np.linspace(x_min * 0.9, x_max * 1.1, 1000)
        poly_func = np.poly1d(results['best_poly_fit']['coeffs'])
        y_curve = poly_func(x_curve)
        plot_data['poly_fit_x'] = x_curve.tolist()
        plot_data['poly_fit_y'] = y_curve.tolist()
    
    # 添加平滑曲线（如果有）
    if results['smooth_curve']:
        plot_data['smooth_curve_x'] = results['smooth_curve'][0].tolist()
        plot_data['smooth_curve_y'] = results['smooth_curve'][1].tolist()
    
    return plot_data