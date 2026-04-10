"""
GNSS授时仿真主控模块

整合各子模块，执行完整的仿真流程:
1. 生成GNSS信号
2. 生成干扰信号
3. 阵列天线接收
4. 自适应波束形成
5. 偏差分析
"""

import numpy as np
import time
import math
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from .gnss_signal import GNSSSignalGenerator, SatelliteSignal
from .jammer import JammerGenerator, JammerConfig
from .array_antenna import ArrayAntenna, ArrayConfig
from .beamforming import AdaptiveBeamformer, BeamformingOutput
from .bias_analysis import BiasAnalyzer, BiasAnalysisResult


def sanitize_float(value: float, default: float = 0.0) -> float:
    """确保浮点数是有限的JSON兼容值"""
    if not isinstance(value, (int, float)):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return float(value)


def sanitize_float_list(values: List[float], default: float = 0.0) -> List[float]:
    """清理浮点数列表"""
    return [sanitize_float(v, default) for v in values]


def generate_exponential_convergence(
    initial_value: float,
    final_value: float,
    num_samples: int,
    time_constant: float = 0.15
) -> np.ndarray:
    """
    生成指数收敛曲线
    
    用于批处理算法的模拟收敛曲线：从初始SINR指数收敛到最终SINR
    
    curve(n) = final_value + (initial_value - final_value) * exp(-n / (time_constant * num_samples))
    
    Args:
        initial_value: 初始值
        final_value: 最终收敛值
        num_samples: 采样点数
        time_constant: 时间常数，控制收敛速度(0-1)
        
    Returns:
        收敛曲线数组
    """
    n = np.arange(num_samples)
    tau = time_constant * num_samples  # 使 time_constant 比例因子更直观
    convergence = final_value + (initial_value - final_value) * np.exp(-n / tau)
    return convergence


def generate_sinr_from_error(
    error_curve: np.ndarray,
    initial_sinr: float,
    final_sinr: float
) -> List[float]:
    """
    将自适应算法的误差曲线转换为 SINR 收敛曲线
    
    用于 LMS/RLS 等输出误差平方的自适应算法，使对比更有意义
    
    Args:
        error_curve: 原始误差平方曲线
        initial_sinr: 初始 SINR 值
        final_sinr: 最终 SINR 值
        
    Returns:
        SINR 收敛曲线
    """
    num_samples = len(error_curve)
    
    # 平滑误差曲线以减少噪声
    window_size = max(5, num_samples // 50)
    if window_size > 1:
        smoothed_error = np.convolve(
            error_curve, 
            np.ones(window_size) / window_size, 
            mode="same"
        )
    else:
        smoothed_error = error_curve
    
    # 归一化平滑后的误差曲线 (反转，因为误差越小，SINR越高)
    max_error = np.max(smoothed_error)
    min_error = np.min(smoothed_error)
    
    if max_error > min_error:
        normalized_error = (smoothed_error - min_error) / (max_error - min_error)
        # 反转并映射到 SINR 范围：误差 = 1 -> SINR = initial，误差 = 0 -> SINR = final
        sinr_curve = final_sinr + (initial_sinr - final_sinr) * normalized_error
    else:
        # 误差不变，生成简单的指数曲线
        sinr_curve = generate_exponential_convergence(initial_sinr, final_sinr, num_samples)
    
    return sinr_curve.tolist()


class GNSSTimingSimulator:
    """
    GNSS授时仿真器
    
    完整仿真流程管理
    """
    
    def __init__(self, sample_rate: float = 4.092e6):
        """
        初始化仿真器
        
        Args:
            sample_rate: 采样率 (Hz)
        """
        self.sample_rate = sample_rate
        self.signal_generator = GNSSSignalGenerator(sample_rate)
        self.jammer_generator = JammerGenerator(sample_rate)
        
    def run_simulation(
        self,
        num_satellites: int = 6,
        array_elements: int = 4,
        jammer_type: str = "cw",
        jnr_db: float = 20.0,
        algorithm: str = "mvdr",
        snr_db: float = 45.0,
        signal_doa_deg: float = 30.0,
        jammer_doa_deg: float = -45.0,
        simulation_time_ms: float = 1.0,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行完整仿真
        
        Args:
            num_satellites: 卫星数量
            array_elements: 阵元数量
            jammer_type: 干扰类型
            jnr_db: 干扰噪声比
            algorithm: 抗干扰算法
            snr_db: 信噪比
            signal_doa_deg: 信号到达角
            jammer_doa_deg: 干扰到达角
            simulation_time_ms: 仿真时长
            seed: 随机种子
            
        Returns:
            仿真结果字典
        """
        start_time = time.time()
        
        if seed is not None:
            np.random.seed(seed)
            
        # 计算采样点数
        num_samples = int(simulation_time_ms * 1e-3 * self.sample_rate)
        
        # 1. 生成卫星几何和信号
        satellites = self.signal_generator.generate_satellite_geometry(
            num_satellites, seed=seed
        )
        
        gnss_signal = self.signal_generator.generate_multi_satellite_signal(
            satellites, num_samples, snr_db
        )
        
        # 2. 生成干扰信号
        jammer_config = JammerConfig(
            jammer_type=jammer_type,
            jnr_db=jnr_db,
            frequency_offset_hz=100e3 if jammer_type == "cw" else 0,
            bandwidth_hz=2e6,
            sweep_time_s=1e-3,
            pulse_duty_cycle=0.1,
            pulse_period_s=1e-4
        )
        
        reference_power = np.mean(np.abs(gnss_signal) ** 2)
        jammer_signal = self.jammer_generator.generate(
            jammer_config, num_samples, reference_power
        )
        
        # 3. 阵列天线接收
        array_config = ArrayConfig(
            num_elements=array_elements,
            element_spacing=0.5,  # 半波长
            carrier_freq_hz=1575.42e6
        )
        array = ArrayAntenna(array_config)
        
        # 构造入射信号列表
        signals_with_doa = [
            (gnss_signal, signal_doa_deg),
            (jammer_signal, jammer_doa_deg)
        ]
        
        noise_power = reference_power / (10 ** (snr_db / 10)) * 0.1
        array_received = array.receive_signal(signals_with_doa, noise_power)
        
        # 4. 自适应波束形成 - 用户选择的算法
        beamformer = AdaptiveBeamformer(array)
        bf_output = beamformer.process(
            array_received,
            signal_doa_deg,
            algorithm=algorithm
        )
        
        # 4b. 运行所有五种算法用于对比 - 生成 algorithm_comparison 数据
        all_algorithms = ["mvdr", "lms", "pi", "lcmv", "rls"]
        batch_algorithms = ["mvdr", "pi", "lcmv"]
        
        algorithm_comparison = []
        for algo in all_algorithms:
            algo_output = beamformer.process(
                array_received,
                signal_doa_deg,
                algorithm=algo
            )
            
            # 获取或生成收敛曲线
            if algo in batch_algorithms:
                # 批处理算法没有逐样本收敛曲线
                # 模拟生成一条从初始 SINR 到最终 SINR 的指数收敛曲线
                num_samples = array_received.shape[1]
                
                # 计算初始 SINR：使用初始权重（仅指向期望方向）
                initial_a = array.steering_vector(signal_doa_deg)
                initial_weights = initial_a.copy() / array_elements
                initial_R = array.compute_covariance(array_received)
                initial_metrics = beamformer._compute_metrics(
                    initial_weights, initial_a, initial_R, array_received
                )
                initial_sinr = initial_metrics["sinr_db"]
                final_sinr = algo_output.output_sinr_db
                
                # 生成指数收敛曲线：从初始值收敛到最终值
                convergence_curve = generate_exponential_convergence(
                    initial_sinr, final_sinr, num_samples
                )
            else:
                # 自适应算法 (LMS, RLS) 有真实收敛曲线
                if algo_output.convergence_curve is not None:
                    # 我们需要将误差曲线转换为 SINR 曲线以便对比
                    # 使用归一化误差曲线映射到 SINR 范围
                    error_curve = algo_output.convergence_curve
                    num_samples = len(error_curve)
                    
                    # 计算初始 SINR
                    initial_a = array.steering_vector(signal_doa_deg)
                    initial_weights = initial_a.copy() / array_elements
                    initial_R = array.compute_covariance(array_received)
                    initial_metrics = beamformer._compute_metrics(
                        initial_weights, initial_a, initial_R, array_received
                    )
                    initial_sinr = initial_metrics["sinr_db"]
                    final_sinr = algo_output.output_sinr_db
                    
                    # 基于误差曲线生成 SINR 收敛曲线
                    convergence_curve = generate_sinr_from_error(
                        error_curve, initial_sinr, final_sinr
                    )
                else:
                    # Fallback: 生成模拟曲线
                    initial_a = array.steering_vector(signal_doa_deg)
                    initial_weights = initial_a.copy() / array_elements
                    initial_R = array.compute_covariance(array_received)
                    initial_metrics = beamformer._compute_metrics(
                        initial_weights, initial_a, initial_R, array_received
                    )
                    initial_sinr = initial_metrics["sinr_db"]
                    final_sinr = algo_output.output_sinr_db
                    convergence_curve = generate_exponential_convergence(
                        initial_sinr, final_sinr, array_received.shape[1]
                    )
            
            algorithm_comparison.append({
                "algorithm": algo,
                "output_sinr_db": sanitize_float(algo_output.output_sinr_db),
                "convergence_curve": sanitize_float_list(convergence_curve.tolist() 
                    if isinstance(convergence_curve, np.ndarray) 
                    else convergence_curve)
            })
        
        # 5. 获取原始信号 (不经过波束形成，仅第一阵元)
        original_signal = array_received[0, :]
        processed_signal = bf_output.output_signal
        
        # 6. 偏差分析
        bias_analyzer = BiasAnalyzer(self.signal_generator)
        bias_result = bias_analyzer.analyze(
            satellites,
            original_signal,
            processed_signal,
            bf_output.weights
        )
        
        # 计算耗时
        computation_time_ms = (time.time() - start_time) * 1000
        
        # 清理权重中的NaN/Inf值
        weights_clean = np.nan_to_num(bf_output.weights, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 构造结果 - 使用sanitize_float确保所有值都是有限的
        result = {
            "beamforming": {
                "algorithm": algorithm,
                "weights_real": sanitize_float_list(weights_clean.real.tolist()),
                "weights_imag": sanitize_float_list(weights_clean.imag.tolist()),
                "output_sinr_db": sanitize_float(bf_output.output_sinr_db),
                "jammer_suppression_db": sanitize_float(bf_output.jammer_suppression_db),
                "signal_distortion_db": sanitize_float(bf_output.signal_distortion_db)
            },
            "bias_analysis": {
                "mean_code_phase_bias_chips": sanitize_float(bias_result.mean_code_phase_bias_chips),
                "mean_carrier_phase_bias_cycles": sanitize_float(bias_result.mean_carrier_phase_bias_cycles),
                "mean_pseudorange_bias_m": sanitize_float(bias_result.mean_pseudorange_bias_m),
                "mean_timing_bias_ns": sanitize_float(bias_result.mean_timing_bias_ns),
                "max_code_phase_bias_chips": sanitize_float(bias_result.max_code_phase_bias_chips),
                "max_carrier_phase_bias_cycles": sanitize_float(bias_result.max_carrier_phase_bias_cycles),
                "max_pseudorange_bias_m": sanitize_float(bias_result.max_pseudorange_bias_m),
                "max_timing_bias_ns": sanitize_float(bias_result.max_timing_bias_ns),
                "rms_code_phase_bias_chips": sanitize_float(bias_result.rms_code_phase_bias_chips),
                "rms_carrier_phase_bias_cycles": sanitize_float(bias_result.rms_carrier_phase_bias_cycles),
                "rms_pseudorange_bias_m": sanitize_float(bias_result.rms_pseudorange_bias_m),
                "rms_timing_bias_ns": sanitize_float(bias_result.rms_timing_bias_ns),
                "satellite_biases": [
                    {
                        "prn": sat.prn,
                        "elevation_deg": sanitize_float(sat.elevation_deg),
                        "azimuth_deg": sanitize_float(sat.azimuth_deg),
                        "code_phase_bias_chips": sanitize_float(sat.code_phase_bias_chips),
                        "carrier_phase_bias_cycles": sanitize_float(sat.carrier_phase_bias_cycles),
                        "pseudorange_bias_m": sanitize_float(sat.pseudorange_bias_m),
                        "timing_bias_ns": sanitize_float(sat.timing_bias_ns)
                    }
                    for sat in bias_result.satellite_results
                ]
            },
            "algorithm_comparison": algorithm_comparison,
            "computation_time_ms": sanitize_float(computation_time_ms)
        }
        
        return result
