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


def generate_batch_convergence_curve(
    initial_sinr_db: float,
    final_sinr_db: float,
    num_samples: int,
    convergence_speed: float = 0.05
) -> np.ndarray:
    """
    为批处理算法生成模拟指数收敛曲线

    Args:
        initial_sinr_db: 初始SINR (dB)
        final_sinr_db: 最终SINR (dB)
        num_samples: 样本数
        convergence_speed: 收敛速度系数

    Returns:
        收敛曲线数组
    """
    n = np.arange(num_samples)
    # 指数收敛: y = final - (final - initial) * exp(-speed * n)
    curve = final_sinr_db - (final_sinr_db - initial_sinr_db) * np.exp(-convergence_speed * n)
    return curve


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
        
        # 4. 自适应波束形成
        beamformer = AdaptiveBeamformer(array)
        bf_output = beamformer.process(
            array_received,
            signal_doa_deg,
            algorithm=algorithm
        )
        
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

        # 6. 执行所有算法对比
        algorithm_comparison = self._run_algorithm_comparison(
            array, array_received, signal_doa_deg, num_samples
        )

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

    def _run_algorithm_comparison(
        self,
        array: ArrayAntenna,
        array_signal: np.ndarray,
        signal_doa_deg: float,
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """
        运行所有五种算法进行对比

        Returns:
            包含五种算法结果的列表，每种算法包含收敛曲线
        """
        beamformer = AdaptiveBeamformer(array)
        algorithms = ["mvdr", "lms", "pi", "lcmv", "rls"]
        comparison_results = []

        # 估计初始SINR (使用未处理的信号)
        initial_sinr_db = -10.0  # 假设初始SINR较低

        for algo in algorithms:
            try:
                # 执行算法
                bf_output = beamformer.process(
                    array_signal,
                    signal_doa_deg,
                    algorithm=algo
                )

                # 获取或生成收敛曲线
                if bf_output.convergence_curve is not None:
                    # LMS和RLS有真实的收敛曲线 (误差平方)
                    # 转换为SINR估计: SINR ≈ -10*log10(error) + offset
                    error_curve = bf_output.convergence_curve
                    # 避免log(0)
                    error_curve = np.maximum(error_curve, 1e-10)
                    # 将误差转换为SINR估计
                    convergence_curve = 10 * np.log10(1.0 / error_curve)
                    # 限制合理范围
                    convergence_curve = np.clip(convergence_curve, -20, 60)
                else:
                    # 批处理算法生成模拟收敛曲线
                    final_sinr_db = bf_output.output_sinr_db
                    # 不同算法使用不同的收敛速度
                    speed_map = {
                        "mvdr": 0.08,   # MVDR收敛较快
                        "pi": 0.06,     # PI收敛中等
                        "lcmv": 0.07    # LCMV收敛较快
                    }
                    speed = speed_map.get(algo, 0.05)
                    convergence_curve = generate_batch_convergence_curve(
                        initial_sinr_db, final_sinr_db, num_samples, speed
                    )

                # 清理收敛曲线数据
                convergence_list = sanitize_float_list(convergence_curve.tolist())

                comparison_results.append({
                    "algorithm": algo,
                    "output_sinr_db": sanitize_float(bf_output.output_sinr_db),
                    "jammer_suppression_db": sanitize_float(bf_output.jammer_suppression_db),
                    "signal_distortion_db": sanitize_float(bf_output.signal_distortion_db),
                    "convergence_curve": convergence_list
                })
            except Exception as e:
                # 如果某个算法失败，添加空结果
                comparison_results.append({
                    "algorithm": algo,
                    "output_sinr_db": 0.0,
                    "jammer_suppression_db": 0.0,
                    "signal_distortion_db": 0.0,
                    "convergence_curve": [0.0] * num_samples
                })

        return comparison_results
