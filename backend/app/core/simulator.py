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
        
        def generate_exponential_convergence(
            initial_value: float,
            final_value: float,
            num_points: int,
            tau: float = 0.15
        ) -> np.ndarray:
            """生成指数收敛曲线用于可视化对比"""
            t = np.linspace(0, 1, num_points)
            curve = final_value + (initial_value - final_value) * np.exp(-t / tau)
            return curve
        
        # 执行五种算法对比 (相同条件下)
        algorithms_to_compare = ["mvdr", "lms", "pi", "lcmv", "rls"]
        algorithm_outputs = {}
        initial_sinr_db = 5.0  # 初始SINR估计值
        
        for alg in algorithms_to_compare:
            algorithm_outputs[alg] = beamformer.process(
                array_received,
                signal_doa_deg,
                algorithm=alg
            )
        
        # 用户选择的算法的结果
        bf_output = algorithm_outputs[algorithm]
        
        # 5. 为批处理算法生成模拟收敛曲线，构建算法对比数据
        num_samples = array_received.shape[1]
        algorithm_comparison = []
        for alg in algorithms_to_compare:
            alg_output = algorithm_outputs[alg]
            final_sinr = alg_output.output_sinr_db
            
            if alg in ["lms", "rls"]:
                curve = alg_output.convergence_curve
                # 将误差曲线转换为SINR形式便于对比
                if curve is not None:
                    curve_norm = curve / (np.max(curve) + 1e-10)
                    sinr_curve = final_sinr + (initial_sinr_db - final_sinr) * curve_norm
                    curve_list = sanitize_float_list(sinr_curve.tolist())
                else:
                    curve_list = []
            else:
                exp_curve = generate_exponential_convergence(initial_sinr_db, final_sinr, num_samples)
                curve_list = sanitize_float_list(exp_curve.tolist())
            
            algorithm_comparison.append({
                "algorithm": alg,
                "convergence_curve": curve_list,
                "final_sinr_db": sanitize_float(final_sinr)
            })
        
        # 6. 获取原始信号 (不经过波束形成，仅第一阵元)
        original_signal = array_received[0, :]
        processed_signal = bf_output.output_signal
        
        # 7. 偏差分析
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
        
        # 获取当前算法的收敛曲线
        if algorithm in ["lms", "rls"]:
            current_curve = bf_output.convergence_curve
            convergence_curve_list = sanitize_float_list(current_curve.tolist()) if current_curve is not None else None
        else:
            alg_final_sinr = bf_output.output_sinr_db
            exp_curve = generate_exponential_convergence(initial_sinr_db, alg_final_sinr, num_samples)
            convergence_curve_list = sanitize_float_list(exp_curve.tolist())
        
        # 构造结果 - 使用sanitize_float确保所有值都是有限的
        result = {
            "beamforming": {
                "algorithm": algorithm,
                "weights_real": sanitize_float_list(weights_clean.real.tolist()),
                "weights_imag": sanitize_float_list(weights_clean.imag.tolist()),
                "output_sinr_db": sanitize_float(bf_output.output_sinr_db),
                "jammer_suppression_db": sanitize_float(bf_output.jammer_suppression_db),
                "signal_distortion_db": sanitize_float(bf_output.signal_distortion_db),
                "convergence_curve": convergence_curve_list
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
            "computation_time_ms": sanitize_float(computation_time_ms),
            "algorithm_comparison": algorithm_comparison
        }
        
        return result
