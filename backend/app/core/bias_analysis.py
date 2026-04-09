"""
偏差分析模块

分析自适应波束形成引起的GNSS测量偏差:
1. 码相位偏差 (Code Phase Bias)
2. 载波相位偏差 (Carrier Phase Bias)
3. 伪距偏差 (Pseudorange Bias)
4. 授时偏差 (Timing Bias)

偏差来源:
- 自适应权重对信号波形的畸变
- 群时延变化
- 相位中心偏移

参考:
- Hatke et al., "Effects of Spatial Processing on GPS Receiver Accuracy"
- 阵列天线对GNSS接收机影响研究
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from .gnss_signal import GNSSSignalGenerator, SatelliteSignal


@dataclass
class SatelliteBiasResult:
    """单颗卫星偏差结果"""
    prn: int
    elevation_deg: float
    azimuth_deg: float
    code_phase_bias_chips: float
    carrier_phase_bias_cycles: float
    pseudorange_bias_m: float
    timing_bias_ns: float


@dataclass
class BiasAnalysisResult:
    """偏差分析总结果"""
    satellite_results: List[SatelliteBiasResult]
    
    # 统计量
    mean_code_phase_bias_chips: float
    mean_carrier_phase_bias_cycles: float
    mean_pseudorange_bias_m: float
    mean_timing_bias_ns: float
    
    max_code_phase_bias_chips: float
    max_carrier_phase_bias_cycles: float
    max_pseudorange_bias_m: float
    max_timing_bias_ns: float
    
    rms_code_phase_bias_chips: float
    rms_carrier_phase_bias_cycles: float
    rms_pseudorange_bias_m: float
    rms_timing_bias_ns: float


class BiasAnalyzer:
    """
    GNSS测量偏差分析器
    
    通过比较波束形成前后的信号，计算各类偏差
    """
    
    # GPS L1 C/A 参数
    CHIP_RATE = 1.023e6  # chips/s
    CARRIER_FREQ = 1575.42e6  # Hz
    SPEED_OF_LIGHT = 299792458.0  # m/s
    CHIP_DISTANCE = 299792458.0 / 1.023e6  # m/chip ≈ 293.05m
    WAVELENGTH = 299792458.0 / 1575.42e6  # m ≈ 0.1903m
    
    def __init__(self, signal_generator: GNSSSignalGenerator):
        """
        初始化偏差分析器
        
        Args:
            signal_generator: GNSS信号生成器
        """
        self.signal_generator = signal_generator
        self.sample_rate = signal_generator.sample_rate
        
    def analyze(
        self,
        satellites: List[SatelliteSignal],
        original_signal: np.ndarray,
        processed_signal: np.ndarray,
        weights: np.ndarray
    ) -> BiasAnalysisResult:
        """
        分析波束形成引起的偏差
        
        Args:
            satellites: 卫星参数列表
            original_signal: 原始信号 (无波束形成)
            processed_signal: 波束形成后的信号
            weights: 波束形成权重
            
        Returns:
            BiasAnalysisResult对象
        """
        satellite_results = []
        
        for sat in satellites:
            bias_result = self._analyze_single_satellite(
                sat, original_signal, processed_signal, weights
            )
            satellite_results.append(bias_result)
            
        # 计算统计量
        return self._compute_statistics(satellite_results)
    
    def _analyze_single_satellite(
        self,
        satellite: SatelliteSignal,
        original_signal: np.ndarray,
        processed_signal: np.ndarray,
        weights: np.ndarray
    ) -> SatelliteBiasResult:
        """
        分析单颗卫星的偏差
        """
        # 1. 码相位偏差分析
        code_phase_bias = self._compute_code_phase_bias(
            satellite, original_signal, processed_signal, weights
        )
        
        # 2. 载波相位偏差分析
        carrier_phase_bias = self._compute_carrier_phase_bias(
            satellite, original_signal, processed_signal, weights
        )
        
        # 3. 伪距偏差 (由码相位偏差导出)
        pseudorange_bias = code_phase_bias * self.CHIP_DISTANCE
        
        # 4. 授时偏差 (由伪距偏差导出)
        timing_bias = pseudorange_bias / self.SPEED_OF_LIGHT * 1e9  # 转换为纳秒
        
        return SatelliteBiasResult(
            prn=satellite.prn,
            elevation_deg=satellite.elevation_deg,
            azimuth_deg=satellite.azimuth_deg,
            code_phase_bias_chips=code_phase_bias,
            carrier_phase_bias_cycles=carrier_phase_bias,
            pseudorange_bias_m=pseudorange_bias,
            timing_bias_ns=timing_bias
        )
    
    def _compute_code_phase_bias(
        self,
        satellite: SatelliteSignal,
        original_signal: np.ndarray,
        processed_signal: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        计算码相位偏差
        
        通过相关峰位置的偏移来计算
        
        方法:
        1. 与本地码做相关
        2. 搜索相关峰
        3. 比较原始信号和处理后信号的峰值位置差异
        """
        # 简化处理: 通过相关函数分析
        # 实际中应该使用DLL鉴别器
        
        num_samples = min(len(original_signal), len(processed_signal))
        
        # 本地码生成
        ca_code = self.signal_generator.generate_ca_code(satellite.prn)
        
        # 采样索引
        t = np.arange(num_samples) / self.sample_rate
        code_phase_samples = (satellite.code_phase_chips + self.CHIP_RATE * t) % 1023
        code_indices = code_phase_samples.astype(int)
        local_code = ca_code[code_indices].astype(np.float64)
        
        # 本地载波
        local_carrier = np.exp(-1j * 2 * np.pi * satellite.doppler_hz * t)
        local_replica = local_code * local_carrier
        
        # 相关 - 原始信号
        corr_original = self._compute_correlation_peak(original_signal, local_replica)
        
        # 相关 - 处理后信号
        corr_processed = self._compute_correlation_peak(processed_signal, local_replica)
        
        # 码相位偏差 (通过相关峰位置差异估计)
        # 使用相位差来估计时延偏差
        phase_diff = np.angle(corr_processed) - np.angle(corr_original)
        
        # 将相位差转换为码相位偏差
        # 相位变化与群时延相关
        # 简化模型: 假设相位偏差与时延线性相关
        # Δτ ≈ Δφ / (2π * f_code)
        # 转换为chips
        code_phase_bias = phase_diff / (2 * np.pi) * (self.CHIP_RATE / self.CARRIER_FREQ) * 1023
        
        # 加入波束形成导致的群时延估计
        # 实际偏差与权重相位梯度相关
        weights_phase = np.angle(weights)
        if len(weights_phase) > 1:
            phase_gradient = np.gradient(weights_phase)
            group_delay_chips = np.mean(phase_gradient) * 0.01  # 简化模型
            code_phase_bias += group_delay_chips
            
        return float(code_phase_bias)
    
    def _compute_carrier_phase_bias(
        self,
        satellite: SatelliteSignal,
        original_signal: np.ndarray,
        processed_signal: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        计算载波相位偏差
        
        由自适应权重引起的相位偏移
        
        方法:
        1. 通过PLL鉴相器输出分析
        2. 比较原始和处理后信号的相位
        """
        num_samples = min(len(original_signal), len(processed_signal))
        
        # 提取载波相位
        phase_original = np.angle(original_signal[:num_samples])
        phase_processed = np.angle(processed_signal[:num_samples])
        
        # 解缠绕
        phase_original = np.unwrap(phase_original)
        phase_processed = np.unwrap(phase_processed)
        
        # 相位差
        phase_diff = phase_processed - phase_original
        
        # 去除趋势 (频率偏差)
        t = np.arange(num_samples) / self.sample_rate
        if len(t) > 1:
            p = np.polyfit(t, phase_diff, 1)
            phase_diff_detrend = phase_diff - np.polyval(p, t)
        else:
            phase_diff_detrend = phase_diff
            
        # 载波相位偏差 (cycles)
        mean_phase_bias_rad = np.mean(phase_diff_detrend)
        carrier_phase_bias_cycles = mean_phase_bias_rad / (2 * np.pi)
        
        # 权重引起的相位偏差
        # 波束指向偏差导致的相位中心偏移
        weights_phase = np.unwrap(np.angle(weights.flatten()))
        if len(weights_phase) > 1:
            phase_center_shift = np.mean(weights_phase) / (2 * np.pi)
            carrier_phase_bias_cycles += phase_center_shift * 0.1  # 缩放因子
            
        return float(carrier_phase_bias_cycles)
    
    def _compute_correlation_peak(
        self,
        signal: np.ndarray,
        local_replica: np.ndarray
    ) -> complex:
        """
        计算相关峰
        """
        num_samples = min(len(signal), len(local_replica))
        correlation = np.sum(signal[:num_samples] * local_replica[:num_samples].conj())
        return correlation
    
    def _compute_statistics(
        self,
        satellite_results: List[SatelliteBiasResult]
    ) -> BiasAnalysisResult:
        """
        计算偏差统计量
        """
        if not satellite_results:
            # 返回零偏差
            return BiasAnalysisResult(
                satellite_results=[],
                mean_code_phase_bias_chips=0.0,
                mean_carrier_phase_bias_cycles=0.0,
                mean_pseudorange_bias_m=0.0,
                mean_timing_bias_ns=0.0,
                max_code_phase_bias_chips=0.0,
                max_carrier_phase_bias_cycles=0.0,
                max_pseudorange_bias_m=0.0,
                max_timing_bias_ns=0.0,
                rms_code_phase_bias_chips=0.0,
                rms_carrier_phase_bias_cycles=0.0,
                rms_pseudorange_bias_m=0.0,
                rms_timing_bias_ns=0.0
            )
            
        # 提取偏差数组
        code_biases = np.array([r.code_phase_bias_chips for r in satellite_results])
        carrier_biases = np.array([r.carrier_phase_bias_cycles for r in satellite_results])
        pr_biases = np.array([r.pseudorange_bias_m for r in satellite_results])
        timing_biases = np.array([r.timing_bias_ns for r in satellite_results])
        
        return BiasAnalysisResult(
            satellite_results=satellite_results,
            
            # 均值
            mean_code_phase_bias_chips=float(np.mean(np.abs(code_biases))),
            mean_carrier_phase_bias_cycles=float(np.mean(np.abs(carrier_biases))),
            mean_pseudorange_bias_m=float(np.mean(np.abs(pr_biases))),
            mean_timing_bias_ns=float(np.mean(np.abs(timing_biases))),
            
            # 最大值
            max_code_phase_bias_chips=float(np.max(np.abs(code_biases))),
            max_carrier_phase_bias_cycles=float(np.max(np.abs(carrier_biases))),
            max_pseudorange_bias_m=float(np.max(np.abs(pr_biases))),
            max_timing_bias_ns=float(np.max(np.abs(timing_biases))),
            
            # RMS
            rms_code_phase_bias_chips=float(np.sqrt(np.mean(code_biases ** 2))),
            rms_carrier_phase_bias_cycles=float(np.sqrt(np.mean(carrier_biases ** 2))),
            rms_pseudorange_bias_m=float(np.sqrt(np.mean(pr_biases ** 2))),
            rms_timing_bias_ns=float(np.sqrt(np.mean(timing_biases ** 2)))
        )
