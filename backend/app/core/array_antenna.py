"""
阵列天线模型模块

实现均匀线阵(ULA - Uniform Linear Array)模型
用于空间信号接收和自适应波束形成

参考:
- Van Trees, "Optimum Array Processing"
- GPS L1频率: 1575.42 MHz, 波长 λ ≈ 19.05 cm
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ArrayConfig:
    """阵列配置参数"""
    num_elements: int  # 阵元数量
    element_spacing: float  # 阵元间距 (波长的倍数)
    carrier_freq_hz: float = 1575.42e6  # 载波频率


class ArrayAntenna:
    """
    均匀线阵(ULA)天线模型
    
    假设:
    - 各阵元全向性
    - 窄带信号模型
    - 远场平面波假设
    """
    
    SPEED_OF_LIGHT = 299792458.0  # 光速 (m/s)
    
    def __init__(self, config: ArrayConfig):
        """
        初始化阵列天线
        
        Args:
            config: 阵列配置
        """
        self.config = config
        self.num_elements = config.num_elements
        self.wavelength = self.SPEED_OF_LIGHT / config.carrier_freq_hz
        self.element_spacing_m = config.element_spacing * self.wavelength
        
        # 阵元位置 (一维线阵，沿x轴排列)
        self.element_positions = np.arange(self.num_elements) * self.element_spacing_m
        
    def steering_vector(self, theta_deg: float) -> np.ndarray:
        """
        计算导向矢量 (Steering Vector)
        
        对于ULA，导向矢量为:
        a(θ) = [1, e^(-j*2π*d*sin(θ)/λ), ..., e^(-j*2π*(N-1)*d*sin(θ)/λ)]^T
        
        Args:
            theta_deg: 信号到达角 (度), 相对于阵列法向
            
        Returns:
            导向矢量 (N x 1 复向量)
        """
        theta_rad = np.radians(theta_deg)
        
        # 空间频率
        spatial_freq = 2 * np.pi * self.element_spacing_m * np.sin(theta_rad) / self.wavelength
        
        # 导向矢量
        indices = np.arange(self.num_elements)
        steering = np.exp(-1j * spatial_freq * indices)
        
        return steering.reshape(-1, 1)
    
    def receive_signal(
        self,
        signals: List[Tuple[np.ndarray, float]],
        noise_power: float = 0.01
    ) -> np.ndarray:
        """
        模拟阵列接收信号
        
        Args:
            signals: 信号列表，每个元素为 (信号波形, 到达角度)
            noise_power: 噪声功率
            
        Returns:
            阵列接收信号 (N_elements x N_samples)
        """
        if not signals:
            raise ValueError("At least one signal is required")
            
        num_samples = len(signals[0][0])
        array_signal = np.zeros((self.num_elements, num_samples), dtype=np.complex128)
        
        # 叠加各入射信号
        for signal, doa_deg in signals:
            sv = self.steering_vector(doa_deg)
            array_signal += sv @ signal.reshape(1, -1)
            
        # 添加空间白噪声
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(self.num_elements, num_samples) + 
            1j * np.random.randn(self.num_elements, num_samples)
        )
        
        return array_signal + noise
    
    def compute_covariance(
        self,
        array_signal: np.ndarray,
        num_snapshots: Optional[int] = None
    ) -> np.ndarray:
        """
        估计协方差矩阵
        
        R = (1/K) * Σ x(k) * x(k)^H
        
        Args:
            array_signal: 阵列信号 (N_elements x N_samples)
            num_snapshots: 快拍数 (默认使用全部样本)
            
        Returns:
            协方差矩阵 (N_elements x N_elements)
        """
        if num_snapshots is None:
            num_snapshots = array_signal.shape[1]
            
        # 取前K个快拍
        X = array_signal[:, :num_snapshots]
        
        # 样本协方差矩阵
        R = (X @ X.conj().T) / num_snapshots
        
        return R
    
    def apply_weights(
        self,
        array_signal: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        应用波束形成权重
        
        y(t) = w^H * x(t)
        
        Args:
            array_signal: 阵列信号 (N_elements x N_samples)
            weights: 权重向量 (N_elements x 1)
            
        Returns:
            波束形成输出 (N_samples,)
        """
        # 确保权重为列向量
        w = weights.reshape(-1, 1)
        
        # 波束形成输出
        output = (w.conj().T @ array_signal).flatten()
        
        return output
    
    def compute_beam_pattern(
        self,
        weights: np.ndarray,
        theta_range: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算波束方向图
        
        Args:
            weights: 权重向量
            theta_range: 角度范围 (度)
            
        Returns:
            (角度数组, 增益数组(dB))
        """
        if theta_range is None:
            theta_range = np.linspace(-90, 90, 361)
            
        pattern = np.zeros(len(theta_range), dtype=np.complex128)
        
        for i, theta in enumerate(theta_range):
            sv = self.steering_vector(theta)
            pattern[i] = weights.conj().T @ sv
            
        # 转换为dB
        pattern_db = 20 * np.log10(np.abs(pattern) / np.max(np.abs(pattern)) + 1e-10)
        
        return theta_range, pattern_db.flatten()
    
    def compute_sinr(
        self,
        weights: np.ndarray,
        signal_covariance: np.ndarray,
        interference_covariance: np.ndarray,
        noise_power: float
    ) -> float:
        """
        计算输出信干噪比(SINR)
        
        SINR = (w^H * Rs * w) / (w^H * Ri * w + σ^2 * w^H * w)
        
        Args:
            weights: 权重向量
            signal_covariance: 信号协方差矩阵
            interference_covariance: 干扰协方差矩阵
            noise_power: 噪声功率
            
        Returns:
            SINR (线性值)
        """
        w = weights.flatten()
        
        signal_power = np.real(w.conj() @ signal_covariance @ w)
        interference_power = np.real(w.conj() @ interference_covariance @ w)
        noise_term = noise_power * np.sum(np.abs(w) ** 2)
        
        sinr = signal_power / (interference_power + noise_term + 1e-10)
        
        return sinr
