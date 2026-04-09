"""
干扰信号生成模块

支持多种典型GNSS干扰类型:
1. 宽带噪声干扰 (Broadband Noise)
2. 窄带连续波干扰 (Continuous Wave, CW)
3. 线性调频扫频干扰 (Swept/Chirp)
4. 脉冲干扰 (Pulsed)

参考文献:
- Kaplan & Hegarty, "Understanding GPS/GNSS: Principles and Applications"
- GNSS干扰与抗干扰技术相关研究
"""

import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass


@dataclass
class JammerConfig:
    """干扰配置参数"""
    jammer_type: Literal["noise", "cw", "sweep", "pulse"]
    jnr_db: float  # 干扰噪声比 (dB)
    frequency_offset_hz: float = 0.0  # 频率偏移 (Hz)
    bandwidth_hz: float = 2.0e6  # 带宽 (Hz), 用于扫频
    sweep_time_s: float = 1.0e-3  # 扫频周期 (s)
    pulse_duty_cycle: float = 0.1  # 脉冲占空比
    pulse_period_s: float = 1.0e-4  # 脉冲周期 (s)


class JammerGenerator:
    """
    干扰信号生成器
    
    生成各类典型GNSS干扰信号
    """
    
    def __init__(self, sample_rate: float = 4.092e6):
        """
        初始化干扰生成器
        
        Args:
            sample_rate: 采样率 (Hz)
        """
        self.sample_rate = sample_rate
        
    def generate(
        self,
        config: JammerConfig,
        num_samples: int,
        reference_power: float = 1.0
    ) -> np.ndarray:
        """
        生成干扰信号
        
        Args:
            config: 干扰配置
            num_samples: 采样点数
            reference_power: 参考信号功率
            
        Returns:
            干扰信号 (复信号)
        """
        # 计算干扰功率
        jammer_power = reference_power * (10 ** (config.jnr_db / 10))
        
        if config.jammer_type == "noise":
            jammer = self._generate_noise(num_samples, jammer_power)
        elif config.jammer_type == "cw":
            jammer = self._generate_cw(
                num_samples, jammer_power, config.frequency_offset_hz
            )
        elif config.jammer_type == "sweep":
            jammer = self._generate_sweep(
                num_samples, jammer_power, config.bandwidth_hz, config.sweep_time_s
            )
        elif config.jammer_type == "pulse":
            jammer = self._generate_pulse(
                num_samples, jammer_power, 
                config.pulse_duty_cycle, config.pulse_period_s
            )
        else:
            raise ValueError(f"Unknown jammer type: {config.jammer_type}")
            
        return jammer
    
    def _generate_noise(
        self, 
        num_samples: int, 
        power: float
    ) -> np.ndarray:
        """
        生成宽带高斯白噪声干扰
        
        特点: 频谱平坦，覆盖整个GNSS频段
        """
        std = np.sqrt(power / 2)
        noise = std * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        return noise
    
    def _generate_cw(
        self,
        num_samples: int,
        power: float,
        freq_offset_hz: float
    ) -> np.ndarray:
        """
        生成窄带连续波(CW)干扰
        
        特点: 单频正弦信号，在频谱上表现为尖峰
        典型应用: 模拟单频干扰源
        """
        t = np.arange(num_samples) / self.sample_rate
        amplitude = np.sqrt(power)
        
        # 添加随机初相位
        initial_phase = 2 * np.pi * np.random.rand()
        
        cw = amplitude * np.exp(1j * (2 * np.pi * freq_offset_hz * t + initial_phase))
        return cw
    
    def _generate_sweep(
        self,
        num_samples: int,
        power: float,
        bandwidth_hz: float,
        sweep_time_s: float
    ) -> np.ndarray:
        """
        生成线性调频扫频干扰
        
        特点: 频率随时间线性变化，覆盖一定带宽
        典型应用: 模拟扫频干扰机
        
        频率变化: f(t) = f0 + k*t, k = bandwidth/sweep_time
        """
        t = np.arange(num_samples) / self.sample_rate
        amplitude = np.sqrt(power)
        
        # 调频斜率
        chirp_rate = bandwidth_hz / sweep_time_s
        
        # 相位: φ(t) = 2π * (f0*t + 0.5*k*t^2)
        # 使用模运算实现周期性扫频
        t_mod = t % sweep_time_s
        f_start = -bandwidth_hz / 2
        phase = 2 * np.pi * (f_start * t_mod + 0.5 * chirp_rate * t_mod ** 2)
        
        sweep = amplitude * np.exp(1j * phase)
        return sweep
    
    def _generate_pulse(
        self,
        num_samples: int,
        power: float,
        duty_cycle: float,
        period_s: float
    ) -> np.ndarray:
        """
        生成脉冲干扰
        
        特点: 周期性脉冲，对接收机AGC和跟踪环路影响大
        典型应用: 模拟雷达干扰、DME干扰等
        """
        t = np.arange(num_samples) / self.sample_rate
        
        # 脉冲包络
        period_samples = int(period_s * self.sample_rate)
        pulse_samples = int(period_samples * duty_cycle)
        
        envelope = np.zeros(num_samples)
        for i in range(0, num_samples, max(1, period_samples)):
            end_idx = min(i + pulse_samples, num_samples)
            envelope[i:end_idx] = 1.0
            
        # 脉冲内为CW信号
        # 调整功率以保持平均功率
        peak_power = power / duty_cycle
        amplitude = np.sqrt(peak_power)
        
        # 随机频率偏移
        freq_offset = 100e3 * (2 * np.random.rand() - 1)
        carrier = amplitude * np.exp(1j * 2 * np.pi * freq_offset * t)
        
        pulse = envelope * carrier
        return pulse
    
    def get_jammer_description(self, config: JammerConfig) -> str:
        """获取干扰类型描述"""
        descriptions = {
            "noise": f"宽带高斯噪声干扰, JNR={config.jnr_db}dB",
            "cw": f"窄带连续波干扰, 频偏={config.frequency_offset_hz/1e3:.1f}kHz, JNR={config.jnr_db}dB",
            "sweep": f"线性扫频干扰, 带宽={config.bandwidth_hz/1e6:.1f}MHz, 周期={config.sweep_time_s*1e3:.1f}ms, JNR={config.jnr_db}dB",
            "pulse": f"脉冲干扰, 占空比={config.pulse_duty_cycle*100:.0f}%, JNR={config.jnr_db}dB",
        }
        return descriptions.get(config.jammer_type, "未知干扰类型")
