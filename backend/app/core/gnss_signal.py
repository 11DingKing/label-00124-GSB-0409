"""
GNSS信号生成模块

基于GPS L1 C/A码信号生成，参考开源项目:
- gps-sdr-sim (https://github.com/osqzss/gps-sdr-sim)
- gnss-sdr (https://github.com/gnss-sdr/gnss-sdr)

GPS L1 C/A信号参数:
- 载波频率: 1575.42 MHz
- 码速率: 1.023 Mcps
- 码长度: 1023 chips
- 数据速率: 50 bps
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# GPS L1 C/A 码生成多项式 (G1和G2寄存器的抽头位置)
# 参考: IS-GPS-200 Interface Specification
G2_TAPS = {
    1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9),
    6: (2, 10), 7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3),
    11: (3, 4), 12: (5, 6), 13: (6, 7), 14: (7, 8), 15: (8, 9),
    16: (9, 10), 17: (1, 4), 18: (2, 5), 19: (3, 6), 20: (4, 7),
    21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6), 25: (5, 7),
    26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
    31: (3, 8), 32: (4, 9),
}


@dataclass
class SatelliteSignal:
    """单颗卫星信号参数"""
    prn: int
    elevation_deg: float
    azimuth_deg: float
    doppler_hz: float
    code_phase_chips: float
    carrier_phase_rad: float
    cn0_dbhz: float


class GNSSSignalGenerator:
    """
    GNSS信号生成器
    
    生成GPS L1 C/A码信号，支持多卫星场景
    """
    
    # GPS L1 C/A 参数
    L1_FREQ_HZ = 1575.42e6  # L1载波频率
    CA_CODE_RATE = 1.023e6   # C/A码速率 (chips/s)
    CA_CODE_LENGTH = 1023    # C/A码长度 (chips)
    CHIP_DURATION = 1.0 / CA_CODE_RATE  # 码片持续时间
    WAVELENGTH = 299792458.0 / L1_FREQ_HZ  # L1波长 (m)
    CHIP_DISTANCE = 299792458.0 / CA_CODE_RATE  # 码片距离 (m) ≈ 293.05m
    
    def __init__(self, sample_rate: float = 4.092e6):
        """
        初始化信号生成器
        
        Args:
            sample_rate: 采样率 (Hz), 默认4.092 MHz (4倍码速率)
        """
        self.sample_rate = sample_rate
        self.samples_per_chip = sample_rate / self.CA_CODE_RATE
        self._ca_codes = {}  # PRN码缓存
        
    def generate_ca_code(self, prn: int) -> np.ndarray:
        """
        生成GPS C/A码
        
        基于Gold码序列生成，参考IS-GPS-200规范
        
        Args:
            prn: 卫星PRN号 (1-32)
            
        Returns:
            C/A码序列 (+1/-1)
        """
        if prn in self._ca_codes:
            return self._ca_codes[prn]
            
        if prn < 1 or prn > 32:
            raise ValueError(f"PRN must be 1-32, got {prn}")
            
        # G1寄存器 (1 + X^3 + X^10)
        g1 = np.ones(10, dtype=np.int8)
        # G2寄存器 (1 + X^2 + X^3 + X^6 + X^8 + X^9 + X^10)
        g2 = np.ones(10, dtype=np.int8)
        
        ca_code = np.zeros(self.CA_CODE_LENGTH, dtype=np.int8)
        
        tap1, tap2 = G2_TAPS[prn]
        
        for i in range(self.CA_CODE_LENGTH):
            # G1输出
            g1_out = g1[9]
            # G2输出 (选择性相加)
            g2_out = g2[tap1 - 1] ^ g2[tap2 - 1]
            # C/A码
            ca_code[i] = g1_out ^ g2_out
            
            # G1反馈
            g1_fb = g1[2] ^ g1[9]
            g1 = np.roll(g1, 1)
            g1[0] = g1_fb
            
            # G2反馈
            g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
            g2 = np.roll(g2, 1)
            g2[0] = g2_fb
            
        # 转换为 +1/-1
        ca_code = 1 - 2 * ca_code
        self._ca_codes[prn] = ca_code
        return ca_code
    
    def generate_satellite_geometry(
        self, 
        num_satellites: int,
        seed: Optional[int] = None
    ) -> List[SatelliteSignal]:
        """
        生成卫星几何配置
        
        Args:
            num_satellites: 卫星数量
            seed: 随机种子
            
        Returns:
            卫星信号参数列表
        """
        if seed is not None:
            np.random.seed(seed)
            
        satellites = []
        # 使用固定的PRN号
        prns = list(range(1, min(num_satellites + 1, 33)))
        
        for i, prn in enumerate(prns[:num_satellites]):
            # 生成合理的仰角 (15-85度)
            elevation = 15 + 70 * np.random.rand()
            # 均匀分布的方位角
            azimuth = 360 * i / num_satellites + 30 * (np.random.rand() - 0.5)
            azimuth = azimuth % 360
            # 多普勒频移 (-4kHz to +4kHz)
            doppler = 4000 * (2 * np.random.rand() - 1)
            # 随机码相位
            code_phase = self.CA_CODE_LENGTH * np.random.rand()
            # 随机载波相位
            carrier_phase = 2 * np.pi * np.random.rand()
            # 载噪比 (与仰角相关)
            cn0 = 35 + 15 * np.sin(np.radians(elevation))
            
            satellites.append(SatelliteSignal(
                prn=prn,
                elevation_deg=elevation,
                azimuth_deg=azimuth,
                doppler_hz=doppler,
                code_phase_chips=code_phase,
                carrier_phase_rad=carrier_phase,
                cn0_dbhz=cn0
            ))
            
        return satellites
    
    def generate_baseband_signal(
        self,
        satellite: SatelliteSignal,
        num_samples: int,
        snr_db: float = 45.0
    ) -> np.ndarray:
        """
        生成单颗卫星的基带信号
        
        Args:
            satellite: 卫星参数
            num_samples: 采样点数
            snr_db: 信噪比 (dB)
            
        Returns:
            复基带信号
        """
        # 时间向量
        t = np.arange(num_samples) / self.sample_rate
        
        # 获取C/A码
        ca_code = self.generate_ca_code(satellite.prn)
        
        # 码相位索引 (考虑多普勒对码速率的影响)
        code_freq = self.CA_CODE_RATE * (1 + satellite.doppler_hz / self.L1_FREQ_HZ)
        code_phase_samples = (satellite.code_phase_chips + code_freq * t) % self.CA_CODE_LENGTH
        code_indices = code_phase_samples.astype(int)
        
        # 扩频后的信号
        spreading_signal = ca_code[code_indices].astype(np.float64)
        
        # 载波信号 (多普勒频移)
        carrier_phase = satellite.carrier_phase_rad + 2 * np.pi * satellite.doppler_hz * t
        carrier = np.exp(1j * carrier_phase)
        
        # 复基带信号
        signal = spreading_signal * carrier
        
        # 信号功率归一化并加入噪声
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        return signal + noise
    
    def generate_multi_satellite_signal(
        self,
        satellites: List[SatelliteSignal],
        num_samples: int,
        snr_db: float = 45.0
    ) -> np.ndarray:
        """
        生成多卫星复合信号
        
        Args:
            satellites: 卫星参数列表
            num_samples: 采样点数
            snr_db: 信噪比 (dB)
            
        Returns:
            复合基带信号
        """
        combined_signal = np.zeros(num_samples, dtype=np.complex128)
        
        for sat in satellites:
            # 每颗卫星独立的信噪比，与仰角相关
            sat_snr = snr_db - 10 + 10 * np.sin(np.radians(sat.elevation_deg))
            signal = self.generate_baseband_signal(sat, num_samples, sat_snr)
            combined_signal += signal
            
        return combined_signal
    
    def correlate_with_local(
        self,
        received_signal: np.ndarray,
        prn: int,
        doppler_hz: float = 0.0,
        code_phase_chips: float = 0.0
    ) -> Tuple[float, float]:
        """
        与本地信号进行相关处理，检测码相位和载波相位
        
        Args:
            received_signal: 接收信号
            prn: 卫星PRN号
            doppler_hz: 本地多普勒频率
            code_phase_chips: 本地码相位
            
        Returns:
            (相关幅度, 相关相位)
        """
        num_samples = len(received_signal)
        t = np.arange(num_samples) / self.sample_rate
        
        # 本地C/A码
        ca_code = self.generate_ca_code(prn)
        
        # 本地码相位
        code_freq = self.CA_CODE_RATE * (1 + doppler_hz / self.L1_FREQ_HZ)
        code_phase_samples = (code_phase_chips + code_freq * t) % self.CA_CODE_LENGTH
        code_indices = code_phase_samples.astype(int)
        local_code = ca_code[code_indices].astype(np.float64)
        
        # 本地载波
        local_carrier = np.exp(-1j * 2 * np.pi * doppler_hz * t)
        
        # 相关运算
        corr = np.sum(received_signal * local_code * local_carrier)
        
        return np.abs(corr), np.angle(corr)
