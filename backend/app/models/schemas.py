"""Pydantic models for API request/response schemas."""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class SimulationRequest(BaseModel):
    """仿真请求参数"""
    
    num_satellites: int = Field(
        default=6,
        ge=4,
        le=12,
        description="可见卫星数量 (4-12)"
    )
    array_elements: int = Field(
        default=4,
        ge=4,
        le=8,
        description="阵列天线阵元数量 (4-8)"
    )
    jammer_type: Literal["noise", "cw", "sweep", "pulse"] = Field(
        default="cw",
        description="干扰类型: noise(宽带噪声), cw(窄带连续波), sweep(扫频), pulse(脉冲)"
    )
    jnr_db: float = Field(
        default=20.0,
        ge=0,
        le=60,
        description="干扰噪声比 (dB)"
    )
    algorithm: Literal["mvdr", "lms", "pi", "lcmv", "rls"] = Field(
        default="mvdr",
        description="抗干扰算法: mvdr, lms, pi, lcmv, rls"
    )
    snr_db: float = Field(
        default=45.0,
        ge=20,
        le=60,
        description="信噪比 (dB)"
    )
    signal_doa_deg: float = Field(
        default=30.0,
        ge=-90,
        le=90,
        description="信号到达角 (度)"
    )
    jammer_doa_deg: float = Field(
        default=-45.0,
        ge=-90,
        le=90,
        description="干扰到达角 (度)"
    )
    simulation_time_ms: float = Field(
        default=1.0,
        ge=0.1,
        le=100.0,
        description="仿真时长 (毫秒)"
    )


class SatelliteBias(BaseModel):
    """单颗卫星的偏差结果"""
    
    prn: int = Field(description="卫星PRN号")
    elevation_deg: float = Field(description="仰角 (度)")
    azimuth_deg: float = Field(description="方位角 (度)")
    code_phase_bias_chips: float = Field(description="码相位偏差 (chips)")
    carrier_phase_bias_cycles: float = Field(description="载波相位偏差 (cycles)")
    pseudorange_bias_m: float = Field(description="伪距偏差 (meters)")
    timing_bias_ns: float = Field(description="授时偏差 (nanoseconds)")


class BiasResult(BaseModel):
    """偏差分析结果"""
    
    # 平均偏差
    mean_code_phase_bias_chips: float = Field(description="平均码相位偏差 (chips)")
    mean_carrier_phase_bias_cycles: float = Field(description="平均载波相位偏差 (cycles)")
    mean_pseudorange_bias_m: float = Field(description="平均伪距偏差 (meters)")
    mean_timing_bias_ns: float = Field(description="平均授时偏差 (nanoseconds)")
    
    # 最大偏差
    max_code_phase_bias_chips: float = Field(description="最大码相位偏差 (chips)")
    max_carrier_phase_bias_cycles: float = Field(description="最大载波相位偏差 (cycles)")
    max_pseudorange_bias_m: float = Field(description="最大伪距偏差 (meters)")
    max_timing_bias_ns: float = Field(description="最大授时偏差 (nanoseconds)")
    
    # RMS偏差
    rms_code_phase_bias_chips: float = Field(description="RMS码相位偏差 (chips)")
    rms_carrier_phase_bias_cycles: float = Field(description="RMS载波相位偏差 (cycles)")
    rms_pseudorange_bias_m: float = Field(description="RMS伪距偏差 (meters)")
    rms_timing_bias_ns: float = Field(description="RMS授时偏差 (nanoseconds)")
    
    # 各卫星详细结果
    satellite_biases: List[SatelliteBias] = Field(description="各卫星偏差详情")


class BeamformingResult(BaseModel):
    """波束形成结果"""

    algorithm: str = Field(description="使用的算法")
    weights_real: List[float] = Field(description="波束权重实部")
    weights_imag: List[float] = Field(description="波束权重虚部")
    output_sinr_db: float = Field(description="输出信干噪比 (dB)")
    jammer_suppression_db: float = Field(description="干扰抑制量 (dB)")
    signal_distortion_db: float = Field(description="信号失真 (dB)")


class AlgorithmComparisonResult(BaseModel):
    """算法对比结果"""

    algorithm: str = Field(description="算法名称")
    output_sinr_db: float = Field(description="输出信干噪比 (dB)")
    jammer_suppression_db: float = Field(description="干扰抑制量 (dB)")
    signal_distortion_db: float = Field(description="信号失真 (dB)")
    convergence_curve: List[float] = Field(description="收敛曲线数据")


class SimulationResponse(BaseModel):
    """仿真响应结果"""

    simulation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="仿真任务ID"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="仿真时间戳"
    )
    status: str = Field(default="completed", description="仿真状态")

    # 输入参数回显
    parameters: SimulationRequest = Field(description="仿真输入参数")

    # 波束形成结果
    beamforming: BeamformingResult = Field(description="波束形成结果")

    # 偏差分析结果
    bias_analysis: BiasResult = Field(description="偏差分析结果")

    # 算法对比结果
    algorithm_comparison: Optional[List[AlgorithmComparisonResult]] = Field(
        default=None,
        description="五种算法对比数据，包含收敛曲线"
    )

    # 性能指标
    computation_time_ms: float = Field(description="计算耗时 (毫秒)")


class HealthResponse(BaseModel):
    """健康检查响应"""
    
    status: str = Field(default="healthy", description="服务状态")
    version: str = Field(default="1.0.0", description="服务版本")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="响应时间戳"
    )
