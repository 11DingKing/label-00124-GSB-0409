"""
API路由定义

提供GNSS授时仿真的REST API接口
"""

import uuid
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from ..models import (
    SimulationRequest,
    SimulationResponse,
    BiasResult,
    HealthResponse,
    SatelliteBias,
    BeamformingResult
)
from ..core import GNSSTimingSimulator

router = APIRouter()

# 仿真器实例
simulator = GNSSTimingSimulator()

# 结果缓存 (简单内存存储)
results_cache: Dict[str, SimulationResponse] = {}


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    健康检查接口
    
    返回服务状态信息
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now()
    )


@router.post("/api/v1/simulate", response_model=SimulationResponse, tags=["Simulation"])
async def run_simulation(request: SimulationRequest):
    """
    执行GNSS授时仿真
    
    分析自适应阵列天线抗干扰处理对GNSS信号的影响，
    计算由此引起的码相位偏差、载波相位偏差、伪距偏差和授时偏差。
    
    **参数说明:**
    - num_satellites: 可见卫星数量 (4-12)
    - array_elements: 阵列天线阵元数量 (4-8)
    - jammer_type: 干扰类型
        - noise: 宽带高斯噪声
        - cw: 窄带连续波
        - sweep: 线性扫频
        - pulse: 脉冲干扰
    - jnr_db: 干扰噪声比 (dB)
    - algorithm: 抗干扰算法
        - mvdr: 最小方差无失真响应
        - lms: 最小均方
        - pi: 功率反演
        - lcmv: 线性约束最小方差
        - rls: 递归最小二乘
    """
    try:
        # 执行仿真
        result = simulator.run_simulation(
            num_satellites=request.num_satellites,
            array_elements=request.array_elements,
            jammer_type=request.jammer_type,
            jnr_db=request.jnr_db,
            algorithm=request.algorithm,
            snr_db=request.snr_db,
            signal_doa_deg=request.signal_doa_deg,
            jammer_doa_deg=request.jammer_doa_deg,
            simulation_time_ms=request.simulation_time_ms
        )
        
        # 构造响应
        simulation_id = str(uuid.uuid4())
        
        # 转换卫星偏差列表
        satellite_biases = [
            SatelliteBias(
                prn=sat["prn"],
                elevation_deg=sat["elevation_deg"],
                azimuth_deg=sat["azimuth_deg"],
                code_phase_bias_chips=sat["code_phase_bias_chips"],
                carrier_phase_bias_cycles=sat["carrier_phase_bias_cycles"],
                pseudorange_bias_m=sat["pseudorange_bias_m"],
                timing_bias_ns=sat["timing_bias_ns"]
            )
            for sat in result["bias_analysis"]["satellite_biases"]
        ]
        
        response = SimulationResponse(
            simulation_id=simulation_id,
            timestamp=datetime.now(),
            status="completed",
            parameters=request,
            beamforming=BeamformingResult(
                algorithm=result["beamforming"]["algorithm"],
                weights_real=result["beamforming"]["weights_real"],
                weights_imag=result["beamforming"]["weights_imag"],
                output_sinr_db=result["beamforming"]["output_sinr_db"],
                jammer_suppression_db=result["beamforming"]["jammer_suppression_db"],
                signal_distortion_db=result["beamforming"]["signal_distortion_db"]
            ),
            bias_analysis=BiasResult(
                mean_code_phase_bias_chips=result["bias_analysis"]["mean_code_phase_bias_chips"],
                mean_carrier_phase_bias_cycles=result["bias_analysis"]["mean_carrier_phase_bias_cycles"],
                mean_pseudorange_bias_m=result["bias_analysis"]["mean_pseudorange_bias_m"],
                mean_timing_bias_ns=result["bias_analysis"]["mean_timing_bias_ns"],
                max_code_phase_bias_chips=result["bias_analysis"]["max_code_phase_bias_chips"],
                max_carrier_phase_bias_cycles=result["bias_analysis"]["max_carrier_phase_bias_cycles"],
                max_pseudorange_bias_m=result["bias_analysis"]["max_pseudorange_bias_m"],
                max_timing_bias_ns=result["bias_analysis"]["max_timing_bias_ns"],
                rms_code_phase_bias_chips=result["bias_analysis"]["rms_code_phase_bias_chips"],
                rms_carrier_phase_bias_cycles=result["bias_analysis"]["rms_carrier_phase_bias_cycles"],
                rms_pseudorange_bias_m=result["bias_analysis"]["rms_pseudorange_bias_m"],
                rms_timing_bias_ns=result["bias_analysis"]["rms_timing_bias_ns"],
                satellite_biases=satellite_biases
            ),
            algorithm_comparison=result.get("algorithm_comparison"),
            computation_time_ms=result["computation_time_ms"]
        )
        
        # 缓存结果
        results_cache[simulation_id] = response
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"仿真执行失败: {str(e)}")


@router.get("/api/v1/results/{simulation_id}", response_model=SimulationResponse, tags=["Simulation"])
async def get_simulation_result(simulation_id: str):
    """
    查询仿真结果
    
    根据仿真ID获取之前执行的仿真结果
    """
    if simulation_id not in results_cache:
        raise HTTPException(status_code=404, detail=f"未找到仿真结果: {simulation_id}")
        
    return results_cache[simulation_id]


@router.get("/api/v1/algorithms", tags=["Info"])
async def list_algorithms():
    """
    获取支持的抗干扰算法列表
    """
    return {
        "algorithms": [
            {
                "id": "mvdr",
                "name": "MVDR",
                "full_name": "Minimum Variance Distortionless Response",
                "description": "最小方差无失真响应，Capon波束形成，最优性能但计算复杂"
            },
            {
                "id": "lms",
                "name": "LMS",
                "full_name": "Least Mean Squares",
                "description": "最小均方自适应算法，实时性好，适合在线处理"
            },
            {
                "id": "pi",
                "name": "PI",
                "full_name": "Power Inversion",
                "description": "功率反演算法，工程中常用，计算简单"
            },
            {
                "id": "lcmv",
                "name": "LCMV",
                "full_name": "Linearly Constrained Minimum Variance",
                "description": "线性约束最小方差，支持多约束条件"
            },
            {
                "id": "rls",
                "name": "RLS",
                "full_name": "Recursive Least Squares",
                "description": "递归最小二乘，收敛比 LMS 更快，使用递归方法估计最优权重"
            }
        ]
    }


@router.get("/api/v1/jammers", tags=["Info"])
async def list_jammer_types():
    """
    获取支持的干扰类型列表
    """
    return {
        "jammer_types": [
            {
                "id": "noise",
                "name": "宽带噪声",
                "description": "高斯白噪声干扰，频谱平坦覆盖整个GNSS频段"
            },
            {
                "id": "cw",
                "name": "窄带连续波",
                "description": "单频正弦干扰，在频谱上表现为尖峰"
            },
            {
                "id": "sweep",
                "name": "线性扫频",
                "description": "频率随时间线性变化的chirp信号"
            },
            {
                "id": "pulse",
                "name": "脉冲干扰",
                "description": "周期性脉冲，模拟雷达或DME干扰"
            }
        ]
    }
