"""
自适应波束形成模块

实现多种经典自适应抗干扰算法:
1. MVDR (Minimum Variance Distortionless Response) - 最小方差无失真响应
2. LMS (Least Mean Squares) - 最小均方
3. PI (Power Inversion) - 功率反演
4. LCMV (Linearly Constrained Minimum Variance) - 线性约束最小方差
5. RLS (Recursive Least Squares) - 递归最小二乘

参考:
- Van Trees, "Optimum Array Processing"
- Compton, "Adaptive Antennas: Concepts and Performance"
- 阵列信号处理相关文献
"""

import numpy as np
from typing import Tuple, Optional, Literal
from dataclasses import dataclass

from .array_antenna import ArrayAntenna


@dataclass
class BeamformingOutput:
    """波束形成输出"""
    weights: np.ndarray  # 波束权重
    output_signal: np.ndarray  # 输出信号
    output_sinr_db: float  # 输出SINR
    jammer_suppression_db: float  # 干扰抑制量
    signal_distortion_db: float  # 信号失真
    convergence_curve: Optional[np.ndarray] = None  # 收敛曲线(用于LMS/RLS)


class AdaptiveBeamformer:
    """
    自适应波束形成器
    
    实现多种抗干扰算法
    """
    
    def __init__(self, array: ArrayAntenna):
        """
        初始化波束形成器
        
        Args:
            array: 阵列天线对象
        """
        self.array = array
        self.num_elements = array.num_elements
        
    def process(
        self,
        array_signal: np.ndarray,
        signal_doa_deg: float,
        algorithm: Literal["mvdr", "lms", "pi", "lcmv", "rls"] = "mvdr",
        diagonal_loading: float = 0.01,
        lms_step_size: float = 0.01,
        rls_forgetting_factor: float = 0.99,
        rls_delta: float = 0.01,
        reference_signal: Optional[np.ndarray] = None
    ) -> BeamformingOutput:
        """
        执行自适应波束形成

        Args:
            array_signal: 阵列接收信号 (N_elements x N_samples)
            signal_doa_deg: 期望信号到达角
            algorithm: 算法选择
            diagonal_loading: 对角加载系数
            lms_step_size: LMS步长
            rls_forgetting_factor: RLS遗忘因子 (默认0.99)
            rls_delta: RLS初始化参数 (默认0.01)
            reference_signal: 参考信号(用于LMS/RLS)

        Returns:
            BeamformingOutput对象
        """
        if algorithm == "mvdr":
            return self._mvdr(array_signal, signal_doa_deg, diagonal_loading)
        elif algorithm == "lms":
            return self._lms(array_signal, signal_doa_deg, lms_step_size, reference_signal)
        elif algorithm == "pi":
            return self._power_inversion(array_signal, signal_doa_deg, diagonal_loading)
        elif algorithm == "lcmv":
            return self._lcmv(array_signal, signal_doa_deg, diagonal_loading)
        elif algorithm == "rls":
            return self._rls(array_signal, signal_doa_deg, rls_forgetting_factor, rls_delta, reference_signal)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _mvdr(
        self,
        array_signal: np.ndarray,
        signal_doa_deg: float,
        diagonal_loading: float
    ) -> BeamformingOutput:
        """
        MVDR波束形成 (Capon波束形成)
        
        优化问题:
        min w^H*R*w  subject to w^H*a(θ0) = 1
        
        解:
        w = R^(-1)*a(θ0) / (a(θ0)^H*R^(-1)*a(θ0))
        """
        # 估计协方差矩阵
        R = self.array.compute_covariance(array_signal)
        
        # 对角加载提高数值稳定性
        R_loaded = R + diagonal_loading * np.trace(R) * np.eye(self.num_elements) / self.num_elements
        
        # 期望信号导向矢量
        a = self.array.steering_vector(signal_doa_deg)
        
        # MVDR权重
        try:
            R_inv = np.linalg.inv(R_loaded)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R_loaded)
            
        R_inv_a = R_inv @ a
        denominator = a.conj().T @ R_inv_a
        weights = R_inv_a / denominator
        
        # 计算输出
        output_signal = self.array.apply_weights(array_signal, weights)
        
        # 计算性能指标
        metrics = self._compute_metrics(weights, a, R, array_signal)
        
        return BeamformingOutput(
            weights=weights.flatten(),
            output_signal=output_signal,
            output_sinr_db=metrics["sinr_db"],
            jammer_suppression_db=metrics["jammer_suppression_db"],
            signal_distortion_db=metrics["signal_distortion_db"]
        )
    
    def _lms(
        self,
        array_signal: np.ndarray,
        signal_doa_deg: float,
        step_size: float,
        reference_signal: Optional[np.ndarray]
    ) -> BeamformingOutput:
        """
        LMS自适应算法
        
        更新规则:
        w(n+1) = w(n) + μ * e(n) * x*(n)
        e(n) = d(n) - w^H(n)*x(n)
        """
        num_samples = array_signal.shape[1]
        
        # 初始权重 (指向期望方向)
        a = self.array.steering_vector(signal_doa_deg)
        weights = a.copy() / self.num_elements
        
        # 如果没有参考信号，使用导向矢量约束
        if reference_signal is None:
            reference_signal = np.ones(num_samples, dtype=np.complex128)
            
        # 输出信号和收敛曲线
        output_signal = np.zeros(num_samples, dtype=np.complex128)
        convergence = np.zeros(num_samples)
        
        # LMS迭代
        for n in range(num_samples):
            x_n = array_signal[:, n:n+1]
            
            # 输出
            y_n = (weights.conj().T @ x_n).item()
            output_signal[n] = y_n
            
            # 误差
            e_n = reference_signal[n] - y_n
            convergence[n] = np.abs(e_n) ** 2
            
            # 权重更新 (归一化LMS)
            norm_factor = x_n.conj().T @ x_n + 1e-6
            weights = weights + step_size * e_n * x_n / norm_factor
            
            # 约束: 保持期望方向增益为1
            constraint_error = 1.0 - (weights.conj().T @ a).item()
            weights = weights + constraint_error * a / (a.conj().T @ a)
            
        # 计算最终协方差矩阵用于性能评估
        R = self.array.compute_covariance(array_signal)
        metrics = self._compute_metrics(weights, a, R, array_signal)
        
        return BeamformingOutput(
            weights=weights.flatten(),
            output_signal=output_signal,
            output_sinr_db=metrics["sinr_db"],
            jammer_suppression_db=metrics["jammer_suppression_db"],
            signal_distortion_db=metrics["signal_distortion_db"],
            convergence_curve=convergence
        )
    
    def _power_inversion(
        self,
        array_signal: np.ndarray,
        signal_doa_deg: float,
        diagonal_loading: float
    ) -> BeamformingOutput:
        """
        功率反演(PI)算法
        
        简化的MVDR，直接使用R^(-1)作为权重基础
        w = R^(-1) * a(θ0)
        
        工程中常用，计算简单
        """
        # 估计协方差矩阵
        R = self.array.compute_covariance(array_signal)
        
        # 对角加载
        R_loaded = R + diagonal_loading * np.trace(R) * np.eye(self.num_elements) / self.num_elements
        
        # 期望信号导向矢量
        a = self.array.steering_vector(signal_doa_deg)
        
        # 功率反演权重
        try:
            R_inv = np.linalg.inv(R_loaded)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R_loaded)
            
        weights = R_inv @ a
        
        # 归一化
        weights = weights / np.linalg.norm(weights)
        
        # 计算输出
        output_signal = self.array.apply_weights(array_signal, weights)
        
        # 计算性能指标
        metrics = self._compute_metrics(weights, a, R, array_signal)
        
        return BeamformingOutput(
            weights=weights.flatten(),
            output_signal=output_signal,
            output_sinr_db=metrics["sinr_db"],
            jammer_suppression_db=metrics["jammer_suppression_db"],
            signal_distortion_db=metrics["signal_distortion_db"]
        )
    
    def _lcmv(
        self,
        array_signal: np.ndarray,
        signal_doa_deg: float,
        diagonal_loading: float
    ) -> BeamformingOutput:
        """
        LCMV波束形成
        
        线性约束最小方差，可设置多个约束
        
        优化问题:
        min w^H*R*w  subject to C^H*w = f
        
        这里使用单约束: 期望方向增益为1
        """
        # 估计协方差矩阵
        R = self.array.compute_covariance(array_signal)
        
        # 对角加载
        R_loaded = R + diagonal_loading * np.trace(R) * np.eye(self.num_elements) / self.num_elements
        
        # 约束矩阵 (这里只使用单约束，与MVDR等价)
        # 可扩展为多约束
        C = self.array.steering_vector(signal_doa_deg)
        f = np.array([[1.0]])
        
        # LCMV权重
        try:
            R_inv = np.linalg.inv(R_loaded)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R_loaded)
            
        R_inv_C = R_inv @ C
        denominator = C.conj().T @ R_inv_C
        weights = R_inv_C @ np.linalg.inv(denominator) @ f
        
        # 计算输出
        output_signal = self.array.apply_weights(array_signal, weights)
        
        # 计算性能指标
        metrics = self._compute_metrics(weights, C, R, array_signal)
        
        return BeamformingOutput(
            weights=weights.flatten(),
            output_signal=output_signal,
            output_sinr_db=metrics["sinr_db"],
            jammer_suppression_db=metrics["jammer_suppression_db"],
            signal_distortion_db=metrics["signal_distortion_db"]
        )

    def _rls(
        self,
        array_signal: np.ndarray,
        signal_doa_deg: float,
        forgetting_factor: float,
        delta: float,
        reference_signal: Optional[np.ndarray]
    ) -> BeamformingOutput:
        """
        RLS (Recursive Least Squares) 自适应算法

        算法原理:
        通过递归更新逆相关矩阵P(n)来最小化加权最小二乘代价函数，
        利用矩阵求逆引理避免直接求逆，实现快速收敛。

        更新规则:
        P(n) = (P(n-1) - K(n)*x^H(n)*P(n-1)) / λ
        K(n) = P(n-1)*x(n) / (λ + x^H(n)*P(n-1)*x(n))
        w(n) = w(n-1) + K(n)*e*(n)

        参数含义:
        - λ (遗忘因子): 控制历史数据权重，0 < λ ≤ 1，默认0.99
        - δ (初始化参数): P(0) = δ^(-1)*I，默认0.01
        - K(n): 增益向量，决定权重更新步长
        - e(n): 误差信号，参考信号与输出的差值

        时间复杂度:
        - 每次迭代: O(M^2)，其中M为阵元数
        - 总复杂度: O(N*M^2)，N为样本数
        - 比LMS的O(N*M)高，但收敛速度显著更快
        """
        num_samples = array_signal.shape[1]

        # 期望信号导向矢量
        a = self.array.steering_vector(signal_doa_deg)

        # 初始化权重
        weights = a.copy() / self.num_elements

        # 初始化逆相关矩阵 P(0) = δ^(-1)*I
        P = (1.0 / delta) * np.eye(self.num_elements, dtype=np.complex128)

        # 如果没有参考信号，使用导向矢量约束
        if reference_signal is None:
            reference_signal = np.ones(num_samples, dtype=np.complex128)

        # 输出信号和收敛曲线
        output_signal = np.zeros(num_samples, dtype=np.complex128)
        convergence = np.zeros(num_samples)

        # RLS迭代
        for n in range(num_samples):
            x_n = array_signal[:, n:n+1]  # 输入向量 (N_elements x 1)

            # 输出
            y_n = (weights.conj().T @ x_n).item()
            output_signal[n] = y_n

            # 误差
            e_n = reference_signal[n] - y_n
            convergence[n] = np.abs(e_n) ** 2

            # 计算增益向量 K(n) = P(n-1)*x(n) / (λ + x^H(n)*P(n-1)*x(n))
            Px = P @ x_n
            denominator = forgetting_factor + (x_n.conj().T @ Px).item()
            K = Px / denominator

            # 更新逆相关矩阵 P(n) = (P(n-1) - K(n)*x^H(n)*P(n-1)) / λ
            P = (P - K @ x_n.conj().T @ P) / forgetting_factor

            # 确保P矩阵保持厄米特对称性 (数值稳定性)
            P = (P + P.conj().T) / 2

            # 更新权重 w(n) = w(n-1) + K(n)*e*(n)
            weights = weights + K * np.conj(e_n)

            # 约束: 保持期望方向增益为1
            constraint_error = 1.0 - (weights.conj().T @ a).item()
            weights = weights + constraint_error * a / (a.conj().T @ a)

        # 计算最终协方差矩阵用于性能评估
        R = self.array.compute_covariance(array_signal)
        metrics = self._compute_metrics(weights, a, R, array_signal)

        return BeamformingOutput(
            weights=weights.flatten(),
            output_signal=output_signal,
            output_sinr_db=metrics["sinr_db"],
            jammer_suppression_db=metrics["jammer_suppression_db"],
            signal_distortion_db=metrics["signal_distortion_db"],
            convergence_curve=convergence
        )

    def _compute_metrics(
        self,
        weights: np.ndarray,
        steering_vector: np.ndarray,
        covariance: np.ndarray,
        array_signal: np.ndarray
    ) -> dict:
        """
        计算波束形成性能指标
        """
        w = weights.flatten()
        a = steering_vector.flatten()
        
        # 处理可能的NaN/Inf权重
        if not np.all(np.isfinite(w)):
            w = np.nan_to_num(w, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 输出功率
        output_power = np.real(w.conj() @ covariance @ w)
        
        # 期望方向增益
        signal_gain = np.abs(w.conj() @ a) ** 2
        
        # 信号失真 (期望增益偏离1的程度)
        gain_val = np.abs(w.conj() @ a)
        if gain_val < 1e-10:
            gain_val = 1e-10
        signal_distortion_db = 20 * np.log10(gain_val)
        signal_distortion_db = np.clip(signal_distortion_db, -60, 60)
        
        # 估计SINR (使用输出信号的统计特性)
        output_signal = self.array.apply_weights(array_signal, weights)
        output_var = np.var(output_signal)
        if output_var < 1e-20:
            output_var = 1e-20
        
        # 简化的SINR估计
        # 假设期望信号功率与输出方差成正比
        sinr_linear = signal_gain / output_var * np.mean(np.abs(output_signal) ** 2)
        if sinr_linear < 1e-10:
            sinr_linear = 1e-10
        sinr_db = 10 * np.log10(sinr_linear)
        
        # 限制合理范围
        sinr_db = np.clip(sinr_db, -20, 60)
        
        # 干扰抑制估计 (通过比较输入输出功率)
        input_power = np.mean(np.abs(array_signal) ** 2)
        if input_power < 1e-20:
            input_power = 1e-20
        jammer_suppression_db = 10 * np.log10(input_power / output_var)
        jammer_suppression_db = np.clip(jammer_suppression_db, 0, 60)
        
        # 确保所有值都是有限的
        sinr_db = float(np.nan_to_num(sinr_db, nan=0.0, posinf=60.0, neginf=-20.0))
        jammer_suppression_db = float(np.nan_to_num(jammer_suppression_db, nan=0.0, posinf=60.0, neginf=0.0))
        signal_distortion_db = float(np.nan_to_num(signal_distortion_db, nan=0.0, posinf=60.0, neginf=-60.0))
        
        return {
            "sinr_db": sinr_db,
            "jammer_suppression_db": jammer_suppression_db,
            "signal_distortion_db": signal_distortion_db
        }
