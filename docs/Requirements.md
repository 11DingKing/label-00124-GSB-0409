# GNSS授时仿真系统需求文档

## 1. 项目概述

### 1.1 项目目标
构建一个基于Python的GNSS授时仿真系统，用于研究自适应阵列天线抗干扰处理对GNSS信号的影响，量化分析由此引起的码相位偏差、载波相位偏差、伪距偏差和授时偏差。

### 1.2 应用场景
- GNSS接收机抗干扰算法研究
- 阵列天线对授时精度影响分析
- 自适应波束形成算法性能评估

---

## 2. 功能需求

### 2.1 GNSS信号仿真模块
| 功能 | 描述 | 开源依赖 |
|------|------|----------|
| GPS L1 C/A信号生成 | 生成标准GPS L1 C/A码信号 | gps-sdr-sim / 自实现PRN码 |
| 多卫星场景 | 支持4-12颗可见卫星 | gnss-lib-py |
| 信号参数可配置 | 载噪比、多普勒、码相位 | NumPy |

### 2.2 干扰信号仿真模块
| 干扰类型 | 描述 | 典型参数 |
|----------|------|----------|
| 宽带噪声干扰 | 高斯白噪声 | 干信比 0-40dB |
| 窄带干扰 (CW) | 单频连续波 | 频偏 ±1MHz |
| 扫频干扰 | 线性调频 | 带宽 2-10MHz |
| 脉冲干扰 | 周期脉冲 | 占空比 1-50% |

### 2.3 阵列天线接收模块
| 功能 | 描述 |
|------|------|
| 阵列几何 | 均匀线阵(ULA) 4-8阵元 |
| 阵元间距 | 半波长 (λ/2 ≈ 9.5cm @ L1) |
| 信号到达角 | 可配置入射角度 |

### 2.4 自适应抗干扰处理模块
| 算法 | 描述 | 特点 |
|------|------|------|
| MVDR | 最小方差无失真响应 | 最优性能，计算复杂 |
| LMS | 最小均方自适应 | 实时性好 |
| PI (功率反演) | 功率最小化 | 工程常用 |
| LCMV | 线性约束最小方差 | 多约束支持 |

### 2.5 偏差分析模块
| 偏差类型 | 单位 | 计算方法 |
|----------|------|----------|
| 码相位偏差 | chips | 相关峰偏移检测 |
| 载波相位偏差 | cycles/rad | 鉴相器输出分析 |
| 伪距偏差 | meters | 码相位×码片距离 |
| 授时偏差 | nanoseconds | 伪距偏差/光速 |

---

## 3. 非功能需求

### 3.1 性能需求
- 单次仿真完成时间 < 30秒
- 支持批量参数扫描
- 结果数据可导出 (JSON/CSV)

### 3.2 接口需求
- RESTful API接口
- 端口: 8080 (容器内部) → 8081 (宿主机映射)

### 3.3 部署需求
- Docker容器化部署
- 支持 ARM64 + AMD64 架构
- `docker compose up --build -d` 一键启动

---

## 4. 技术栈

### 4.1 核心依赖
```
Python 3.11+
NumPy >= 1.24
SciPy >= 1.11
FastAPI >= 0.104
Uvicorn >= 0.24
gnss-lib-py >= 1.0 (Stanford NAV Lab)
Matplotlib >= 3.8 (结果可视化)
```

### 4.2 开源参考
| 项目 | 地址 | 用途 |
|------|------|------|
| gps-sdr-sim | github.com/osqzss/gps-sdr-sim | GPS信号生成参考 |
| gnss-lib-py | github.com/Stanford-NavLab/gnss_lib_py | GNSS解算库 |
| gnss-sdr | github.com/gnss-sdr/gnss-sdr | 算法参考 |

---

## 5. API接口定义

### 5.1 仿真接口
```
POST /api/v1/simulate
```
请求参数:
- `num_satellites`: 卫星数量 (4-12)
- `array_elements`: 阵元数量 (4-8)
- `jammer_type`: 干扰类型 (noise/cw/sweep/pulse)
- `jnr_db`: 干扰噪声比 (dB)
- `algorithm`: 抗干扰算法 (mvdr/lms/pi/lcmv)

### 5.2 结果查询
```
GET /api/v1/results/{simulation_id}
```

### 5.3 健康检查
```
GET /health
```

---

## 6. 项目结构
```
124/
├── backend/                    # 后端服务
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py            # FastAPI入口
│   │   ├── api/               # API路由
│   │   ├── core/              # 核心仿真逻辑
│   │   │   ├── gnss_signal.py     # GNSS信号生成
│   │   │   ├── jammer.py          # 干扰信号生成
│   │   │   ├── array_antenna.py   # 阵列天线模型
│   │   │   ├── beamforming.py     # 自适应波束形成
│   │   │   └── bias_analysis.py   # 偏差分析
│   │   └── models/            # 数据模型
├── docs/
│   ├── Requirements.md
│   └── Roadmap.md
├── docker-compose.yml
├── .gitignore
├── README.md
└── .alkaid-sop
```

---

## 7. 验收标准

1. ✅ `docker compose up --build -d` 成功启动
2. ✅ `curl localhost:8081/health` 返回200
3. ✅ 仿真接口返回完整偏差数据
4. ✅ 支持至少4种自适应算法
5. ✅ 支持至少4种干扰类型
6. ✅ 偏差结果符合理论预期范围
