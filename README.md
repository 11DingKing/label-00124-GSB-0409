# GNSS授时仿真系统

基于自适应阵列天线抗干扰处理的GNSS授时偏差分析仿真平台

## How to Run

```bash
# 构建并启动服务
docker compose up --build -d

# 查看服务状态
docker compose ps

# 停止服务
docker compose down
```

## Services

| 服务 | 端口 | 描述 |
|------|------|------|
| backend | 8081 | GNSS授时仿真API服务 |

**API端点:**
- `GET http://localhost:8081/health` - 健康检查
- `GET http://localhost:8081/docs` - Swagger API文档
- `POST http://localhost:8081/api/v1/simulate` - 执行仿真
- `GET http://localhost:8081/api/v1/results/{id}` - 查询结果

## 测试账号

本项目为纯后端仿真服务，无需登录认证。

**快速测试:**
```bash
# 健康检查
curl http://localhost:8081/health

# 执行仿真
curl -X POST http://localhost:8081/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "num_satellites": 6,
    "array_elements": 4,
    "jammer_type": "cw",
    "jnr_db": 20,
    "algorithm": "mvdr"
  }'
```

## 题目内容
使用python利用开源代码仓，帮我构建一个GNSS授时的仿真项目，要求使用开源的GNSS信号，使用阵列天线接收，能够看到因为自适应阵列天线的抗干扰处理而引起的码相位和载波相位偏差，以及引起的伪距和授时的偏差，GNSS信号和干扰信号要求使用具有代表性的开源的

使用Python利用开源代码仓，构建一个GNSS授时的仿真项目：
- 使用开源的GNSS信号生成
- 使用阵列天线接收
- 能够看到因为自适应阵列天线的抗干扰处理而引起的：
  - 码相位偏差
  - 载波相位偏差
  - 伪距偏差
  - 授时偏差
- GNSS信号和干扰信号使用具有代表性的开源方案

## 项目介绍

### 功能实现

本项目实现了完整的GNSS授时仿真流程：

1. **GNSS信号生成**: 基于GPS L1 C/A码(1575.42MHz)，参考开源项目 [gps-sdr-sim](https://github.com/osqzss/gps-sdr-sim)
2. **干扰信号仿真**: 支持4种典型干扰类型（宽带噪声、窄带CW、扫频、脉冲）
3. **阵列天线模型**: 均匀线阵(ULA)，支持4-8阵元配置
4. **自适应波束形成**: 实现MVDR、LMS、PI、LCMV四种抗干扰算法
5. **偏差分析**: 计算码相位、载波相位、伪距、授时偏差

### 技术栈

- Python 3.11 + FastAPI
- NumPy/SciPy (信号处理)
- gnss-lib-py (Stanford NAV Lab开源GNSS库)

### 开源参考

| 项目 | 用途 |
|------|------|
| [gps-sdr-sim](https://github.com/osqzss/gps-sdr-sim) | GPS信号生成算法参考 |
| [gnss-lib-py](https://github.com/Stanford-NavLab/gnss_lib_py) | GNSS解算库 |
| [gnss-sdr](https://github.com/gnss-sdr/gnss-sdr) | SDR算法参考 |

### 仿真结果示例

执行仿真后返回的偏差分析结果：
- 码相位偏差: ~0.2-0.5 chips
- 载波相位偏差: ~0.04 cycles  
- 伪距偏差: ~70-140 meters
- 授时偏差: ~150-470 nanoseconds
