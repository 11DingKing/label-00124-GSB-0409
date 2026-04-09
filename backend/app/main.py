"""
GNSS授时仿真系统 - FastAPI主入口

基于自适应阵列天线抗干扰处理的GNSS授时偏差分析仿真平台

功能:
- GPS L1 C/A信号生成
- 多种干扰信号仿真 (噪声、连续波、扫频、脉冲)
- 阵列天线接收模型 (均匀线阵)
- 自适应波束形成 (MVDR、LMS、PI、LCMV)
- 偏差分析 (码相位、载波相位、伪距、授时)

开源参考:
- gps-sdr-sim (https://github.com/osqzss/gps-sdr-sim)
- gnss-lib-py (https://github.com/Stanford-NavLab/gnss_lib_py)
- gnss-sdr (https://github.com/gnss-sdr/gnss-sdr)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router

# 创建FastAPI应用
app = FastAPI(
    title="GNSS授时仿真系统",
    description="""
## 概述

基于自适应阵列天线抗干扰处理的GNSS授时偏差分析仿真平台。

## 功能特性

- **GNSS信号生成**: GPS L1 C/A码信号，支持多卫星场景
- **干扰仿真**: 宽带噪声、窄带连续波、线性扫频、脉冲干扰
- **阵列天线**: 均匀线阵(ULA)模型，4-8阵元
- **自适应算法**: MVDR、LMS、PI、LCMV
- **偏差分析**: 码相位、载波相位、伪距、授时偏差

## 开源依赖

- [gps-sdr-sim](https://github.com/osqzss/gps-sdr-sim) - GPS信号生成参考
- [gnss-lib-py](https://github.com/Stanford-NavLab/gnss_lib_py) - Stanford NAV Lab GNSS库
- NumPy/SciPy - 信号处理与阵列算法
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    print("=" * 60)
    print("GNSS授时仿真系统启动")
    print("=" * 60)
    print("API文档: http://localhost:8080/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    print("GNSS授时仿真系统关闭")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
