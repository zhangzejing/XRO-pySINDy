"""
sindyro —— 基于 PySINDy 的季节调制 Recharge-Oscillator (XRO) ENSO 预报工具包。

模块组成
--------
core      : 引擎。自定义特征库 / 优化器 / RK4 积分器 / 技巧评估。
forecast  : 高层封装。一行构建 SN-XRO 模型、做实时预报、绘制预报图。

最常用的入口
-----------
    from sindyro import load_indices, build_snxro_model, realtime_forecast, plot_realtime_forecast

详见 run_forecast.py 中的最小可运行示例。
"""

from .core import *          # 特征库 / 优化器 / 积分器 / 技巧评估
from .forecast import (
    load_indices,
    build_snxro_model,
    build_xro_model,
    realtime_forecast,
    plot_realtime_forecast,
)

__all__ = [
    "load_indices",
    "build_snxro_model",
    "build_xro_model",
    "realtime_forecast",
    "plot_realtime_forecast",
]
