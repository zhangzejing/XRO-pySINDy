"""
run_forecast.py —— XRO-pySINDy 最小实时预报脚本。

默认行为：
    1. 读取 1979-01 ~ 最新 的 10 个气候指数（ORAS5）。
    2. 训练 SN-XRO 模型（季节调制 + Nino34/WWV 二次非线性）。
    3. 以最新一个月为初值，向前预报 24 个月。
    4. 绘制实时预报图并保存到 figures/。

用法：
    python run_forecast.py                      # 默认预报 Niño3.4
    python run_forecast.py --target WWV         # 预报其它指数
    python run_forecast.py --horizon 20 --start 1979-01
"""
import argparse
import numpy as np

from sindyro import (
    load_indices,
    build_snxro_model,
    realtime_forecast,
    plot_realtime_forecast,
)

DATA = 'data/oras5_indices_1958-2026-05.nc'   # 随项目附带的实时指数文件

# 目标指数的中文/显示名映射（仅影响标题与坐标轴文字）
DISPLAY = {'Nino34': 'Niño3.4', 'WWV': 'WWV'}


def main():
    p = argparse.ArgumentParser(description='SN-XRO real-time ENSO forecast')
    p.add_argument('--data', default=DATA, help='指数 NetCDF 路径')
    p.add_argument('--target', default='Nino34', help='预报的指数名（须在变量列表中）')
    p.add_argument('--start', default='1979-01', help="训练起始月 'YYYY-MM'")
    p.add_argument('--horizon', type=int, default=20, help='预报提前期（月）')
    p.add_argument('--n-obs-show', type=int, default=6, help='图中展示的历史观测月数')
    args = p.parse_args()

    # 1. 读取数据（1979-present），第 0 列为 Nino34、第 1 列为 WWV
    X, dates, var_names = load_indices(args.data, start=args.start)

    # 2. 构建并训练 SN-XRO 模型
    model = build_snxro_model(var_names)
    model.fit(X, t=1.0, feature_names=var_names)

    # 3. 从最新一个月实时外推
    fcst, fcst_dates = realtime_forecast(model, X, dates, horizon=args.horizon)

    # 4. 取出目标指数并绘图
    j = var_names.index(args.target)
    disp = DISPLAY.get(args.target, args.target)
    out = f"figures/realtime_{args.target}_{dates[-1]:%Y%m}.png"
    plot_realtime_forecast(
        obs_dates=dates, obs_vals=X[:, j],
        forecast_times=fcst_dates, forecast_vals=fcst[:, j],
        start=dates[-1], target=disp, model_name='SN-XRO',
        history_months=args.n_obs_show, save_path=out,
    )
    peak = fcst[1:, j]
    print(f"\n预报完成：起报 {fcst_dates[0]:%Y-%m}，提前 {args.horizon} 个月。")
    print(f"  {disp} 预报区间 [{peak.min():+.2f}, {peak.max():+.2f}] °C")


if __name__ == '__main__':
    main()
