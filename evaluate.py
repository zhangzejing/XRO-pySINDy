"""
evaluate.py —— SN-XRO 最小评估程序。

数据/划分：
  * 数据：data/XRO_indices_oras5.nc（1979-01 ~ 2024-12，10 个气候指数）。
  * 训练：前 12×23 = 276 个月（1979-01 ~ 2001-12）。
  * 验证：其余（2002-01 ~ 2024-12）。

做三件事：
  1. 在训练集上拟合 **SN-XRO**（Niño3.4/WWV 全部二次项 + STLSQ 稀疏选择）
     与 **XRO**（物理预设的少数二次/三次自身项）两个模型。
  2. 在验证集上滚动外推（滚动 1 个月 = lead 1，依此类推），逐 lead 计算 Niño3.4 的
     相关系数与 RMSE（3 个月滑动平均，is_mv3=True），绘制对比曲线 → figures/skill_comparison.png。
  3. 用**最新数据集**（data/oras5_indices_1958-2026-05.nc）重训 SN-XRO 做实时预报，把验证集
     逐 lead RMSE 以 **errorbar** 标注在预报曲线上 → figures/realtime_<target>_errorbar_<YYYYMM>.png。

用法：
    python evaluate.py
    python evaluate.py --target Nino34 --max-lead 19
"""
import argparse
import numpy as np

from sindyro import (
    load_indices,
    build_snxro_model,
    build_xro_model,
    realtime_forecast,
    plot_realtime_forecast,
)
from sindyro.core import batch_predict_rk4, evaluate_skill, plot_skill_comparison

DATA = 'data/XRO_indices_oras5.nc'                       # 训练/验证用数据集（1979–2024）
FORECAST_DATA = 'data/oras5_indices_1958-2026-05.nc'     # 实时预报：最新数据集
DISPLAY = {'Nino34': 'Niño3.4', 'WWV': 'WWV'}


def _fit(model, X, var_names):
    model.fit(X, t=1.0, feature_names=var_names)
    return model


def _rolling_skill(model, X_val, lead_times, var_idx, start_month, is_mv3=True):
    """验证集滚动外推 + 逐 lead 技巧（lead k = 向前 k 个月；is_mv3 按 ENSO 惯例做 3 月滑动平均）。"""
    pred = batch_predict_rk4(model, X_val, lead_times, start_month=start_month)
    model.feature_library._batch_start_months = None      # 复位，避免污染后续单步预报
    corr, rmse = evaluate_skill(X_val, pred, lead_times, var_idx=var_idx, is_mv3=is_mv3)
    return np.asarray(corr), np.asarray(rmse)


def main():
    p = argparse.ArgumentParser(description='SN-XRO vs XRO 技巧评估')
    p.add_argument('--data', default=DATA, help='训练/验证用指数 NetCDF 路径')
    p.add_argument('--forecast-data', default=FORECAST_DATA, help='实时预报用最新指数 NetCDF 路径')
    p.add_argument('--target', default='Nino34', help='评估的目标指数')
    p.add_argument('--start', default='1979-01', help="数据起始月 'YYYY-MM'（建议 1 月）")
    p.add_argument('--train-years', type=int, default=23, help='训练年数（n_train = 12×train_years）')
    p.add_argument('--max-lead', type=int, default=19, help='评估的最大提前期（月）')
    p.add_argument('--raw-monthly', action='store_true',
                   help='关闭 3 个月滑动平均，按逐月原始序列评估技巧')
    args = p.parse_args()
    is_mv3 = not args.raw_monthly
    j_disp = DISPLAY.get(args.target, args.target)
    lead_times = list(range(0, args.max_lead + 1))        # 含 lead 0（起报月）

    # ---- 1. 读取并按索引切分（n_train = 12 × train_years）----
    X, dates, var_names = load_indices(args.data, start=args.start)
    j = var_names.index(args.target)
    n_train = 12 * args.train_years
    X_tr, X_va = X[:n_train], X[n_train:]
    dates_tr, dates_va = dates[:n_train], dates[n_train:]
    val_start_month = dates_va[0].month - 1               # 验证集首月的季节相位（0=1 月）
    print(f"训练 {dates_tr[0]:%Y-%m}~{dates_tr[-1]:%Y-%m}（{len(X_tr)} 月） | "
          f"验证 {dates_va[0]:%Y-%m}~{dates_va[-1]:%Y-%m}（{len(X_va)} 月）")

    # ---- 2. 训练两个模型 ----
    print('\n=== 训练 SN-XRO（全部二次项 + STLSQ 稀疏选择）===')
    snxro = _fit(build_snxro_model(var_names), X_tr, var_names)
    print('\n=== 训练 XRO（物理预设非线性项基线）===')
    xro = _fit(build_xro_model(var_names), X_tr, var_names)

    # ---- 3. 验证集滚动技巧 ----
    print('\n=== 验证集滚动外推 ===')
    corr_sn, rmse_sn = _rolling_skill(snxro, X_va, lead_times, j, val_start_month, is_mv3)
    corr_xr, rmse_xr = _rolling_skill(xro, X_va, lead_times, j, val_start_month, is_mv3)
    results = {
        'XRO':    {'corr': corr_xr, 'rmse': rmse_xr},
        'SN-XRO': {'corr': corr_sn, 'rmse': rmse_sn},
    }
    for name, d in results.items():
        print(f"  {name:7s} lead{lead_times[1]}: corr={d['corr'][1]:.3f} rmse={d['rmse'][1]:.3f}  |  "
              f"lead{args.max_lead}: corr={d['corr'][-1]:.3f} rmse={d['rmse'][-1]:.3f}")

    # ---- 4. 技巧对比曲线 ----
    styles = {
        'XRO':    {'color': 'deepskyblue', 'marker': 'o', 'linewidth': 2,  'markersize': 6, 'alpha': 0.85},
        'SN-XRO': {'color': 'orangered',   'marker': 'x', 'linewidth': 2.,  'markersize': 8, 'alpha': 1.0},
    }
    plot_skill_comparison(
        lead_times, results, styles_dict=styles, figsize=(12, 3),
        title=f'Out-of-sample {j_disp} forecast skill '
              f'({dates_va[0].year}-{dates_va[-1].year})',
        save_path='figures/skill_comparison.png',
    )

    # ---- 5. 用最新数据重训 SN-XRO + 实时预报 + RMSE errorbar ----
    print('\n=== 用最新数据重训 SN-XRO 并实时预报 ===')
    X_rt, dates_rt, _ = load_indices(args.forecast_data, var_names=var_names, start=args.start)
    snxro_rt = _fit(build_snxro_model(var_names), X_rt, var_names)
    fcst, fcst_dates = realtime_forecast(snxro_rt, X_rt, dates_rt, horizon=args.max_lead)
    err = rmse_sn[:args.max_lead + 1]              # lead 0 = 起报月（RMSE≈0），lead k = 逐 lead RMSE
    out = f'figures/realtime_{args.target}_errorbar_{dates_rt[-1]:%Y%m}.png'
    plot_realtime_forecast(
        obs_dates=dates_rt, obs_vals=X_rt[:, j],
        forecast_times=fcst_dates, forecast_vals=fcst[:, j],
        forecast_err=err, start=dates_rt[-1], target=j_disp, model_name='SN-XRO',
        save_path=out,
    )

    print('\n完成：')
    print('  figures/skill_comparison.png')
    print(f'  {out}')


if __name__ == '__main__':
    main()
