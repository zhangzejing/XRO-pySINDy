"""
evaluate.py —— SN-XRO 最小评估程序（训练 1979–2002，验证 2003–至今）。

做三件事：
  1. 在训练集上分别拟合 **SN-XRO**（季节调制 + Niño3.4/WWV 二次稀疏非线性）
     与 **XRO**（纯季节调制线性基线）。
  2. 在验证集上做滚动外推（滚动 1 个月 = lead 1，2 个月 = lead 2 …），逐 lead 计算目标
     指数的 **相关系数** 与 **RMSE**，绘制两模型对比曲线 → figures/skill_comparison.png。
  3. 用全量数据（1979–至今）重训 SN-XRO 做一次实时预报，并把验证集得到的逐 lead RMSE
     以 **errorbar** 标注在预报曲线上 → figures/realtime_<target>_errorbar_<YYYYMM>.png。

用法：
    python evaluate.py
    python evaluate.py --target Nino34 --max-lead 24 --horizon 20
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

DATA = 'data/oras5_indices_1958-2026-05.nc'
DISPLAY = {'Nino34': 'Niño3.4', 'WWV': 'WWV'}


def _fit(model, X, var_names):
    model.fit(X, t=1.0, feature_names=var_names)
    return model


def _rolling_skill(model, X_val, lead_times, var_idx, start_month, is_mv3=True):
    """
    在验证集上滚动外推并逐 lead 评估技巧。

    lead 与滚动月数一一对应（lead 1 = 向前 1 个月，依此类推）。is_mv3=True 时按 ENSO
    业务惯例对各 lead 的预报/观测序列做 3 个月滑动平均后再算相关/RMSE。
    """
    pred = batch_predict_rk4(model, X_val, lead_times, start_month=start_month)
    # 复位批量季节标记，避免污染该模型后续的单步预报
    model.feature_library._batch_start_months = None
    corr, rmse = evaluate_skill(X_val, pred, lead_times, var_idx=var_idx, is_mv3=is_mv3)
    return np.asarray(corr), np.asarray(rmse)


def main():
    p = argparse.ArgumentParser(description='SN-XRO vs XRO 技巧评估')
    p.add_argument('--data', default=DATA, help='指数 NetCDF 路径')
    p.add_argument('--target', default='Nino34', help='评估的目标指数')
    p.add_argument('--train-start', default='1979-01', help="训练起始月 'YYYY-MM'（建议 1 月）")
    p.add_argument('--train-end', default='2002-12', help="训练结束月 'YYYY-MM'")
    p.add_argument('--val-start', default='2003-01', help="验证起始月 'YYYY-MM'（建议 1 月）")
    p.add_argument('--max-lead', type=int, default=24, help='评估的最大提前期（月）')
    p.add_argument('--horizon', type=int, default=20, help='实时预报提前期（月）')
    p.add_argument('--raw-monthly', action='store_true',
                   help='关闭 3 个月滑动平均，按逐月原始序列评估技巧')
    args = p.parse_args()
    is_mv3 = not args.raw_monthly

    j_disp = DISPLAY.get(args.target, args.target)
    lead_times = list(range(1, args.max_lead + 1))

    # ---- 1. 读取并切分 ----
    X_tr, dates_tr, var_names = load_indices(args.data, start=args.train_start, end=args.train_end)
    X_va, dates_va, _ = load_indices(args.data, var_names=var_names, start=args.val_start)
    j = var_names.index(args.target)
    val_start_month = dates_va[0].month - 1     # 验证集首月的季节相位（0=1 月）

    # ---- 2. 训练两个模型 ----
    print('\n=== 训练 SN-XRO（含稀疏二次非线性）===')
    snxro = _fit(build_snxro_model(var_names), X_tr, var_names)
    print('\n=== 训练 XRO（纯线性基线）===')
    xro = _fit(build_xro_model(var_names), X_tr, var_names)

    # ---- 3. 验证集滚动技巧 ----
    print('\n=== 验证集滚动外推（2003 → 至今）===')
    corr_sn, rmse_sn = _rolling_skill(snxro, X_va, lead_times, j, val_start_month, is_mv3)
    corr_xr, rmse_xr = _rolling_skill(xro, X_va, lead_times, j, val_start_month, is_mv3)

    results = {
        'XRO':    {'corr': corr_xr, 'rmse': rmse_xr},
        'SN-XRO': {'corr': corr_sn, 'rmse': rmse_sn},
    }
    for name, d in results.items():
        print(f"  {name:7s} lead1: corr={d['corr'][0]:.3f} rmse={d['rmse'][0]:.3f}  |  "
              f"lead{args.max_lead}: corr={d['corr'][-1]:.3f} rmse={d['rmse'][-1]:.3f}")

    # ---- 4. 技巧对比曲线 ----
    styles = {
        'XRO':    {'color': 'gray',      'marker': 's', 'linewidth': 2},
        'SN-XRO': {'color': 'orangered', 'marker': 'o', 'linewidth': 2.4},
    }
    plot_skill_comparison(
        lead_times, results, styles_dict=styles,
        title=f'{j_disp} forecast skill — train 1979–{dates_tr[-1].year}, '
              f'validate {dates_va[0].year}–{dates_va[-1].year}',
        save_path='figures/skill_comparison.png',
    )

    # ---- 5. 全量重训 + 实时预报 + RMSE errorbar ----
    print('\n=== 全量数据（1979 → 至今）重训 SN-XRO 并实时预报 ===')
    X_all, dates_all, _ = load_indices(args.data, var_names=var_names, start=args.train_start)
    snxro_full = _fit(build_snxro_model(var_names), X_all, var_names)
    fcst, fcst_dates = realtime_forecast(snxro_full, X_all, dates_all, horizon=args.horizon)

    # lead 0 = 起报月初值（误差 0）；lead k = 验证集 SN-XRO 的逐 lead RMSE
    err = np.concatenate([[0.0], rmse_sn[:args.horizon]])
    out = f'figures/realtime_{args.target}_errorbar_{dates_all[-1]:%Y%m}.png'
    plot_realtime_forecast(
        obs_dates=dates_all, obs_vals=X_all[:, j],
        forecast_times=fcst_dates, forecast_vals=fcst[:, j],
        forecast_err=err, start=dates_all[-1], target=j_disp, model_name='SN-XRO',
        save_path=out,
    )

    print('\n完成：')
    print('  figures/skill_comparison.png')
    print(f'  {out}')


if __name__ == '__main__':
    main()
