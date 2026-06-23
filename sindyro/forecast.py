"""
sindyro.forecast —— 高层封装：构建 SN-XRO 模型、做实时预报、绘制预报图。

约定
----
* 输入数据为月度气候指数距平（anomaly），第一列推荐为 Nino34，第二列为 WWV。
* 训练数组的第 0 个样本必须是「1 月」，季节相位才能与真实日历对齐。
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import pysindy as ps
from pysindy.differentiation import FiniteDifference

from .core import (
    SeasonalNonlinearLibrary,
    HybridOptimizer,
    predict_n_steps_rk4,
)

# ENSO 业务里 Nino34 这 10 个指数的标准顺序（与随附 .nc 文件一致）
DEFAULT_VARS = ['Nino34', 'WWV', 'NPMM', 'SPMM', 'IOB',
                'TNA', 'ATL3', 'IOD', 'SIOD', 'SASD']

MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# ---------------------------------------------------------------------------
# 1. 读取数据
# ---------------------------------------------------------------------------
def load_indices(nc_path, var_names=None, start='1979-01', end=None):
    """
    读取气候指数 NetCDF，切片到指定时间段并堆叠成训练数组。

    Returns
    -------
    X : ndarray (n_months, n_vars)   训练用状态矩阵
    dates : DatetimeIndex (n_months,) 每个样本对应的月份（原始时间戳）
    var_names : list[str]            实际使用的变量名
    """
    if var_names is None:
        var_names = DEFAULT_VARS
    ds = xr.open_dataset(nc_path).sel(time=slice(start, end))
    X = np.column_stack([ds[v].values for v in var_names])
    dates = pd.DatetimeIndex(ds['time'].values)

    if dates[0].month != 1:
        print(f"[load_indices] 警告：起始月份是 {dates[0]:%Y-%m} 而非 1 月，"
              f"季节相位可能与日历错位。建议 start 设为某年 1 月。")
    print(f"[load_indices] {var_names[0]}..{var_names[-1]}  "
          f"{dates[0]:%Y-%m} → {dates[-1]:%Y-%m}  共 {len(dates)} 个月")
    return X, dates, var_names


# ---------------------------------------------------------------------------
# 2. 构建并训练 SN-XRO 模型
# ---------------------------------------------------------------------------
def build_snxro_model(var_names, ac_order=2,
                      threshold=0.02, alpha_linear=0.0, alpha_nonlinear=10.0):
    """
    构建 Sparse-Nonlinear XRO（SN-XRO）模型。

    结构：
      * 线性块：所有指数 × 季节调制（ac_order 阶傅里叶），用 Ridge 拟合。
      * 非线性块：仅 Nino34、WWV 两个方程含二次项（nth_only=True），用 STLSQ 稀疏回归。

    Returns
    -------
    model : ps.SINDy   尚未 fit 的模型。调用 model.fit(X, t=1.0, feature_names=var_names) 训练。
    """
    n_vars = len(var_names)
    n_seasonal_terms = {0: 1, 1: 3, 2: 5, 3: 7}[ac_order]

    feature_library = SeasonalNonlinearLibrary(
        ac_order=ac_order,
        include_quadratic=True,
        include_cubic_self=False,
        include_cubic_couple=False,
        nth_only=True,          # 只有 Nino34 / WWV 方程启用非线性项
        mask_nt=None,
        var_names=var_names,
    )
    optimizer = HybridOptimizer(
        n_vars=n_vars,
        n_seasonal_terms=n_seasonal_terms,
        threshold=threshold,
        alpha_linear=alpha_linear,
        alpha_nonlinear=alpha_nonlinear,
    )
    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=optimizer,
        differentiation_method=FiniteDifference(order=1),
    )
    return model


# XRO 原型在 Niño3.4 / WWV 方程中规定的非线性项。
XRO_MASK_STD = {
    'Nino34': ['Nino34*Nino34', 'Nino34*WWV', 'Nino34^3'],
    'WWV':    ['WWV*WWV', 'Nino34*Nino34', 'WWV^3'],
}


def build_xro_model(var_names, ac_order=2,
                    threshold=0.01, alpha_linear=0.0, alpha_nonlinear=10.0):
    """
    构建标准 XRO（Extended nonlinear Recharge Oscillator）基线模型，作为 SN-XRO 的对照。

    与 SN-XRO 不同，XRO 的非线性项是**物理预设**的：仅 Niño3.4、WWV 方程引入
    XRO 原型规定的少数二次 / 三次自身项（见 ``XRO_MASK_STD``），其余方程纯线性。

    Returns
    -------
    model : ps.SINDy   尚未 fit 的模型。调用 model.fit(X, t=1.0, feature_names=var_names) 训练。
    """
    n_vars = len(var_names)
    n_seasonal_terms = {0: 1, 1: 3, 2: 5, 3: 7}[ac_order]

    feature_library = SeasonalNonlinearLibrary(
        ac_order=ac_order,
        include_quadratic=True,
        include_cubic_self=True,
        include_cubic_couple=False,
        nth_only=False,
        mask_nt=XRO_MASK_STD,      # 物理预设的 XRO 非线性项
        var_names=var_names,
    )
    optimizer = HybridOptimizer(
        n_vars=n_vars,
        n_seasonal_terms=n_seasonal_terms,
        threshold=threshold,
        alpha_linear=alpha_linear,
        alpha_nonlinear=alpha_nonlinear,
    )
    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=optimizer,
        differentiation_method=FiniteDifference(order=1),
    )
    return model


# ---------------------------------------------------------------------------
# 3. 实时预报（从最新一个月向前外推）
# ---------------------------------------------------------------------------
def realtime_forecast(model, X, dates, horizon=20):
    """
    以最新观测 X[-1] 为初值，用 RK4 把动力系统向前积分 `horizon` 个月。

    Returns
    -------
    fcst : ndarray (horizon+1, n_vars)   预报轨迹，fcst[0] == X[-1]（初值，即起报月）
    fcst_dates : DatetimeIndex (horizon+1,)   预报轨迹对应的月份（月初，含起报月）
    """
    start_idx = dates[-1].month - 1     # 初值所在月份的季节索引（0=1月），RK4 内部以 12 为周期
    fcst = predict_n_steps_rk4(model, X[-1], start_idx=start_idx, n_steps=horizon)
    init_month = pd.Timestamp(year=dates[-1].year, month=dates[-1].month, day=1)
    fcst_dates = pd.date_range(init_month, periods=horizon + 1, freq='MS')
    return fcst, fcst_dates


# ===========================================================================
# 4. 绘图
# ===========================================================================
def _to_month_start(ts):
    return pd.Timestamp(ts).to_period('M').to_timestamp()


def _month_year_label(ts, base_month):
    return f"{MONTH_ABBR[ts.month - 1]}({ts.year - base_month.year})"


def _date_label(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.year}-{MONTH_ABBR[ts.month - 1]}-{ts.day:02d}"


def _densify_time_series(times, values, freq='D'):
    dense_times = pd.date_range(times[0], times[-1], freq=freq)
    dense_values = np.interp(dense_times.asi8, times.asi8, values)
    return dense_times, dense_values


def _anomaly_color(cmap, value, scale):
    signed = np.clip(value / scale, -1, 1)
    norm = 0.5 + 0.5 * np.sign(signed) * (0.35 + 0.65 * abs(signed))
    return cmap(np.clip(norm, 0, 1))


def _shade_enso_lanina(ax, times, values, alpha=0.52):
    dense_times, dense_values = _densify_time_series(times, values)
    cmap = plt.get_cmap('RdBu_r')
    scale = max(2.5, np.nanmax(np.abs(values)))

    for i, lower in enumerate(np.arange(0.5, max(0.5, np.nanmax(dense_values)) + 0.5, 0.5)):
        upper = lower + 0.5
        ax.fill_between(
            dense_times, lower, np.minimum(dense_values, upper), where=dense_values >= lower,
            color=_anomaly_color(cmap, lower, scale), alpha=alpha, linewidth=0,
            label='El Nino months' if i == 0 else None)

    for i, upper in enumerate(-np.arange(0.5, max(0.5, abs(np.nanmin(dense_values))) + 0.5, 0.5)):
        lower = upper - 0.5
        ax.fill_between(
            dense_times, np.maximum(dense_values, lower), upper, where=dense_values <= upper,
            color=_anomaly_color(cmap, upper, scale), alpha=alpha, linewidth=0,
            label='La Nina months' if i == 0 else None)


def _annotate_extreme(ax, times, values, mask, pick, color, prefer='above'):
    if not np.any(mask):
        return

    masked_idx = np.where(mask)[0]
    idx = masked_idx[pick(values[masked_idx])]
    value = values[idx]
    ylim_low, ylim_high = ax.get_ylim()
    yrange = ylim_high - ylim_low
    margin = 0.08 * yrange
    gap = 0.07 * yrange

    if prefer == 'above':
        text_y = value + gap
        va = 'bottom'
        if text_y > ylim_high - margin:
            text_y = max(value - gap, ylim_low + margin)
            va = 'top'
    else:
        text_y = value - gap
        va = 'top'
        if text_y < ylim_low + margin:
            text_y = min(value + gap, ylim_high - margin)
            va = 'bottom'

    # 横向偏移文字，使箭头略微倾斜，避开标注点处（垂直的）误差棒
    span = times[-1] - times[0]
    x_off = span * 0.045
    if idx > len(times) * 0.75:        # 靠近右端则朝左偏
        text_x, ha = times[idx] - x_off, 'right'
    else:                              # 否则朝右偏
        text_x, ha = times[idx] + x_off, 'left'

    ax.annotate(
        f'{value:.2f}', xy=(times[idx], value), xytext=(text_x, text_y),
        textcoords='data', ha=ha, va=va, fontsize=13, color=color,
        arrowprops=dict(arrowstyle='->', color=color, linewidth=0.7, shrinkA=2, shrinkB=2))


def _add_past_future_arrows(ax, base_month, ylim_low, ylim_high):
    yrange = ylim_high - ylim_low
    arrow_y = ylim_low + yrange * 0.22
    label_y = arrow_y + yrange * 0.04
    past_start, past_end = base_month - pd.DateOffset(months=1), base_month - pd.DateOffset(days=8)
    future_start, future_end = base_month + pd.DateOffset(days=8), base_month + pd.DateOffset(months=1)

    arrow_style = dict(arrowstyle='-|>', color='#374151', linewidth=1.8, mutation_scale=13, shrinkA=0, shrinkB=0)
    ax.annotate('', xy=(past_start, arrow_y), xytext=(past_end, arrow_y), arrowprops=arrow_style)
    ax.annotate('', xy=(future_end, arrow_y), xytext=(future_start, arrow_y), arrowprops=arrow_style)
    ax.text(past_start + (past_end - past_start) / 2, label_y, 'past', ha='center', va='bottom', fontsize=13, color='#374151')
    ax.text(future_start + (future_end - future_start) / 2, label_y, 'future', ha='center', va='bottom', fontsize=13, color='#374151')


def plot_realtime_forecast(obs_dates, obs_vals, forecast_times, forecast_vals,
                           start, target='Niño3.4', model_name='SN-XRO',
                           history_months=6, shade_events=True,
                           forecast_err=None, title=None, save_path=None):
    """
    绘制 ENSO 实时预报图。

    Parameters
    ----------
    obs_dates, obs_vals : 完整观测的月份与目标指数值（函数内部取最近 history_months+1 个月展示）。
    forecast_times, forecast_vals : realtime_forecast 返回的预报月份与目标指数值（[0] 为起报月初值）。
    start : 起报的原始时间戳（用于标题日期，如 2026-05-16）。
    target : 目标指数显示名。
    model_name : 预报模型名（图例 / 标题）。
    history_months : 图中展示的历史观测月数。
    forecast_err : array-like or None
        与 forecast_vals 等长的逐 lead 误差幅度（如验证集 RMSE），以 errorbar 形式标注在
        预报曲线上。约定 forecast_err[0]（起报月初值）为 0，forecast_err[k] 对应 lead k。
        None 则不画误差棒。
    save_path : 保存路径；None 则不保存。
    """
    start = pd.Timestamp(start)
    base_month = _to_month_start(start)
    if title is None:
        title = f'{model_name} real-time {target} forecast ({_date_label(start)})'

    # 观测时间统一到「月初」
    obs_dates = pd.DatetimeIndex(obs_dates).to_period('M').to_timestamp()
    forecast_times = pd.DatetimeIndex(forecast_times)

    hist_times = obs_dates[-(history_months + 1):]
    hist_vals = np.asarray(obs_vals)[-(history_months + 1):]
    past_mask = hist_times < base_month

    plot_times = hist_times[past_mask].append(forecast_times)
    mean_vals = np.concatenate([hist_vals[past_mask], np.asarray(forecast_vals)])
    y_vals = mean_vals.copy()

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=140)
    if shade_events:
        _shade_enso_lanina(ax, plot_times, mean_vals)

    ax.plot(hist_times, hist_vals, color='#111827', linewidth=1.8, marker='o', markersize=3.5, label='Observed')
    ax.plot(forecast_times, forecast_vals, color='#0072B2', linewidth=2.2,
            marker='o', markersize=4.5, markerfacecolor='white', markeredgewidth=1.2,
            label=f'{model_name} forecast')

    if forecast_err is not None:
        forecast_err = np.abs(np.asarray(forecast_err, dtype=float))
        ax.errorbar(forecast_times, forecast_vals, yerr=forecast_err,
                    fmt='none', ecolor='#0072B2', elinewidth=1.2,
                    capsize=2.8, capthick=1.1, zorder=2.5,
                    label='±RMSE (validation)')

    ax.axhline(0, color='#111827', linewidth=0.7, linestyle=':')
    ax.axhline(0.5, color='#d73027', linewidth=0.7, linestyle='--', alpha=0.75)
    ax.axhline(-0.5, color='#4575b4', linewidth=0.7, linestyle='--', alpha=0.75)
    ax.axvline(base_month, color='#111827', linewidth=0.8, linestyle='--', alpha=0.7)

    ymin, ymax = np.nanmin(y_vals), np.nanmax(y_vals)
    if forecast_err is not None:
        ymin = min(ymin, np.nanmin(np.asarray(forecast_vals) - forecast_err))
        ymax = max(ymax, np.nanmax(np.asarray(forecast_vals) + forecast_err))
    y_pad = max(0.25, 0.12 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - y_pad, ymax + y_pad)
    _add_past_future_arrows(ax, base_month, ymin - y_pad, ymax + y_pad)

    _annotate_extreme(ax, plot_times, mean_vals, mean_vals >= 0.5, np.argmax, '#a50026', 'above')
    _annotate_extreme(ax, plot_times, mean_vals, mean_vals <= -0.5, np.argmin, '#313695', 'below')

    tick_times = pd.date_range(plot_times[0], plot_times[-1], freq='MS')
    ax.set_xticks(tick_times)
    ax.set_xticklabels([_month_year_label(ts, base_month) for ts in tick_times], rotation=45, ha='right', fontsize=12)

    ax.set_title(title, fontsize=16)
    ax.set_ylabel(f'{target} Index (°C)', fontsize=14)
    ax.set_xlabel('Month', fontsize=14)
    ax.set_xlim(plot_times[0], plot_times[-1])

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(axis='both', width=0.6, length=3, colors='#374151', labelsize=12)
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.6)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#d1d5db', framealpha=0.92, ncol=2, fontsize=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f"[plot] 已保存预报图 → {save_path}")
    return fig, ax
