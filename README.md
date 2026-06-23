# XRO-pySINDy

基于 [PySINDy](https://github.com/dynamicslab/pysindy) 复现并扩展 [XRO 模型](https://github.com/senclimate/XRO)的 ENSO 实时预报工具包。

[**XRO**](https://github.com/senclimate/XRO)（Extended nonlinear Recharge Oscillator，
[Zhao et al., *Nature*, 2024](https://doi.org/10.1038/s41586-024-07534-6)）把 Niño3.4、WWV 等
10 个气候模态指数耦合成一个**季节调制的低阶动力系统**。本项目在其线性框架之上，用
[SINDy](https://github.com/dynamicslab/pysindy) 的**稀疏回归**仅为 Niño3.4、WWV 两个核心方程
引入少量二次非线性项，得到 **SN-XRO（Sparse-Nonlinear Extended Recharge-Oscillator）**；训练后用
RK4 把系统从最新观测向前积分，得到逐月预报。相比纯线性 XRO，SN-XRO 在**较长提前期上预报技巧略有领先**。

---

## 快速开始

最短路径（约 30 秒跑通，用随项目附带的数据）：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一行跑通：读取 ORAS5 指数、训练 SN-XRO、给出预报图
python run_forecast.py
```

默认行为：读取 1979-present 的 10 个 ORAS5 指数 → 训练 SN-XRO 模型 → 以最新一个月为初值用 RK4
向前外推 20 个月 → 在 `figures/` 下生成形如 `realtime_Nino34_202605.png` 的预报图：
观测（黑）+ 预报（蓝），El Niño / La Niña 阈值分层填色，past/future 分界，并标注预报极值。

### 常用参数

```bash
python run_forecast.py --target WWV                      # 换一个预报目标
python run_forecast.py --horizon 12 --start 1979-01      # 改提前期 / 训练起始月
python run_forecast.py --data data/XRO_indices_oras5.nc  # 换数据集
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data` | `data/oras5_indices_1958-2026-05.nc` | 指数 NetCDF 路径 |
| `--target` | `Nino34` | 预报的指数（须在变量列表中）|
| `--start` | `1979-01` | 训练起始月（建议某年 1 月，季节相位对齐）|
| `--horizon` | `20` | 预报提前期（月）|
| `--n-obs-show` | `6` | 图中展示的历史观测月数 |

---

## 稀疏非线性模型与 SINDy

**动力系统与候选库。** 把 $n=10$ 个指数的距平堆成状态向量 $\mathbf{x}(t)\in\mathbb{R}^{n}$，
其逐月演变看作一个动力系统 $\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x},t)$。[SINDy](https://github.com/dynamicslab/pysindy)
假设 $\mathbf{f}$ 可由一个**候选函数库** $\Theta$（线性、二次、……项）线性张成，且真正起作用的项**很少**：

$$\dot{\mathbf{X}} \;=\; \Theta(\mathbf{X},t)\,\Xi,
\qquad \Theta=[\,\boldsymbol{\theta}_1,\ \boldsymbol{\theta}_2,\ \dots\,],$$

其中 $\Xi$ 是**稀疏**系数矩阵——大多数候选项的系数为 $0$，从而方程既可解释、又不过拟合。

**季节调制。** 每个系数都不是常数，而是随年循环变化，用 $a=$ `ac_order` 阶傅里叶基
$\boldsymbol{\phi}(t)$ 展开（$\omega=2\pi/12$，月为单位）：

$$\boldsymbol{\phi}(t)=\big[\,1,\ \sin\omega t,\ \cos\omega t,\ \dots,\ \sin a\omega t,\ \cos a\omega t\,\big].$$

**SN-XRO 的方程。** 第 $i$ 个指数的演变写成线性块 + 二次非线性块：

$$\dot{x}_i \;=\;
\underbrace{\sum_{j}\big(\mathbf{L}_{ij}\!\cdot\!\boldsymbol{\phi}(t)\big)\,x_j}_{\text{线性块（所有方程，Ridge 拟合）}}
\;+\;
\underbrace{\sum_{p\le q}\big(\mathbf{Q}^{(i)}_{pq}\!\cdot\!\boldsymbol{\phi}(t)\big)\,x_p x_q}_{\text{二次块（仅 }i\in\{\text{Niño3.4, WWV}\}\text{，STLSQ 稀疏回归）}}$$

线性块刻画 XRO 振子的主体动力；二次块只在 Niño3.4、WWV 两个核心方程启用，引入 ENSO 的非线性反馈。

**稀疏回归（STLSQ）。** 非线性块的系数用**序贯阈值最小二乘**求解——每轮先做岭回归，再把绝对值
小于阈值 $\lambda$（`threshold`）的系数置零，迭代至收敛：

$$\hat{\Xi}=\arg\min_{\Xi}\ \tfrac12\big\|\dot{\mathbf{X}}-\Theta(\mathbf{X})\,\Xi\big\|_2^2+\alpha\|\Xi\|_2^2,
\qquad \text{并令 } |\xi|<\lambda \text{ 的项 } \xi\!\leftarrow\!0 \text{ 后重复。}$$

线性块改用普通岭回归（Ridge）保留全部线性项；二者由自定义优化器 `HybridOptimizer` 组合，
特征库为 `SeasonalNonlinearLibrary`（`nth_only=True` 时只让 Niño3.4/WWV 带非线性项），均在
[`sindyro/core.py`](sindyro/core.py) 中。

---

## 目录结构

```
XRO-pySINDy/
├── run_forecast.py            # 最小实时预报脚本入口
├── sindyro/
│   ├── __init__.py
│   ├── core.py                # 引擎：特征库 / 优化器 / RK4 积分器 / 技巧评估
│   └── forecast.py            # 高层封装：建模 / 实时预报 / 绘图
├── data/
│   ├── oras5_indices_1958-2026-05.nc   # 实时指数（默认，含至 2026-05）
│   └── XRO_indices_oras5.nc            # 备用（1979–2024）
├── figures/                   # 输出预报图
├── requirements.txt
└── README.md
```

---

## 数据说明

`data/*.nc` 为月度气候指数**距平**（anomaly）：`Nino34` 单位 °C，`WWV` 为暖水体积距平等。
若要更新到更新的月份，替换为同结构的 NetCDF 并用 `--data` 指定即可（变量名需一致）。
