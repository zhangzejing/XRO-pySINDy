# XRO-pySINDy

基于 [PySINDy](https://github.com/dynamicslab/pysindy) 的**Extended Recharge-Oscillator（XRO）**，即 **SN-XRO，Sparse-Nonlinear Extended Recharge-Oscillator**
ENSO 实时预报工具包。

模型把 Niño3.4、WWV 等 10 个气候指数当作一个季节调制的动力系统：

- **线性块**：所有指数的线性项 × 年循环傅里叶基（`ac_order` 阶），用 Ridge 拟合。
- **非线性块**：仅 Niño3.4、WWV 两个方程引入二次项，用 STLSQ 做稀疏回归。

训练后用 RK4 把系统从最新观测向前积分，得到逐月预报。

---

## 什么是稀疏非线性模型与 SINDy

**动力系统视角。** 把每个气候指数的逐月演变看成一个动力系统：

```
dx/dt = f(x, seasonal cycles)
```

其中 `x` 是所有指数组成的状态向量，`f` 是支配它们如何随时间变化的（含季节调制的）函数。
建模的目标，就是**从历史观测数据里反推出这个 `f`**。

**为什么要"稀疏非线性"。** `f` 里除了线性项，还可能有各种非线性项（二次、三次、变量间的耦合
项……）。候选项很多，但真实物理系统中真正起作用的往往只有少数几个。如果把所有候选项都塞进去
拟合，模型会过拟合、也难以解释。**稀疏回归**（如 STLSQ，序贯阈值最小二乘）的作用，就是在一大堆
候选项里**只挑出少数几个非零系数**，得到一个既简洁可解释、又不过拟合的方程。

**SINDy 一句话。** [SINDy](https://github.com/dynamicslab/pysindy)（Sparse Identification of
Nonlinear Dynamics，稀疏非线性动力学辨识）的思路是：先构造一个**候选函数库**（线性项 + 各类
非线性项），再用稀疏回归从库里选出**最少**的项来拟合 `dx/dt`，从而"发现"支配方程。

**对应到本项目（SN-XRO）。** 本工具包基于 PySINDy，用一个自定义特征库 + 自定义优化器实现
"季节调制 + 稀疏非线性"：

- **线性块**：所有指数的线性项 × 年循环傅里叶基（`ac_order` 阶），用 **Ridge** 拟合 ——
  刻画 XRO 振子的主体动力。
- **非线性块**：仅 Niño3.4、WWV 两个方程引入二次项，用 **STLSQ 稀疏回归** 只保留少数关键项
  —— 即 **SN-XRO（Sparse-Nonlinear Extended RO）**。

---

## 快速开始

最短路径（约 30 秒跑通，用随项目附带的数据）：

```bash
# 1. 安装依赖（若已有装好 pysindy 的 conda 环境，可跳过、直接复用）
pip install -r requirements.txt

# 2. 一行跑通：读取 ORAS5 指数、训练 SN-XRO、出预报图
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

## 目录结构

```
XRO-pySINDy/
├── run_forecast.py            # 最小实时预报脚本（入口）
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
