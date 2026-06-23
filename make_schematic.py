"""
生成「SN-XRO training」示意图（仅蓝色框内容；Arial 文本 + 默认数学字体；
三栏等高居中、对称美观；季节循环写全；用 ⊗/⊕ 写出全部特征）。
输出: forecast/xro_forecast_2026-05/snxro_training.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

plt.rcParams['font.family'] = 'Arial'            # 文本用 Arial
plt.rcParams['mathtext.fontset'] = 'cm'           # 数学保持默认字体

BLUE, ORANGE, GREEN = '#3b5bdb', '#e8590c', '#2f9e44'
INK, SUB = '#1e293b', '#475569'

OUT = 'forecast/xro_forecast_2026-05/snxro_training.png'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

fig = plt.figure(figsize=(9.4, 5.8), dpi=170)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 6.5)

# ---- 蓝色外框 ----
ax.add_patch(FancyBboxPatch((0.15, 0.15), 9.7, 6.2,
                            boxstyle="round,pad=0.02,rounding_size=0.16",
                            linewidth=1.8, edgecolor=BLUE, facecolor='#f7faff'))

# ---- 三栏等高框（对称：两侧等宽、关于 x=5 对称） ----
BY0, BY1 = 2.45, 5.55                  # 三栏统一上下边界 → 等高
B1 = (0.50, 2.70)                      # Input
B2 = (3.35, 6.65)                      # Library
B3 = (7.30, 9.50)                      # Sparse Xi
for (x0, x1), fc in [(B1, '#eef3ff'), (B2, '#eaf0ff'), (B3, '#eef3ff')]:
    ax.add_patch(FancyBboxPatch((x0, BY0), x1 - x0, BY1 - BY0,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=1.3, edgecolor=BLUE, facecolor=fc))
c1, c2, c3 = [(x0 + x1) / 2 for (x0, x1) in (B1, B2, B3)]
HY = 5.85                              # 统一表头高度

# ============================================================
# A. Input layer（含完整季节循环）
# ============================================================
ax.text(c1, HY, 'Input layer', ha='center', fontsize=15, fontweight='bold', color=INK)
inputs = ['Nino34', 'WWV', 'NPMM', r'$\vdots$', 'SASD']
for name, y in zip(inputs, np.linspace(4.95, 3.05, len(inputs))):
    if name == r'$\vdots$':
        ax.text(1.05, y, r'$\vdots$', ha='center', va='center', fontsize=16, color=BLUE)
        continue
    ax.add_patch(Circle((1.05, y), 0.13, color=BLUE, zorder=3))
    ax.text(1.30, y, name, ha='left', va='center', fontsize=12.5, color=INK)

# ============================================================
# B. Library Θ(x,t)：用 ⊕ / ⊗ 写出全部特征
# ============================================================
ax.text(c2, HY, r'Library $\Theta(\mathbf{x},t)$', ha='center', fontsize=15,
        fontweight='bold', color=INK)
ax.text(c2, 5.12, r'$(\,\mathrm{lin}\ \oplus\ \mathrm{quad}\,)\ \otimes\ \mathrm{season}$',
        ha='center', va='center', fontsize=13, color=INK)
ax.plot([B2[0] + 0.25, B2[1] - 0.25], [4.80, 4.80], color='#c7d2fe', lw=1)
ax.text(c2, 4.45, r'$\mathrm{lin}:\ x_1,\,x_2,\,\dots,\,x_{10}$',
        ha='center', va='center', fontsize=10.5, color='#1e3a8a')
ax.text(c2, 3.88, r'$\mathrm{quad}:\ \mathrm{Nino34}\!\cdot\! x_j,\ \ \mathrm{WWV}\!\cdot\! x_j$',
        ha='center', va='center', fontsize=10.5, color='#9a3412')
ax.text(c2, 3.28, r'$\mathrm{season}:\ 1,\ \sin\omega t,\ \cos\omega t,$',
        ha='center', va='center', fontsize=10.5, color=GREEN)
ax.text(c2, 2.86, r'$\sin 2\omega t,\ \cos 2\omega t$',
        ha='center', va='center', fontsize=10.5, color=GREEN)

# ============================================================
# C. Sparse Ξ
# ============================================================
ax.text(c3, HY, r'Sparse $\Xi$', ha='center', fontsize=15, fontweight='bold', color=INK)
ncol, nrow, cw, ch = 6, 8, 0.30, 0.32
gx0 = c3 - ncol * cw / 2
gy0 = 4.05 - nrow * ch / 2
rng = np.random.default_rng(3)
fill = np.zeros((nrow, ncol), dtype=int)
fill[0:3] = (rng.random((3, ncol)) < 0.6).astype(int)
fill[3:] = (rng.random((nrow - 3, ncol)) < 0.14).astype(int)
for r in range(nrow):
    for c in range(ncol):
        x = gx0 + c * cw
        y = gy0 + (nrow - 1 - r) * ch
        if fill[r, c]:
            ax.add_patch(Rectangle((x, y), cw * 0.86, ch * 0.82,
                                   facecolor=BLUE if r < 3 else ORANGE, edgecolor='none', alpha=0.9))
        else:
            ax.add_patch(Rectangle((x, y), cw * 0.86, ch * 0.82,
                                   facecolor='white', edgecolor='#d6deeb', linewidth=0.6))
ax.text(c3, 2.62, 'sparse', ha='center', fontsize=10, color=SUB, style='italic')

# ---- 栏间箭头 ----
ax.add_patch(FancyArrowPatch((B1[1] + 0.05, 4.05), (B2[0] - 0.08, 4.05),
                             arrowstyle='-|>', mutation_scale=15, linewidth=1.7, color='#475569'))
ax.add_patch(FancyArrowPatch((B2[1] + 0.05, 4.05), (B3[0] - 0.08, 4.05),
                             arrowstyle='-|>', mutation_scale=15, linewidth=1.7, color='#475569'))
ax.text((B2[1] + B3[0]) / 2, 4.55, 'Ridge\n+ STLSQ', ha='center', va='center',
        fontsize=8.5, color=INK, fontweight='bold', linespacing=1.05)

# ============================================================
# 底部：辨识出的控制方程（⊕ 求和、⊗ 二次；L(t),N(t) 季节调制）
# ============================================================
ax.text(5.0, 1.30,
        r'$\dfrac{d\mathbf{x}}{dt}=\mathbf{L}(t)\,\mathbf{x}\ \oplus\ '
        r'\mathbf{N}(t)\,[\mathbf{x}\otimes\mathbf{x}]_{\mathrm{Nino34,\,WWV}}$',
        ha='center', va='center', fontsize=18, color=INK)

fig.savefig(OUT, bbox_inches='tight', facecolor='white')
print('saved', OUT)
