import numpy as np
import xarray as xr
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'arial'
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from pysindy.differentiation import FiniteDifference
from pysindy.utils import AxesArray


# ============================================================================
# 自定义特征库
# ============================================================================

class SeasonalNonlinearLibrary(ps.feature_library.base.BaseFeatureLibrary):
    """
    标准化特征库 + 掩码支持
    
    标准特征结构（每个变量）：
    1. 线性项: x_i * seasonal (n_seasonal_terms)
    2. 二次项: x_i*x_j * seasonal (n_vars × n_seasonal_terms)
    3. 三次自身项: x_i^3 * seasonal (n_seasonal_terms)
    4. 三次耦合项: Nino34*WWV*x_j * seasonal (n_vars × n_seasonal_terms)
    
    掩码支持：
    - nth_only: 只保留Nino34和WWV的非线性项
    - mask_nt: 自定义每个变量使用哪些非线性项
    
    Parameters:
    -----------
    ac_order : int (0, 1, 2, 3)
        季节循环阶数
    include_quadratic : bool
        是否包含二次项
    include_cubic_self : bool
        是否包含三次自身项
    include_cubic_couple : bool
        是否包含三次耦合项
    nth_only : bool
        如果为True，只保留Nino34和WWV的非线性项（覆盖mask_nt）
    mask_nt : dict or None
        自定义掩码，格式: {'Nino34': ['Nino34*WWV', 'Nino34^3'], ...}
    var_names : list of str
        变量名称
    start_month : int
        起始月份
    """
    
    def __init__(self, ac_order=2, 
                 include_quadratic=True, 
                 include_cubic_self=False,
                 include_cubic_couple=False,
                 nth_only=False,
                 mask_nt=None,
                 var_names=None, start_month=0):
        super().__init__()
        self.ac_order = ac_order
        self.include_quadratic = include_quadratic
        self.include_cubic_self = include_cubic_self
        self.include_cubic_couple = include_cubic_couple
        self.nth_only = nth_only
        self.mask_nt = mask_nt
        self.var_names = var_names
        self.start_month = start_month
        
        # 季节项数量
        if ac_order == 0:
            self.n_seasonal_terms = 1
        elif ac_order == 1:
            self.n_seasonal_terms = 3
        elif ac_order == 2:
            self.n_seasonal_terms = 5
        else:  # ac_order == 3
            self.n_seasonal_terms = 7
        
        # 找到Nino34和WWV的索引
        if var_names is not None:
            try:
                self.nino34_idx = var_names.index('Nino34')
                self.wwv_idx = var_names.index('WWV')
            except ValueError:
                print("警告: 找不到Nino34或WWV，使用前两个变量")
                self.nino34_idx = 0
                self.wwv_idx = 1
        else:
            self.nino34_idx = None
            self.wwv_idx = None
        
        # 掩码矩阵（延迟到fit时构建）
        self.nonlinear_mask_ = None
    
    def fit(self, x, y=None):
        x = np.asarray(x)
        while x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
        
        n_samples, n_features = x.shape
        self.n_input_features_ = n_features
        self.n_features_in_ = n_features
        self.n_output_features_ = self._compute_n_output_features()
        
        # 构建掩码矩阵
        self._build_mask()
        
        return self
    
    def _compute_n_output_features(self):
        """计算总特征数（标准情况，不考虑掩码）"""
        n_vars = self.n_input_features_
        n_s = self.n_seasonal_terms
        
        n_linear = n_vars * n_s
        n_quadratic = n_vars * n_vars * n_s if self.include_quadratic else 0
        n_cubic_self = n_vars * n_s if self.include_cubic_self else 0
        n_cubic_couple = n_vars * n_vars * n_s if self.include_cubic_couple else 0
        
        return n_linear + n_quadratic + n_cubic_self + n_cubic_couple
    
    def _build_mask(self):
        """构建掩码矩阵"""
        n_vars = self.n_input_features_
        n_s = self.n_seasonal_terms
        
        # 计算每个变量的非线性特征数
        n_nonlinear_per_var = 0
        if self.include_quadratic:
            n_nonlinear_per_var += n_vars * n_s
        if self.include_cubic_self:
            n_nonlinear_per_var += n_s
        if self.include_cubic_couple:
            n_nonlinear_per_var += n_vars * n_s
        
        if n_nonlinear_per_var == 0:
            self.nonlinear_mask_ = None
            return
        
        # 默认全1
        mask = np.ones((n_vars, n_nonlinear_per_var))
        
        # 应用nth_only
        if self.nth_only:
            mask[:] = 0
            mask[self.nino34_idx, :] = 1
            mask[self.wwv_idx, :] = 1
            print(f"nth_only: 只保留变量{self.nino34_idx}和{self.wwv_idx}的非线性项")
        
        # 应用mask_nt
        elif self.mask_nt is not None:
            mask[:] = 0
            print("\n应用mask_nt:")
            
            input_features = self.var_names if self.var_names else [f"x{i}" for i in range(n_vars)]
            
            for var_name, allowed_terms in self.mask_nt.items():
                if var_name not in input_features:
                    continue
                
                var_idx = input_features.index(var_name)
                print(f"  {var_name}:")
                
                idx = 0
                
                # 二次项
                if self.include_quadratic:
                    for j in range(n_vars):
                        term = f"{var_name}*{input_features[j]}"
                        if term in allowed_terms:
                            mask[var_idx, idx:idx+n_s] = 1
                            print(f"    ✓ {term}")
                        idx += n_s
                
                # 三次自身项
                if self.include_cubic_self:
                    term = f"{var_name}^3"
                    if term in allowed_terms:
                        mask[var_idx, idx:idx+n_s] = 1
                        print(f"    ✓ {term}")
                    idx += n_s
                
                # 三次耦合项
                if self.include_cubic_couple:
                    for j in range(n_vars):
                        term = f"Nino34*WWV*{input_features[j]}"
                        if term in allowed_terms:
                            mask[var_idx, idx:idx+n_s] = 1
                            print(f"    ✓ {term}")
                        idx += n_s
        
        self.nonlinear_mask_ = mask
    
    def transform(self, x):
        x_array = np.asarray(x)
        
        if x_array.ndim == 3:
            n_trajectories = x_array.shape[0]
            n_samples_per_traj = x_array.shape[1]
            x_2d = x_array.reshape(-1, x_array.shape[-1])
            result = self._transform_single(x_2d)
            result_3d = result.reshape(n_trajectories, n_samples_per_traj, -1)
            return AxesArray(result_3d, {"ax_sample": 0, "ax_time": 1, "ax_coord": 2})
        else:
            result = self._transform_single(x_array)
            return AxesArray(result, {"ax_sample": 0, "ax_coord": 1})
    
    def _transform_single(self, x):
        """
        生成特征矩阵 - 每个变量的特征连续排列
        支持批量季节信息和浮点数时间
        """
        n_samples, n_vars = x.shape
        
        # 确定每个样本的起始月份（支持浮点数和批量）
        if hasattr(self, '_batch_start_months') and self._batch_start_months is not None:
            # 批量模式：每个样本有独立的start_month
            months_array = self._batch_start_months.copy()
            if len(months_array) < n_samples:
                # 如果批量数小于样本数，扩展到所有样本
                months_array = np.tile(months_array, (n_samples // len(months_array) + 1))[:n_samples]
        else:
            # 单一模式：所有样本使用相同的start_month
            months_array = self.start_month + np.arange(n_samples, dtype=float)
        
        # 构建季节函数（支持浮点数时间）
        omega = 2 * np.pi / 12
        
        seasonal_funcs = [np.ones((n_samples, 1))]
        if self.ac_order >= 1:
            seasonal_funcs.extend([
                np.sin(omega * months_array).reshape(-1, 1),
                np.cos(omega * months_array).reshape(-1, 1)
            ])
        if self.ac_order >= 2:
            seasonal_funcs.extend([
                np.sin(2 * omega * months_array).reshape(-1, 1),
                np.cos(2 * omega * months_array).reshape(-1, 1)
            ])
        if self.ac_order >= 3:
            seasonal_funcs.extend([
                np.sin(3 * omega * months_array).reshape(-1, 1),
                np.cos(3 * omega * months_array).reshape(-1, 1)
            ])
        
        all_features = []
        
        # ========== 1. 线性项 ==========
        for i in range(n_vars):
            X_i = x[:, i:i+1]
            for seasonal_func in seasonal_funcs:
                all_features.append(X_i * seasonal_func)
        
        # ========== 2. 非线性项 - 按变量组织 ==========
        for i in range(n_vars):
            # 该变量的所有非线性特征
            
            # 2.1 二次项: x_i * x_j
            if self.include_quadratic:
                X_i = x[:, i:i+1]
                for j in range(n_vars):
                    X_j = x[:, j:j+1]
                    X_ij = X_i * X_j
                    for seasonal_func in seasonal_funcs:
                        all_features.append(X_ij * seasonal_func)
            
            # 2.2 三次自身项: x_i^3
            if self.include_cubic_self:
                X_i = x[:, i:i+1]
                X_i3 = X_i ** 3
                for seasonal_func in seasonal_funcs:
                    all_features.append(X_i3 * seasonal_func)
            
            # 2.3 三次耦合项: Nino34*WWV*x_j
            if self.include_cubic_couple:
                X_nino34 = x[:, self.nino34_idx:self.nino34_idx+1]
                X_wwv = x[:, self.wwv_idx:self.wwv_idx+1]
                X_nw = X_nino34 * X_wwv
                
                for j in range(n_vars):
                    X_j = x[:, j:j+1]
                    X_nwj = X_nw * X_j
                    for seasonal_func in seasonal_funcs:
                        all_features.append(X_nwj * seasonal_func)
        
        # 拼接所有特征
        result = np.hstack(all_features)

        # === 应用掩码 ===
        if self.nonlinear_mask_ is not None:
            n_linear = n_vars * self.n_seasonal_terms
            n_nonlinear_per_var = self.nonlinear_mask_.shape[1]
            
            for i in range(n_vars):
                start_col = n_linear + i * n_nonlinear_per_var
                end_col = start_col + n_nonlinear_per_var
                
                # 应用掩码
                result[:, start_col:end_col] = result[:, start_col:end_col] * self.nonlinear_mask_[i, :]
        
        return result

    def get_feature_names(self, input_features=None):
        """生成特征名称 - 与_transform_single顺序一致"""
        if input_features is None:
            if self.var_names is not None:
                input_features = self.var_names
            else:
                input_features = [f"x{i}" for i in range(self.n_input_features_)]
        
        n_vars = len(input_features)
        feature_names = []
        
        # 季节名称
        seasonal_names = [""]
        if self.ac_order >= 1:
            seasonal_names.extend(["*sin(ωt)", "*cos(ωt)"])
        if self.ac_order >= 2:
            seasonal_names.extend(["*sin(2ωt)", "*cos(2ωt)"])
        if self.ac_order >= 3:
            seasonal_names.extend(["*sin(3ωt)", "*cos(3ωt)"])
        
        # 1. 线性项（所有变量）
        for var in input_features:
            for season in seasonal_names:
                if season == "":
                    feature_names.append(f"{var}")
                else:
                    feature_names.append(f"{var}{season}")
        
        # 2. 非线性项（按变量循环）
        for i in range(n_vars):
            var_i = input_features[i]
            
            # 2.1 二次项: var_i * var_j
            if self.include_quadratic:
                for j in range(n_vars):
                    var_j = input_features[j]
                    for season in seasonal_names:
                        if season == "":
                            feature_names.append(f"{var_i}*{var_j}")
                        else:
                            feature_names.append(f"{var_i}*{var_j}{season}")
            
            # 2.2 三次自身项: var_i^3
            if self.include_cubic_self:
                for season in seasonal_names:
                    if season == "":
                        feature_names.append(f"{var_i}^3")
                    else:
                        feature_names.append(f"{var_i}^3{season}")
            
            # 2.3 三次耦合项: Nino34*WWV*var_j
            if self.include_cubic_couple:
                for j in range(n_vars):
                    var_j = input_features[j]
                    for season in seasonal_names:
                        if season == "":
                            feature_names.append(f"Nino34*WWV*{var_j}")
                        else:
                            feature_names.append(f"Nino34*WWV*{var_j}{season}")
        
        return feature_names
    
class SeasonalSINDyLibrary(ps.feature_library.base.BaseFeatureLibrary):
    """
    季节调制 SINDy 特征库，非线性项完全由 mask_nt 控制。

    特征布局
    --------
    [线性块] | [变量0非线性块] | [变量1非线性块] | ...

    - 线性块：所有变量的线性×季节项，各方程共享。
    - 非线性块：每个变量方程独立拥有一份完整的「项库×季节」列，
      不在 mask_nt 中的项对应列值恒为 0（列仍存在，保证系数矩阵形状固定）。

    支持的非线性项类型
    ------------------
    类型            项名格式示例               数学含义
    -------         ----------------------     -------------------
    二次项          'Nino34*WWV'               x_i · x_j
                    'Nino34*Nino34'            x_i²（写成 x_i*x_i）
    三次自身项      'Nino34^3'                 x_i³
    三次耦合项      'Nino34*WWV*Nino34'        Niño34 · WWV · x_j

    Parameters
    ----------
    ac_order : int, default=2
        季节循环阶数（0/1/2/3），对应 1/3/5/7 个季节基函数。

    nonlinear_terms : list of str or None, default=None
        **全局非线性项库**，决定非线性块中的列顺序与总列数。
        每个元素是一个项名字符串（格式见上表）。

        用法一：None（推荐，自动推断）
            - 若 mask_nt 不为 None：自动收集 mask_nt 中所有出现的项，
              按「二次→三次自身→三次耦合」标准顺序排列，只生成用到的列，
              避免不必要的稀疏列。
            - 若 mask_nt 也为 None：生成所有变量的完整二次+三次项库。

        用法二：显式指定
            手动给出项库列表，精确控制列顺序，适用于：
            (a) 多次实验需要保证不同 mask_nt 配置之间列数/顺序一致；
            (b) 只想保留部分二次项但不用 mask_nt 过滤（直接不放进库）；
            (c) 自定义项排列顺序以方便解读系数矩阵。

            示例：
                nonlinear_terms = [
                    'Nino34*Nino34',   # T²，用于 WWV 方程的非线性反馈
                    'Nino34*WWV',      # 标准双线性耦合
                    'Nino34^3',        # T 的三次饱和
                    'WWV^3',           # H 的三次饱和
                    'Nino34*WWV*Nino34',
                ]

            注意：nonlinear_terms 只定义「库中有哪些列」，
            实际哪些方程使用哪些列仍由 mask_nt 控制。
            若某项出现在 mask_nt 中却不在 nonlinear_terms 里，
            fit() 会抛出 ValueError。

    mask_nt : dict or None, default=None
        每个变量方程允许包含的非线性项。格式：
            {
                '变量名': ['项名1', '项名2', ...],
                ...
            }
        - 未出现在字典 key 中的变量方程不含任何非线性项。
        - None 表示所有变量方程包含项库中的全部项（无掩码）。

        典型示例——在 WWV 方程中添加 T² 项：
            mask_nt = {
                'Nino34': ['Nino34*WWV', 'Nino34^3'],
                'WWV':    ['Nino34*Nino34', 'Nino34*WWV', 'WWV^3'],
            }

    var_names : list of str or None
        变量名称列表，须与输入数据的列顺序一致。
        None 时自动命名为 ['x0', 'x1', ...]。

    start_month : int or float, default=0
        第一个样本对应的月份编号（0=1月）。
    """

    def __init__(self,
                 ac_order=2,
                 nonlinear_terms=None,
                 mask_nt=None,
                 var_names=None,
                 start_month=0):
        super().__init__()
        self.ac_order = ac_order
        self.nonlinear_terms = nonlinear_terms
        self.mask_nt = mask_nt
        self.var_names = var_names
        self.start_month = start_month
        self._batch_start_months = None

        if ac_order == 0:
            self.n_seasonal_terms = 1
        elif ac_order == 1:
            self.n_seasonal_terms = 3
        elif ac_order == 2:
            self.n_seasonal_terms = 5
        else:
            self.n_seasonal_terms = 7

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _get_var_names(self, n_vars):
        if self.var_names is not None:
            return list(self.var_names)
        return [f"x{i}" for i in range(n_vars)]

    def _seasonal_names(self):
        names = [""]
        if self.ac_order >= 1:
            names += ["*sin(ωt)", "*cos(ωt)"]
        if self.ac_order >= 2:
            names += ["*sin(2ωt)", "*cos(2ωt)"]
        if self.ac_order >= 3:
            names += ["*sin(3ωt)", "*cos(3ωt)"]
        return names

    def _enumerate_all_terms(self, var_names):
        """按标准顺序枚举所有可能的非线性项名称。"""
        has_nino34 = 'Nino34' in var_names
        has_wwv = 'WWV' in var_names
        terms = []
        for vi in var_names:
            for vj in var_names:
                terms.append(f"{vi}*{vj}")
        for vi in var_names:
            terms.append(f"{vi}^3")
        if has_nino34 and has_wwv:
            for vj in var_names:
                terms.append(f"Nino34*WWV*{vj}")
        return terms

    def _build_term_library(self, var_names):
        """确定最终使用的项库列表。"""
        if self.nonlinear_terms is not None:
            # 用户显式指定：直接使用，但验证格式
            return list(self.nonlinear_terms)

        if self.mask_nt is not None:
            # 从 mask_nt 收集所有出现的项，按标准顺序排列
            all_possible = self._enumerate_all_terms(var_names)
            mentioned = set()
            for terms in self.mask_nt.values():
                mentioned.update(terms)
            return [t for t in all_possible if t in mentioned]

        # 无约束：完整库
        return self._enumerate_all_terms(var_names)

    def _compute_term_value(self, term, x, var_names):
        """计算单个非线性项的值，返回 (n_samples, 1)。"""
        n = x.shape[0]

        # 三次耦合: Nino34*WWV*xj
        if term.startswith("Nino34*WWV*"):
            xj_name = term[len("Nino34*WWV*"):]
            ni = var_names.index("Nino34")
            wi = var_names.index("WWV")
            ji = var_names.index(xj_name)
            return (x[:, ni] * x[:, wi] * x[:, ji]).reshape(n, 1)

        # 三次自身: xi^3
        if term.endswith("^3"):
            vi_name = term[:-2]
            ii = var_names.index(vi_name)
            return (x[:, ii] ** 3).reshape(n, 1)

        # 二次: xi*xj
        if "*" in term:
            parts = term.split("*", 1)
            ii = var_names.index(parts[0])
            ji = var_names.index(parts[1])
            return (x[:, ii] * x[:, ji]).reshape(n, 1)

        raise ValueError(f"无法解析非线性项名称: '{term}'")

    # ------------------------------------------------------------------
    # sklearn 接口
    # ------------------------------------------------------------------

    def fit(self, x, y=None):
        x = np.asarray(x)
        while x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])

        n_samples, n_features = x.shape
        self.n_input_features_ = n_features
        self.n_features_in_ = n_features

        var_names = self._get_var_names(n_features)
        self._fitted_var_names = var_names
        self._term_library = self._build_term_library(var_names)

        # 验证：mask_nt 中的项必须全部在项库中
        if self.mask_nt is not None and self.nonlinear_terms is not None:
            lib_set = set(self._term_library)
            for eq_var, terms in self.mask_nt.items():
                missing = [t for t in terms if t not in lib_set]
                if missing:
                    raise ValueError(
                        f"变量 '{eq_var}' 的 mask_nt 包含不在 nonlinear_terms 中的项: {missing}\n"
                        f"请将这些项加入 nonlinear_terms，或将 nonlinear_terms 设为 None。"
                    )

        n_s = self.n_seasonal_terms
        n_terms = len(self._term_library)

        # 方程掩码: shape (n_vars, n_terms)，1=启用，0=置零
        self._eq_mask = np.ones((n_features, n_terms), dtype=float)
        if self.mask_nt is not None:
            self._eq_mask[:] = 0
            for eq_var, allowed_terms in self.mask_nt.items():
                if eq_var not in var_names:
                    print(f"警告: mask_nt 中的 '{eq_var}' 不在 var_names 中，已跳过")
                    continue
                eq_idx = var_names.index(eq_var)
                allowed_set = set(allowed_terms)
                for k, term in enumerate(self._term_library):
                    if term in allowed_set:
                        self._eq_mask[eq_idx, k] = 1

        self.n_output_features_ = (n_features * n_s +
                                   n_features * n_terms * n_s)

        # 打印摘要
        print(f"[SeasonalSINDyLibrary] ac_order={self.ac_order}, "
              f"季节基函数={n_s}, 项库大小={n_terms}, "
              f"总特征列={self.n_output_features_}")
        print(f"  项库: {self._term_library}")
        for i, vn in enumerate(var_names):
            active = [self._term_library[k] for k in range(n_terms)
                      if self._eq_mask[i, k] > 0]
            print(f"  {vn} 方程: {active if active else '(仅线性)'}")

        return self

    def transform(self, x):
        x_array = np.asarray(x)
        if x_array.ndim == 3:
            n_traj, n_t, n_feat = x_array.shape
            result = self._transform_single(x_array.reshape(-1, n_feat))
            return AxesArray(result.reshape(n_traj, n_t, -1),
                             {"ax_sample": 0, "ax_time": 1, "ax_coord": 2})
        result = self._transform_single(x_array)
        return AxesArray(result, {"ax_sample": 0, "ax_coord": 1})

    def _transform_single(self, x):
        n_samples, n_vars = x.shape
        var_names = self._fitted_var_names

        # 季节函数列表，每个元素 shape (n_samples, 1)
        if hasattr(self, '_batch_start_months') and self._batch_start_months is not None:
            months = np.asarray(self._batch_start_months, dtype=float)
            if len(months) < n_samples:
                months = np.tile(months, (n_samples // len(months) + 1))[:n_samples]
        else:
            months = self.start_month + np.arange(n_samples, dtype=float)

        omega = 2 * np.pi / 12
        s_funcs = [np.ones((n_samples, 1))]
        if self.ac_order >= 1:
            s_funcs += [np.sin(omega * months).reshape(-1, 1),
                        np.cos(omega * months).reshape(-1, 1)]
        if self.ac_order >= 2:
            s_funcs += [np.sin(2 * omega * months).reshape(-1, 1),
                        np.cos(2 * omega * months).reshape(-1, 1)]
        if self.ac_order >= 3:
            s_funcs += [np.sin(3 * omega * months).reshape(-1, 1),
                        np.cos(3 * omega * months).reshape(-1, 1)]

        cols = []

        # 1. 线性块
        for i in range(n_vars):
            xi = x[:, i:i+1]
            for sf in s_funcs:
                cols.append(xi * sf)

        # 2. 非线性块（预计算所有项的原始值）
        term_vals = [self._compute_term_value(t, x, var_names)
                     for t in self._term_library]

        for i in range(n_vars):
            for k, tv in enumerate(term_vals):
                masked = tv * self._eq_mask[i, k]
                for sf in s_funcs:
                    cols.append(masked * sf)

        return np.hstack(cols)

    def get_term_library(self, var_names=None):
        """
        在 fit() 之前即可调用，返回项库列表。
        用于提前获取 n_nonlinear_terms 以初始化优化器。
        """
        vnames = var_names if var_names is not None else self._get_var_names(
            len(self.var_names) if self.var_names else 2
        )
        return self._build_term_library(vnames)

    def get_feature_names(self, input_features=None):
        if input_features is None:
            input_features = self._fitted_var_names

        s_names = self._seasonal_names()
        names = []

        # 线性块
        for var in input_features:
            for sn in s_names:
                names.append(f"{var}{sn}" if sn else var)

        # 非线性块
        for eq_var in input_features:
            i = input_features.index(eq_var)
            for k, term in enumerate(self._term_library):
                active = self._eq_mask[i, k] > 0
                tag = f"[{eq_var}]" if active else f"[{eq_var},off]"
                for sn in s_names:
                    names.append(f"{tag}{term}{sn}" if sn else f"{tag}{term}")

        return names

# ============================================================================
# 自定义优化器
# ============================================================================

class HybridOptimizer(ps.optimizers.BaseOptimizer):
    """
    混合优化器：线性用Ridge，非线性用STLSQ
    
    假设特征结构：
    - 线性: n_vars × n_seasonal_terms
    - 非线性: n_vars × n_nonlinear_per_var (每个变量的非线性特征数相同)
    """
    
    def __init__(self, n_vars, n_seasonal_terms, 
                 threshold=0.01, alpha_linear=0.01, alpha_nonlinear=0.01):
        super().__init__()
        self.n_vars = n_vars
        self.n_seasonal_terms = n_seasonal_terms
        self.n_linear_features = n_vars * n_seasonal_terms
        self.threshold = threshold
        self.alpha_linear = alpha_linear
        self.alpha_nonlinear = alpha_nonlinear
    
    def _reduce(self, x, y):
        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        
        # ========== 线性部分：Ridge ==========
        X_linear = x[:, :self.n_linear_features]
        ridge = Ridge(alpha=self.alpha_linear, fit_intercept=False)
        ridge.fit(X_linear, y)
        coef_linear = ridge.coef_
        
        # 计算残差
        y_linear_pred = X_linear @ coef_linear.T
        residual = y - y_linear_pred
        
        r2_scores = [1 - np.var(residual[:, i]) / np.var(y[:, i]) for i in range(n_targets)]
        print(f"  线性拟合R²: {r2_scores}")
        
        # 初始化系数矩阵
        coef_full = np.zeros((n_targets, n_features))
        coef_full[:, :self.n_linear_features] = coef_linear
        
        # ========== 非线性部分：STLSQ ==========
        if n_features > self.n_linear_features:
            n_nonlinear_total = n_features - self.n_linear_features
            n_nonlinear_per_var = n_nonlinear_total // n_targets
            
            print(f"  每个变量的非线性特征数: {n_nonlinear_per_var}")
            
            for i in range(n_targets):
                # 变量i的非线性特征范围
                start_idx = self.n_linear_features + i * n_nonlinear_per_var
                end_idx = start_idx + n_nonlinear_per_var
                
                X_nonlinear_i = x[:, start_idx:end_idx]
                
                # 检查是否全0
                if np.max(np.abs(X_nonlinear_i)) > 1e-10:
                    residual_i = residual[:, i:i+1]
                    
                    # STLSQ拟合
                    stlsq = ps.STLSQ(threshold=self.threshold, alpha=self.alpha_nonlinear)
                    stlsq.fit(X_nonlinear_i, residual_i)
                    
                    coef_full[i, start_idx:end_idx] = stlsq.coef_[0]
                    
                    n_nonzero = np.sum(np.abs(stlsq.coef_[0]) > 1e-10)
                    print(f"    变量{i}: {n_nonzero}/{n_nonlinear_per_var} 个非零非线性项")
                else:
                    print(f"    变量{i}: 无非线性项（全0）")
        
        self.coef_ = coef_full
        return self
    
class HybridSINDyOptimizer(ps.optimizers.BaseOptimizer):
    """
    混合优化器：线性用 Ridge，非线性用 STLSQ。

    与 SeasonalSINDyLibrary 配套使用，特征布局：
        [线性块: n_vars × n_seasonal_terms]
        [变量0非线性块: n_terms × n_seasonal_terms]
        [变量1非线性块: n_terms × n_seasonal_terms]
        ...

    Parameters
    ----------
    n_vars : int
        变量数量。
    n_seasonal_terms : int
        季节基函数数量（SeasonalSINDyLibrary.n_seasonal_terms）。
    n_nonlinear_terms : int
        非线性项库大小（len(library._term_library)）。
        即每个变量非线性块的项数，块总列数 = n_nonlinear_terms × n_seasonal_terms。
    threshold : float
        STLSQ 稀疏阈值。
    alpha_linear : float
        Ridge 正则化系数（线性部分）。
    alpha_nonlinear : float
        STLSQ 正则化系数（非线性部分）。
    """

    def __init__(self, n_vars, n_seasonal_terms, n_nonlinear_terms,
                 threshold=0.01, alpha_linear=0.01, alpha_nonlinear=0.01):
        super().__init__()
        self.n_vars = n_vars
        self.n_seasonal_terms = n_seasonal_terms
        self.n_nonlinear_terms = n_nonlinear_terms
        self.threshold = threshold
        self.alpha_linear = alpha_linear
        self.alpha_nonlinear = alpha_nonlinear

        self.n_linear_features = n_vars * n_seasonal_terms
        self.n_nonlinear_per_var = n_nonlinear_terms * n_seasonal_terms  # 每变量非线性块列数

    def _reduce(self, x, y):
        n_samples, n_features = x.shape
        n_targets = y.shape[1]

        expected = self.n_linear_features + n_targets * self.n_nonlinear_per_var
        if n_features != expected:
            raise ValueError(
                f"特征列数不匹配: 期望 {expected}（线性 {self.n_linear_features} + "
                f"非线性 {n_targets} × {self.n_nonlinear_per_var}），实际 {n_features}。\n"
                f"请检查 n_nonlinear_terms 是否等于 len(library._term_library)。"
            )

        coef_full = np.zeros((n_targets, n_features))

        # ========== 线性部分：Ridge ==========
        X_linear = x[:, :self.n_linear_features]
        ridge = Ridge(alpha=self.alpha_linear, fit_intercept=False)
        ridge.fit(X_linear, y)
        coef_linear = ridge.coef_          # (n_targets, n_linear_features)
        coef_full[:, :self.n_linear_features] = coef_linear

        y_linear_pred = X_linear @ coef_linear.T
        residual = y - y_linear_pred

        r2_scores = [
            1 - np.var(residual[:, i]) / np.var(y[:, i])
            for i in range(n_targets)
        ]
        print(f"  线性拟合 R²: {[f'{v:.4f}' for v in r2_scores]}")

        # ========== 非线性部分：STLSQ（逐变量） ==========
        if n_features <= self.n_linear_features:
            self.coef_ = coef_full
            return self

        for i in range(n_targets):
            start_idx = self.n_linear_features + i * self.n_nonlinear_per_var
            end_idx   = start_idx + self.n_nonlinear_per_var

            X_nl = x[:, start_idx:end_idx]   # (n_samples, n_nonlinear_per_var)

            # 找出非零列（mask 置零的列直接跳过，不参与拟合）
            col_active = np.max(np.abs(X_nl), axis=0) > 1e-10  # (n_nonlinear_per_var,)
            n_active = col_active.sum()

            if n_active == 0:
                print(f"  变量{i}: 无非线性项（全部被掩码置零）")
                continue

            # 仅用活跃列拟合，避免全零列干扰 STLSQ 阈值判断
            X_nl_active = X_nl[:, col_active]
            residual_i  = residual[:, i:i+1]

            stlsq = ps.STLSQ(threshold=self.threshold, alpha=self.alpha_nonlinear)
            stlsq.fit(X_nl_active, residual_i)

            # 将系数写回对应的完整列位置
            coef_nl_full = np.zeros(self.n_nonlinear_per_var)
            coef_nl_full[col_active] = stlsq.coef_[0]
            coef_full[i, start_idx:end_idx] = coef_nl_full

            n_nonzero = np.sum(np.abs(stlsq.coef_[0]) > 1e-10)
            print(f"  变量{i}: {n_nonzero}/{n_active} 个活跃非线性项被保留"
                  f"（共 {self.n_nonlinear_per_var} 列，{n_active} 列活跃）")

        self.coef_ = coef_full
        return self

# ============================================================================
# RK4矩阵积分器
# ============================================================================

def batch_predict_rk4(model, X_data, lead_times, dt=1.0, start_month=0):
    """
    使用RK4进行批量预测 - 正确处理连续时间序列
    
    Parameters:
    -----------
    model : ps.SINDy
        训练好的模型
    X_data : np.ndarray (n_samples, n_vars)
        测试数据（连续时间序列）
    lead_times : list of int or int
        预测的提前期
    dt : float
        时间步长
    start_month : int
        X_data的第一个时间点对应的月份
    
    Returns:
    --------
    all_predictions : np.ndarray (time_len, max_lead + 1, n_vars)
        预测结果（包含初始时刻）
    
    说明:
    ----
    X_data[0] 对应月份 start_month
    X_data[1] 对应月份 start_month + 1
    X_data[t] 对应月份 start_month + t
    
    每个时间点作为初始条件时，使用其实际的月份作为季节起点
    """
    n_samples, n_vars = X_data.shape
    
    if isinstance(lead_times, (np.ndarray, list, tuple)):
        max_lead = max(lead_times)
    else:
        max_lead = lead_times
    
    time_len = n_samples - max_lead
    
    # 初始化
    X_current = X_data[:time_len].copy()  # (time_len, n_vars)
    all_predictions = np.zeros((time_len, max_lead + 1, n_vars))
    all_predictions[:, 0, :] = X_current  # 保存初始条件
    
    # 每个样本的起始月份 = 它在原始序列中的位置 + start_month
    # X_data[0] -> start_month + 0
    # X_data[1] -> start_month + 1
    # X_data[t] -> start_month + t
    start_months = np.arange(time_len, dtype=float) + start_month
    
    # 逐步预测
    for step in range(max_lead):
        # 当前绝对时间 = 起始月份 + 已预测的步数 * dt
        t_current = step * dt
        
        # 更新状态
        X_current = X_current + rk4_step(model, X_current, start_months, t_current, dt)
        X_current = np.clip(X_current, -50, 50)
        
        all_predictions[:, step + 1, :] = X_current
    
    return all_predictions

def rk4_step(model, X_current, start_months, t_current, dt):
    """
    RK4单步 - 批量处理
    
    Parameters:
    -----------
    model : ps.SINDy
        模型
    X_current : np.ndarray (batch_size, n_vars)
        当前状态
    start_months : np.ndarray (batch_size,)
        每个样本的实际起始月份（绝对时间）
    t_current : float
        当前相对时间（从初始条件开始计算）
    dt : float
        时间步长
    
    Returns:
    --------
    dX : np.ndarray (batch_size, n_vars)
        状态增量
    """
    # k1
    model.feature_library._batch_start_months = start_months + t_current
    k1 = model.predict(X_current)
    
    # k2
    model.feature_library._batch_start_months = start_months + t_current 
    k2 = model.predict(X_current + 0.5 * dt * k1)
    
    # k3
    model.feature_library._batch_start_months = start_months + t_current 
    k3 = model.predict(X_current + 0.5 * dt * k2)
    
    # k4
    model.feature_library._batch_start_months = start_months + t_current + dt
    k4 = model.predict(X_current + dt * k3)
    
    return (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================================
# RK4单步积分器
# ============================================================================

def rk4_one_step(model, x, step_idx, dt):
    """RK4单步积分 - 直接调用模型预测"""
    # 临时更新特征库的起始月份
    old_start = model.feature_library.start_month
    
    model.feature_library.start_month = step_idx
    k1 = model.predict(x.reshape(1, -1))[0]
    
    model.feature_library.start_month = step_idx  # RK4中间步的时间相同
    k2 = model.predict((x + 0.5 * dt * k1).reshape(1, -1))[0]
    
    model.feature_library.start_month = step_idx
    k3 = model.predict((x + 0.5 * dt * k2).reshape(1, -1))[0]
    
    model.feature_library.start_month = step_idx + 1
    k4 = model.predict((x + dt * k3).reshape(1, -1))[0]
    
    # 恢复原始值
    model.feature_library.start_month = old_start
    
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return np.clip(x_next, -50, 50)

def predict_n_steps_rk4(model, X_initial, start_idx, n_steps, dt=1.0):
    """使用RK4进行多步预测 - 调用模型predict"""
    n_vars = X_initial.shape[0]
    predictions = np.zeros((n_steps+1, n_vars))
    
    X_current = X_initial.copy()
    
    for step in range(n_steps+1):
        predictions[step] = X_current
        step_idx = start_idx + step
        X_current = rk4_one_step(model, X_current, step_idx, dt)
        
    
    return predictions

def batch_predict_rk4_steps(model, X_data, lead_times):
    """批量RK4预测"""
    n_samples, n_vars = X_data.shape
    max_lead = max(lead_times)
    time_len = n_samples - max_lead
    
    all_predictions = np.zeros((time_len, max_lead+1, n_vars))
    
    for t0 in range(time_len):
        preds = predict_n_steps_rk4(model, X_data[t0], t0, max_lead)
        all_predictions[t0] = preds
    
    return all_predictions

# ============================================================================
# 封装函数定义
# ============================================================================

def moving_average_3m(data):
    """计算三个月滑动平均"""
    if len(data) < 3:
        return data
    return np.convolve(data, np.ones(3)/3, mode='valid')

def evaluate_skill(X_true, X_pred, lead_times, var_idx=0,wl=0, is_mv3=True):
    """评估预测技巧"""
    corr_list, rmse_list = [], []
    
    for lead in lead_times:
        pred = X_pred[wl:, lead, var_idx]
        true = X_true[lead+wl:lead+wl+len(pred), var_idx]
        
        valid_mask = ~(np.isnan(pred) | np.isinf(pred))
        
        if np.sum(valid_mask) > 10:
            true_valid = true[valid_mask]
            pred_valid = pred[valid_mask]
            
            if is_mv3:
                true_valid = moving_average_3m(true_valid)
                pred_valid = moving_average_3m(pred_valid)
                
                if len(true_valid) < 3:
                    corr_list.append(np.nan)
                    rmse_list.append(np.nan)
                    continue
            
            corr = np.corrcoef(true_valid, pred_valid)[0, 1]
            rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
            
            corr_list.append(corr)
            rmse_list.append(rmse)
        else:
            corr_list.append(np.nan)
            rmse_list.append(np.nan)
    
    return corr_list, rmse_list

def plot_skill_comparison(lead_times, 
                         results_dict,
                         styles_dict=None,
                         title=None,
                         figsize=(12, 4),
                         save_path=None):
    """
    可视化多个模型的预测技巧对比
    
    Parameters:
    -----------
    lead_times : list
        提前期列表
    results_dict : dict
        格式: {'模型名': {'corr': [...], 'rmse': [...]}, ...}
    styles_dict : dict or None
        格式: {'模型名': {'color': '...', 'marker': '...', 'linewidth': ..., 'alpha': ...}, ...}
        如果为None，使用默认样式
    title : str or None
        图表标题
    figsize : tuple
        图表大小
    save_path : str or None
        保存路径
        
    Returns:
    --------
    fig, axes : matplotlib objects
    
    Example:
    --------
    results = {
        'Linear XRO': {'corr': [0.8, 0.7, ...], 'rmse': [0.5, 0.6, ...]},
        'Standard XRO': {'corr': [0.85, 0.75, ...], 'rmse': [0.45, 0.55, ...]},
        'MC-XRO': {'corr': [0.9, 0.8, ...], 'rmse': [0.4, 0.5, ...]}
    }
    
    custom_styles = {
        'Linear XRO': {'color': 'gray', 'marker': 's', 'linewidth': 2},
        'MC-XRO': {'color': 'orangered', 'marker': 'x', 'linewidth': 2.5}
    }
    
    plot_skill_comparison(lead_times, results, styles_dict=custom_styles)
    """
    
    # 默认样式
    default_colors = ['gray', 'deepskyblue', 'orangered', 'seagreen', 'purple', 'brown']
    default_markers = ['s', 'o', 'x', '^', 'v', 'D']
    default_styles = {
        'color': default_colors,
        'marker': default_markers,
        'linewidth': 2,
        'markersize': 6,
        'alpha': 0.85
    }
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 遍历每个模型
    for idx, (model_name, data) in enumerate(results_dict.items()):
        # 获取该模型的样式
        if styles_dict and model_name in styles_dict:
            style = styles_dict[model_name]
            color = style.get('color', default_colors[idx % len(default_colors)])
            marker = style.get('marker', default_markers[idx % len(default_markers)])
            linewidth = style.get('linewidth', 2)
            linestyle = style.get('linestyle', '-')
            markersize = style.get('markersize', 6)
            alpha = style.get('alpha', 0.85)
        else:
            # 使用默认样式
            color = default_colors[idx % len(default_colors)]
            marker = default_markers[idx % len(default_markers)]
            linewidth = 2
            linestyle = '-'
            markersize = 6
            alpha = 0.85
        
        corr = data['corr']
        rmse = data['rmse']
        
        # 绘制相关系数
        axes[0].plot(lead_times, corr, 
                    color=color, 
                    linewidth=linewidth,
                    linestyle=linestyle,
                    marker=marker, 
                    markersize=markersize,
                    label=model_name, 
                    alpha=alpha)
        
        # 绘制RMSE
        axes[1].plot(lead_times, rmse, 
                    color=color, 
                    linewidth=linewidth,
                    marker=marker, 
                    markersize=markersize,
                    label=model_name, 
                    alpha=alpha)
    
    # 相关系数图设置
    axes[0].set_xlabel('Lead Time (months)', fontsize=12)
    axes[0].set_ylabel('Correlation', fontsize=12)
    axes[0].axhline(y=0.5, color='k', linestyle='--', linewidth=1, alpha=1)
    axes[0].legend(frameon=False, fontsize=12, loc='best')
    axes[0].set_xticks(lead_times)
    axes[0].set_ylim([0.2, 1.01])
    axes[0].set_xlim([1, max(lead_times)])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # RMSE图设置
    axes[1].set_xlabel('Lead Time (months)', fontsize=12)
    axes[1].set_ylabel('RMSE (°C)', fontsize=12)
    # axes[1].legend(frameon=False, fontsize=12, loc='best')
    axes[1].set_xticks(lead_times)
    axes[1].set_xlim([1, max(lead_times)])
    axes[1].set_ylim([0.,max(rmse)+0.1])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # 标题
    if title:
        fig.suptitle(title, fontsize=14)
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存为 '{save_path}'")
    
    return fig, axes

