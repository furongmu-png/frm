# 因果涌现引擎 — 设计规范（修订版 v2）

> **状态:** 已修订 (2026-07-19)
> **作者:** Agent
> **修订原因:** 第一轮超级军事级审查发现 4 项 CRITICAL + 7 项 HIGH + 9 项 MEDIUM + 5 项 LOW + 2 项 INFO 问题
> **实现计划:** 后续文档位于 `docs/superpowers/plans/`

---

## 1. 动机与公理

本规范定义**因果涌现引擎**（Causal Emergence Engine, CEE）：一个自包含、零数据的推理系统，将持久同调、因果发现、变分 PDE 求解、哈密顿蒙特卡洛与混沌吸引子记忆组合为单一递归循环。设计基于三条公理（按原始简报）：

- **公理 I（物理现实主义）:** 世界的本质是作用量与对称性，而非像素或文字。系统的核心任务是计算守恒量，而非分类或预测下一个 token。
- **公理 II（拓扑优先）:** 语义信息蕴含在拓扑不变量中（连通分量、环路、空洞）。变形、噪声、模态转换均不改变拓扑本质。
- **公理 III（因果优于相关）:** 真正的智能必须基于结构因果模型回答反事实查询（"如果...会怎样？"），而非仅输出概率分布。

本引擎是**零数据**的：无预训练权重、无外部数据集、无重型 ML 依赖。

**依赖契约（明确边界，修复 C2）:**
- 允许：`numpy`、`scipy`（含 `scipy.linalg`、`scipy.stats`、`scipy.spatial`）、Python 标准库（`math`、`itertools`、`collections`、`dataclasses`）。
- **明确禁止**：`scikit-learn`（含 `FastICA`、`LinearRegression`）、`gudhi`、`ripser`、`pymc`、`numpyro`、`causal-learn`、`numba`（仅 optional JIT，非必需）。
- LiNGAM 中的 ICA 由本引擎自行实现简化版（详见 §4.2.2），不引入 scikit-learn。

---

## 2. 架构概览

### 2.1 包位置

```
src/zero_data_model/causal_emergence/
├── __init__.py            # 公开导出 + CausalEmergenceEngine 再导出
├── rules.py               # EmergenceRules dataclass（可配置先验）
├── topology.py            # 模块 A: PersistentHomologyPerceiver（持久同调感知器）
├── causal_discovery.py    # 模块 B: CausalInferenceEngine（PC + 简化 LiNGAM + 线性 do-calculus）
├── differential.py       # 模块 C: DifferentialGenerator（测地线 / 阻尼最小作用量）
├── hmc.py                 # 模块 D: HamiltonianSampler（蛙跳 HMC 采样器）
├── chaotic_memory.py      # 模块 E: ChaoticAssociativeMemory（Lorenz 吸引子盆记忆）
└── engine.py              # CausalEmergenceEngine（递归循环编排器）
```

### 2.2 模块模式

每个模块都是**纯 Python 类**（不继承 `CognitiveModule`、不进入 `self.modules`、不计入 `_N_COGNITIVE_MODULES`）。这与现有 `capabilities/` 模式一致：

- 构造函数签名：`(dim=64, <core_module>=None, rules=None, rng=None)`。核心模块参数默认 `None`，每个构造函数惰性实例化回退（`if x is None: x = CoreModule(...)`）。
- 方法返回纯 dict / ndarray；所有数值输出经 `np.nan_to_num` 守卫。
- 适用处返回 L2 归一化向量。
- **确定性（修复 M8）:** 每个类接受 `rng: np.random.Generator | None`，None 时构造 `np.random.default_rng()`。EmergenceRules **不**含 `seed` 字段，避免双种子源冲突。

### 2.3 与 ZeroDataModel 的集成

引擎作为单一实例属性 + facade 方法接入 `ZeroDataModel`：

```python
# 在 ZeroDataModel.__init__ 中
self.emergence_rules = EmergenceRules()
self.emergence = CausalEmergenceEngine(
    dim=dim,
    active_inference=self.active_inference,
    math_universe=self.math_universe,
    rules=self.emergence_rules,
    rng=_child_rngs[N],
)

# facade 方法（均持锁 with self._lock: ...，锁粒度为整个方法体，修复 M9）
def perceive_topology(self, data: np.ndarray) -> dict: ...
def discover_causal_dynamics(self, data: np.ndarray, var_names=None) -> dict: ...
def generate_trajectory(self, boundary: dict) -> dict: ...
def sample_posterior(self, log_prob_fn, initial_position, n_samples=100, **kwargs) -> dict: ...
def recall_memory(self, query: np.ndarray) -> dict: ...
def emergence_cycle(self, observation: np.ndarray) -> dict: ...
```

引擎实例**不**加入 `self.modules`，**不**改变 `_N_COGNITIVE_MODULES`。这保留了现有认知模块计数契约。

---

## 3. 模块 A：PersistentHomologyPerceiver（`topology.py`）

### 3.1 目的

将任意高维数据感知为拓扑点云并计算其持久同调。输出对旋转、平移、非退化变形不变。

### 3.2 算法（修复 C1：分层降级策略）

 Vietoris-Rips 复形在 n 点上的单形数为 `sum_{k=1}^{dim+2} C(n, k)`。边界矩阵列归约复杂度为 O(s³)。为保证秒级可行，采用**分层降级策略**：

1. **输入整形:** 将输入展平为 2D 点云 `(n_points, n_features)`。若输入为 1D，视为单点点云（退化情形：返回平凡不变量）。
2. **子采样:** 若 `n_points > rules.topology_max_points`（默认 **16**，修复 C1），用 `rng.choice` 做确定性子采样到 16 点。
3. **成对距离矩阵:** 欧氏距离 `D[i, j] = ||x_i - x_j||`。对称、零对角。
4. **Vietoris-Rips 过滤:** 对 `eps` 在 `[0, max(D)]` 上均匀采样 `topology_eps_steps`（默认 50）个阈值。
5. **同调计算（分层策略）:**
   - **n ≤ 16（默认路径）:** 完整边界矩阵列归约（GF(2)），计算 betti_0、betti_1、betti_2 与持久图。n=16 时单形数 ≈ 696，O(s³) ≈ 3.4e8，秒级可行。
   - **16 < n ≤ 64（子采样后必走 n=16 路径）:** 不存在，因步骤 2 已子采样。
   - **可选 JIT 加速:** 若 `hardware/kernels._betti_numbers` 可导入，调用之；否则走纯 numpy 路径。导入失败用 `try/except` 守护。
6. **持久图:** 对每个同调维度 `d`（0, 1, 2），记录 `(birth, death)` 对。
7. **持久熵:** `H = -sum(p_i * log(p_i))`，其中 `p_i = (death_i - birth_i) / total_persistence`，`total_persistence = sum(death_i - birth_i)`。
8. **欧拉示性数:** `chi = sum_d (-1)^d * betti_d`。

**删除的错误陈述:** 原 spec §3.2 说"n ≤ 32 点时可行"——这不正确，n=32 的单形数 = 5488，O(s³) ≈ 1.65e14，不可行。

### 3.3 API（修复 L1：方法参数优先）

```python
class PersistentHomologyPerceiver:
    def __init__(
        self,
        dim: int = 64,
        math_universe=None,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ): ...

    def perceive(self, data: np.ndarray, max_dim: int | None = None) -> dict:
        """计算输入点云的持久同调。

        参数 max_dim 优先于 rules.topology_max_dim；None 时使用 rules 默认值（修复 L1）。

        返回:
            {
                'betti_numbers': list[int],          # [betti_0, betti_1, betti_2]
                'persistence_diagram': list[tuple],  # [(dim, birth, death), ...]
                'persistence_entropy': float,
                'euler_characteristic': int,
                'n_points': int,                      # 子采样后的实际点数
                'max_eps': float,
            }
        """
```

### 3.4 边界情况

- 空输入：返回 `{betti_numbers: [0, 0, 0], persistence_diagram: [], persistence_entropy: 0.0, euler_characteristic: 0, n_points: 0, max_eps: 0.0}`。
- 单点：`betti_numbers = [1, 0, 0]`，持久图为空，熵为 0。
- 输入含 `NaN`/`Inf`：距离计算前用 `np.nan_to_num` 守卫。
- 大输入（> `topology_max_points`）：通过 `rng.choice` 做确定性子采样。
- 非连续数组 / 错误 dtype：内部 `np.ascontiguousarray(data, dtype=float)` 规范化。

---

## 4. 模块 B：CausalInferenceEngine（`causal_discovery.py`）

### 4.1 目的

从多变量观测数据发现有向无环图（DAG），并回答干预 / 反事实查询。

### 4.2 算法

三种发现方法，由 `rules.causal_method` 选择：

1. **PC 算法（默认）:**
   - **独立性检验（修复 M1）:** 用偏相关系数的 **Fisher z 变换**。给定偏相关系数 r，`z = 0.5 * log((1+r)/(1-r))`，统计量 `sqrt(n - |S| - 3) * |z|` 服从标准正态分布。p 值 < `causal_significance` 则拒绝独立性。
   - **骨架学习:** 起始为完全无向图；对每对 `(i, j)`，遍历条件集 `S ⊆ adj(i) \ {j}` 且 `|S| ≤ causal_max_cond_set`，若存在 S 使 `i ⊥ j | S` 则删除边。检验顺序按 `(i, j, |S|)` 字典序确定以保证可复现。
   - **定向规则（修复 M2）:** 应用 Meek (1995) 规则 R1、R2、R3（R4 需要额外的邻接信息，本实现省略）：
     - **R1:** 若 `a -> b` 且 `a - c - b` 且 `a, c` 不相邻，则定向 `c -> b`（避免新 collider）。
     - **R2:** 若 `a -> b -> c` 且 `a - c`，则定向 `a -> c`（避免环）。
     - **R3:** 若 `a - b`、`a - c`、`a - d`、`c -> b`、`d -> b`、`c, d` 不相邻，则定向 `a -> b`。
   - **常数列处理:** 跳过该列（无方差则无信息）。

2. **简化版 LiNGAM（修复 C2，不依赖 scikit-learn）:**
   - **白化:** 用 `scipy.linalg.svd` 对数据做白化，得到 `Z = (X - mean) @ W_white`，其中 `W_white = U @ diag(1/sqrt(S))`。
   - **固定点 ICA（Hyvärinen 1999）:** 迭代 `W <- (Z * g(W^T Z)).mean(axis=1) - g'(W^T Z).mean(axis=1)` 然后对称正交化 `W <- (W @ W^T)^{-1/2} @ W`。`g(u) = tanh(u)`，`g'(u) = 1 - tanh²(u)`。
   - **方向推断:** 对 `W` 做列置换找下三角化（严格下三角 = DAG）。用 `scipy.linalg.solve_triangular` 检查可行性。
   - **限制:** 仅支持 `n_vars ≤ 8`（避免置换组合爆炸）。`n_vars > 8` 或 ICA 不收敛时回退到相关法。
   - **破环:** 若 LiNGAM 产生环（罕见），按 |W[i,j]| 升序贪心删边破环。

3. **相关回退（默认退化路径）:**
   - 当 `n_samples < 3`、`n_vars > 8`（LiNGAM 时）、任一列零方差、ICA 不收敛时回退。
   - 用相关系数矩阵 `C`，对 `|C[i, j]| > causal_significance` 的对 `(i, j)`（i < j）定向为 `i -> j`。
   - 复用现有 `capabilities.causal_advanced.CausalGraphBuilder` 的逻辑。

### 4.3 Do-Calculus（修复 H3：线性近似）

**线性高斯假设下的闭式 do-calculus，而非信念传播。**

给定 DAG 邻接矩阵 `A` 和数据 `X (n_samples, n_vars)`：

1. **估计线性权重:** 对每个节点 `j`，回归 `X[:, j]` on `X[:, parents(j)]`，得权重 `W[:, j]`（`parents(j)` 由 `A` 推导）。
2. **intervene(var, value):**
   - **图割裂:** 删除 `A` 中所有 `A[i, var] = 1` 的边（即去除 var 的入边）。
   - **数据替换:** `X_do[:, var] = value`（常数列）。
   - **传播:** 对每个非干预节点 `j`，`X_do[:, j] = X_do[:, parents(j)] @ W[:, j]`（按拓扑序传播）。
   - **效应:** `effect = mean(X_do) - mean(X)`。
3. **counterfactual(var, value, observed):**
   - 对单条观测 `observed (n_vars,)`，反事实 = `observed - W[:, var] * (observed[var] - value)`。
   - 这是 do-calculus 在线性情形下的闭式解，避免信念传播的复杂性。

### 4.4 API

```python
class CausalInferenceEngine:
    def __init__(
        self,
        dim: int = 64,
        active_inference=None,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ): ...

    def discover(
        self,
        data: np.ndarray,
        var_names: list[str] | None = None,
        method: str | None = None,  # 'pc' | 'lingam' | 'correlation'；默认 rules.causal_method
    ) -> dict:
        """从观测数据发现因果 DAG。

        返回:
            {
                'adjacency': np.ndarray,        # (n_vars, n_vars) 二值
                'edges': list[tuple[int, int]],
                'method': str,
                'var_names': list[str],
                'n_edges': int,
                'is_acyclic': bool,
            }
        """

    def intervene(
        self,
        adjacency: np.ndarray,
        data: np.ndarray,
        intervention_var: int,
        intervention_value: float,
    ) -> dict:
        """估计 do(X[intervention_var] = value) 的效应（线性传播）。

        返回: {'effect': np.ndarray, 'pre_mean': np.ndarray, 'post_mean': np.ndarray}
        """

    def counterfactual(
        self,
        adjacency: np.ndarray,
        observed: np.ndarray,
        intervention_var: int,
        intervention_value: float,
    ) -> dict:
        """反事实查询：'如果 X[var] 当时是 value，会发生什么？'（线性闭式解）。

        返回: {'counterfactual': np.ndarray, 'factual': np.ndarray, 'shift': np.ndarray}
        """
```

### 4.5 边界情况

- `n_samples < 3`：回退到相关法。
- 常数列：在因果检验中跳过该列。
- ICA 不收敛：回退到相关法。
- LiNGAM 产生环：贪心删边破环。
- `n_vars > 8` 且 method='lingam'：自动回退到相关法并记录到 warnings。
- 非有限输入：抛 `ValueError`。

---

## 5. 模块 C：DifferentialGenerator（`differential.py`）

### 5.1 目的

通过求解离散化的最小作用量边值问题，生成连续且物理可行的轨迹。

### 5.2 算法（修复 H1：测地线 + 阻尼项）

**修订：** 原 spec 的拉格朗日量 `L = 0.5*||q_dot||² - 0.5*||q - target||²` 是谐振子，解会振荡而非单调收敛。改用**阻尼最小作用量**：

给定边界条件 `(start_state, end_state, n_steps)`：

1. 初始化 `start` 与 `end` 之间的线性插值轨迹 `q(t)`。
2. 定义拉格朗日量：
   ```
   L(q, q_dot) = 0.5 * ||q_dot||² - 0.5 * lambda * ||q - target||² - 0.5 * gamma * ||q_dot||² * ||q - target||
   ```
   其中 `target = end_state`（修复 H2），`lambda = rules.differential_lambda`（吸引强度），`gamma = rules.differential_gamma`（阻尼系数）。
3. 在均匀网格 `t = 0, 1, ..., n_steps` 上离散化，`dt = rules.differential_dt`。
4. 求解离散化的欧拉-拉格朗日方程。**简化：** 直接采用阻尼谐振子的离散形式：
   ```
   (q[k+1] - 2*q[k] + q[k-1]) / dt² + gamma * (q[k+1] - q[k-1]) / (2*dt) + lambda * (q[k] - target) = 0
   ```
   即"二阶时间导数 + 一阶时间导数（阻尼）+ 弹性恢复力 = 0"。
5. 用 **Gauss-Seidel 松弛** 迭代，更新内点 `q[k]` 直到 `max ||delta|| < rules.differential_tol` 或达到 `rules.differential_max_iter`。边界点固定（`q[0] = start`，`q[n_steps] = end`）。
6. 计算每步拉格朗日量 `L[k] = 0.5*||q_dot[k]||² - 0.5*lambda*||q[k]-target||²`、总作用量 `S = sum(L * dt)`。

**物理意义:** 阻尼项让轨迹单调收敛到 end_state，避免谐振子的振荡。当 `lambda=0, gamma=0` 时退化为最短路径（线性插值）。

### 5.3 API

```python
class DifferentialGenerator:
    def __init__(
        self,
        dim: int = 64,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ): ...

    def generate(
        self,
        start_state: np.ndarray,
        end_state: np.ndarray,
        n_steps: int = 32,
        constraints: dict | None = None,  # placeholder，本期不实现（修复 L2）
    ) -> dict:
        """求解边值问题。

        返回:
            {
                'trajectory': np.ndarray,       # (n_steps+1, dim)
                'lagrangian': np.ndarray,        # (n_steps,) 每步
                'action': float,
                'converged': bool,
                'iterations': int,
            }
        """
```

### 5.4 边界情况

- `start == end`：返回常数轨迹 `q[:] = start`，`action = 0`，`converged = True`。
- `n_steps == 0`：返回单点轨迹 `[start]`。
- `max_iter` 内未收敛：返回最后一次迭代，`converged = False`。
- 所有输出 NaN 守卫。
- 非有限 start/end：抛 `ValueError`。

---

## 6. 模块 D：HamiltonianSampler（`hmc.py`）

### 6.1 目的

用哈密顿蒙特卡洛（蛙跳积分）从后验分布采样。提供校准的不确定性估计。

### 6.2 算法（修复 M3、M4、M5、L4）

1. **输入:** `log_prob_fn(position) -> float`（对数后验）、`initial_position`、`n_samples`、`step_size`、`n_leapfrog`。
2. **势能:** `U(q) = -log_prob_fn(q)`。
3. **梯度（修复 M3）:** 若 `grad_fn` 提供，直接用；否则用**中心差分**，步长 `h = 1e-5`：
   ```
   grad_U[i] = (U(q + h*e_i) - U(q - h*e_i)) / (2*h)
   ```
4. **动量重采样:** `p ~ N(0, M)`，`M` 为单位质量矩阵。
5. **蛙跳积分:**
   ```
   p_half = p - (step_size / 2) * grad_U(q)
   q_new = q + step_size * p_half
   p_new = p_half - (step_size / 2) * grad_U(q_new)
   ```
   重复 `n_leapfrog` 步。
6. **Metropolis 接受/拒绝:** 以概率 `min(1, exp(H_old - H_new))` 接受，`H = U + 0.5 * ||p||²`。
7. **异常处理（修复 L4）:** `log_prob_fn` 抛任何异常时，捕获并拒绝该提议，记录到 `warnings` 列表。返回 `-inf` 或 NaN 时同样拒绝。
8. **ESS 估计（修复 M4）:** 用**初始单调序列法（Geyer 1992）**：从 lag-1 自相关开始，按 lag 累加直到自相关之和首次变负，ESS = `n_samples / (1 + 2 * sum)`。**单链 ESS 估计有较大不确定性**（修复 L5），建议多链运行后用 Gelman-Rubin 综合。
9. **重复** `n_samples` 次收集样本。计算均值、标准差、ESS、`accept_rate`。

### 6.3 API

```python
class HamiltonianSampler:
    def __init__(
        self,
        dim: int = 64,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ): ...

    def sample(
        self,
        log_prob_fn,                       # 可调用: np.ndarray -> float
        initial_position: np.ndarray,
        n_samples: int = 100,
        step_size: float = 0.1,
        n_leapfrog: int = 10,
        grad_fn=None,                      # 可选解析梯度
    ) -> dict:
        """从 log_prob_fn 定义的后验中采样。

        单链 ESS 估计有较大不确定性，建议多链运行后用 Gelman-Rubin 综合。

        返回:
            {
                'samples': np.ndarray,     # (n_samples, dim)
                'mean': np.ndarray,
                'std': np.ndarray,
                'accept_rate': float,
                'ess': float,             # 有效样本量
                'converged': bool,         # 见下
                'warnings': list[str],     # 异常记录
            }
        """
```

### 6.4 收敛判据（修复 M5）

- `converged = True` 当且仅当 `0.5 <= accept_rate <= 0.95`。
- **修订说明:** 原 spec 的 `[0.2, 0.9]` 过严，对高斯后验 HMC 几乎不拒绝（accept_rate 接近 1.0 是正常的）。改为 `[0.5, 0.95]`。
- 真正收敛需要 R-hat < 1.01（多链）或 ESS/samples > 0.4，本引擎单链仅做启发式判据。

### 6.5 边界情况

- `log_prob_fn` 返回 `-inf` 或 NaN：拒绝该提议。
- `log_prob_fn` 抛异常：捕获并拒绝，记录到 warnings。
- `initial_position` 非有限：抛 `ValueError`。
- `n_samples == 0`：返回空数组。
- 接受率超出 `[0.5, 0.95]`：`converged = False`（建议调参）。

---

## 7. 模块 E：ChaoticAssociativeMemory（`chaotic_memory.py`）

### 7.1 目的

将模式存储为 Lorenz 类动力系统的弱吸引子盆。检索时收敛到最近的已存模式；落在盆边界上的查询产生混沌游走（定义为"灵感"或"涌现"）。

### 7.2 算法（修复 C4、H7、M6）

1. **存储（修复 C4 完整编码方案）:**
   每个 dim 维模式 `pattern` 编码为 Lorenz 系统的目标平衡点 `(target_y, target_z)`：
   ```
   target_y = mean(pattern[:dim//2])
   target_z = mean(pattern[dim//2:])
   ```
   即将模式分成前后两半，各取均值降维到 2 维。
   存储为列表 `[(pattern, label, target_y, target_z), ...]`。

2. **修改后的 Lorenz 系统（修复 H7）:**
   ```
   dx/dt = sigma * (y - x)
   dy/dt = x * (rho - z) - y - alpha * sum_i w_i * (y - target_y_i)
   dz/dt = x * y - beta * z - alpha * sum_i w_i * (z - target_z_i)
   ```
   其中：
   - `sigma = rules.chaotic_lorenz_sigma`（默认 10.0）
   - `rho = rules.chaotic_lorenz_rho`（默认 28.0）
   - `beta = rules.chaotic_lorenz_beta`（默认 8/3）
   - `alpha = rules.chaotic_alpha`（**新增**，默认 0.1）控制吸引强度
   - `w_i = exp(-||query - pattern_i||² / (2 * sigma_q²))` 是查询对第 i 个模式的亲和权重（高斯核），`sigma_q = rules.chaotic_sigma_q`（默认 1.0）

   **稳定性说明（修复 C4）:** alpha 太大会消除混沌（系统塌缩到固定点），太小则记忆无效。默认 0.1 在保持混沌的同时提供弱吸引。

3. **检索:** 用 RK4 以查询为初值积分 Lorenz 系统 `n_steps` 步。**最近模式检索:** 积分结束后，按 L2 距离 `||query - pattern_i||` 找最近的已存模式。

4. **涌现检测（修复 M6）:** 用**轨迹发散度**而非 Lyapunov 指数：
   ```
   perturbed = query + rules.chaotic_perturbation * rng.standard_normal(query.shape)
   trajectory_perturbed = integrate(perturbed, n_steps)
   divergence = ||trajectory[-1] - trajectory_perturbed[-1]|| / (||query - perturbed|| + 1e-12)
   emerged = (divergence > rules.chaotic_divergence_threshold)
   ```
   阈值 `chaotic_divergence_threshold`（默认 10.0，远大于 1 表示混沌放大初始扰动）。

5. **容量（修复 C4）:** **删除"理论无上限"陈述。** 实际容量受 `rules.chaotic_memory_capacity`（默认 32）限制，超出时 FIFO 驱逐最旧模式。

### 7.3 API

```python
class ChaoticAssociativeMemory:
    def __init__(
        self,
        dim: int = 64,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ): ...

    def store(self, pattern: np.ndarray, label: int | str) -> dict:
        """将模式存为新吸引子盆。

        返回: {'label': ..., 'n_stored': int, 'capacity': int, 'evicted': list}
        """

    def recall(self, query: np.ndarray, n_steps: int = 100) -> dict:
        """检索最近的已存模式。

        返回:
            {
                'label': int | str | None,
                'similarity': float,
                'emerged': bool,
                'trajectory': np.ndarray,    # (n_steps + 1, 3) Lorenz 状态（fix NEW-M5：与模块 C 一致，含初始 + 终止边界）
                'converged': bool,
                'divergence': float,
                # fix NEW-L1: nearest_pattern 暴露原始存储向量（调试 /
                # 可视化用）；不参与涌现度计算。fix R2-NEW-L1: None 当
                # (a) 记忆为空、(b) 查询含 NaN/Inf、(c) 引擎级失败降级。
                'nearest_pattern': np.ndarray | None,
            }
        """

    def clear(self) -> None:
        """清除所有已存模式。"""
```

### 7.4 边界情况

- 空记忆：`recall` 返回 `label=None, similarity=0.0, emerged=True, trajectory=zeros, converged=False, divergence=0.0`。
- 容量超限：驱逐最旧模式（FIFO），返回 `evicted` 列表。
- 查询含 NaN：返回 `label=None, emerged=False, converged=False`。
- dim 为奇数：`pattern[:dim//2]` 与 `pattern[dim//2:]` 长度差 1，不影响 mean 计算。

---

## 8. CausalEmergenceEngine（`engine.py`）

### 8.1 目的

将五个模块编排为简报所述的递归循环：感知 → 因果锚定 → 反事实模拟 → 不确定性评估 → 记忆交互 → 涌现。

### 8.2 API（修复 C3、H6、M7）

```python
class CausalEmergenceEngine:
    def __init__(
        self,
        dim: int = 64,
        active_inference=None,
        math_universe=None,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.topology = PersistentHomologyPerceiver(dim=dim, ...)
        self.causal = CausalInferenceEngine(dim=dim, ...)
        self.differential = DifferentialGenerator(dim=dim, ...)
        self.hmc = HamiltonianSampler(dim=dim, ...)
        self.memory = ChaoticAssociativeMemory(dim=dim, ...)
        # ... 保存核心模块

    def perceive_topology(self, data: np.ndarray) -> dict:
        """模块 A facade。"""

    def discover_causal_dynamics(self, data: np.ndarray, var_names=None) -> dict:
        """模块 B facade。"""

    def generate_trajectory(self, boundary: dict) -> dict:
        """模块 C facade。"""

    def sample_posterior(self, log_prob_fn, initial_position, **kwargs) -> dict:
        """模块 D facade。"""

    def recall_memory(self, query: np.ndarray) -> dict:
        """模块 E facade。"""

    def emergence_cycle(self, observation: np.ndarray) -> dict:
        """对一条观测运行完整递归循环。

        observation 形状要求（修复 C3）:
        - 2D (n_samples, n_features)，要求 n_samples >= 2 且 n_features >= 2。
        - 1D 输入直接返回零分：{'emergence_score': 0.0, 'reason': 'insufficient_data'}。

        步骤:
        1. perception = perceive_topology(observation)
        2. causal_graph = discover_causal_dynamics(observation)
           （n_samples < 3 时模块 B 自动回退到相关法）
        3. counterfactual = generate_trajectory({
               'start_state': observation.mean(axis=0),  # 用均值作为代表性状态
               'end_state': observation.mean(axis=0) + delta,  # 扰动
           })
           delta = rules.emergence_cycle_perturbation * rng.standard_normal(n_features)
           （修复 M7：明确 delta 形状为随机扰动方向）
        4. posterior = sample_posterior(
               log_prob_fn=lambda x: -0.5 * ||x - observation.mean(axis=0)||²,
               initial_position=observation.mean(axis=0),
               n_samples=rules.hmc_samples,
           )
        5. memory_response = recall_memory(observation.mean(axis=0))
        6. emergence_score = compute_emergence_score(
               perception, causal_graph, counterfactual, posterior, memory_response
           )

        失败降级（修复 H6）:
        - 任一模块失败时，对应项计为 0，warnings 列表记录失败原因。
        - emergence_score 仍可计算但会偏低。

        返回: {
            'perception': dict,
            'causal_graph': dict,
            'counterfactual': dict,
            'posterior': dict,
            'memory': dict,
            'emergence_score': float,    # [0, 1]
            'warnings': list[str],        # 失败原因
        }
        """

    def compute_emergence_score(self, *outputs) -> float:
        """启发式涌现度: 高持久熵 + 高因果图密度 + 高记忆涌现 +
        低后验方差 → 高涌现度。"""
```

### 8.3 涌现度评分（修复 H4、H5）

涌现度是 `[0, 1]` 区间的启发式指标，组合：

- `0.3 * persistence_entropy`（拓扑复杂度，已归一化到 [0, 1]）
- `0.2 * (n_edges / max_edges)`（因果图密度，`max_edges = n_vars * (n_vars - 1) / 2`）
- `0.2 * memory.emerged`（若发生混沌游走取 1.0，否则 0.0）
- `0.15 * (1 - mean(posterior.std) / dim)`（**修复 H5:** 用 `mean(posterior.std)` 而非 `posterior.std / dim`）
- `0.15 * (counterfactual.action / reference_action)`（**修复 H4:** `reference_action = ||observation.mean(axis=0)||² * n_steps / 2`，即直线轨迹的作用量）

所有项在组合前归一化到 `[0, 1]`（用 `np.clip`）。权重总和 = 1.0。

---

## 9. EmergenceRules（`rules.py`）（修复 H7、M8）

```python
@dataclass
class EmergenceRules(DomainRules):
    """因果涌现引擎的先验。"""

    # 模块 A: 拓扑（修复 C1）
    topology_max_points: int = 16          # 点云规模上限（降为 16）
    topology_max_dim: int = 2              # 最大同调维度
    topology_eps_steps: int = 50          # 过滤分辨率

    # 模块 B: 因果发现
    causal_method: str = "pc"              # 'pc' | 'lingam' | 'correlation'
    causal_significance: float = 0.05      # Fisher z 检验 p 值阈值
    causal_max_cond_set: int = 3          # PC 中条件集大小上限
    causal_max_vars_lingam: int = 8        # LiNGAM 变量数上限

    # 模块 C: 微分（修复 H1）
    differential_max_iter: int = 100       # Gauss-Seidel 迭代上限
    differential_tol: float = 1e-6         # 收敛容差
    differential_dt: float = 0.01          # 时间步长
    differential_lambda: float = 1.0        # 吸引强度（新增）
    differential_gamma: float = 0.5        # 阻尼系数（新增）

    # 模块 D: HMC
    hmc_step_size: float = 0.1
    hmc_n_leapfrog: int = 10
    hmc_samples: int = 100
    hmc_target_accept: float = 0.65        # Beskos 等的最优值
    hmc_finite_diff_h: float = 1e-5        # 中心差分步长（修复 M3）

    # 模块 E: 混沌记忆（修复 C4、H7、M6）
    chaotic_memory_capacity: int = 32
    chaotic_lorenz_sigma: float = 10.0
    chaotic_lorenz_rho: float = 28.0
    chaotic_lorenz_beta: float = 8.0 / 3.0
    chaotic_alpha: float = 0.1             # 吸引强度（修复 H7）
    chaotic_sigma_q: float = 1.0           # 高斯核带宽（新增）
    chaotic_perturbation: float = 0.01     # 涌现检测扰动量（新增）
    chaotic_divergence_threshold: float = 10.0  # 涌现检测阈值（修复 M6）

    # 引擎
    emergence_cycle_perturbation: float = 0.1  # 反事实扰动量
    # 修复 M8：删除 seed 字段，仅由 rng 参数控制确定性
```

---

## 10. 测试策略

### 10.1 各模块测试

每个模块对应 `tests/test_causal_emergence_<module>.py`：

- **正确性:** 已知答案测试。
  - **模块 A（修复 I1）:** 在单位圆上采样 16 个点（n=16，符合 max_points），VR 过滤后应得 `betti_0=1, betti_1=1, betti_2=0`。采样方法：`angles = linspace(0, 2*pi, 16, endpoint=False); points = column_stack([cos(angles), sin(angles)])`。
  - **模块 B:** 对线性高斯 DAG 数据（已知 ground truth），PC 算法应恢复正确的 skeleton。对非高斯数据，LiNGAM 应恢复正确方向。
  - **模块 C:** 给定 start=0, end=1, n_steps=10，trajectory 应单调递增（阻尼项确保收敛）。
  - **模块 D:** 对标准正态 log_prob（`-0.5 * ||x||²`），采样均值应接近 0，标准差接近 1。
  - **模块 E:** 存储 3 个模式后，对每个模式本身查询应返回相同 label，`emerged=False`。
- **边界情况:** 空输入、单点、NaN、退化（零方差）、高维。
- **确定性:** 同种子产出相同输出。
- **API 契约:** 返回 dict 含期望键。

### 10.2 引擎集成测试

`tests/test_causal_emergence_engine.py`：
- 在合成观测 `(50, 4)` 上运行完整 `emergence_cycle`。
- 每个 facade 正确委托给底层模块。
- **线程安全（修复 M9）:** `self._lock` 锁粒度为整个 facade 方法体。模块内部不持有可变状态（store 是 append-only deque，recall 是只读）。并发 `emergence_cycle` 调用串行化执行。

### 10.3 测试文件

```
tests/
├── test_causal_emergence_topology.py
├── test_causal_emergence_causal_discovery.py
├── test_causal_emergence_differential.py
├── test_causal_emergence_hmc.py
├── test_causal_emergence_chaotic_memory.py
└── test_causal_emergence_engine.py
```

### 10.4 性能测试（修复 I2）

新增 `tests/test_causal_emergence_performance.py`：对每个模块在最大输入规模下测时延：
- 模块 A：n_points=16, dim=64，单次调用 < 2 秒。
- 模块 B：n_samples=100, n_vars=8, dim=64，单次调用 < 1 秒。
- 模块 C：n_steps=32, dim=64，单次调用 < 1 秒。
- 模块 D：n_samples=100, dim=64，n_leapfrog=10，单次调用 < 5 秒。
- 模块 E：n_stored=32, n_steps=100, dim=64，单次 recall < 1 秒。
- 引擎：单次 emergence_cycle < 10 秒。

### 10.5 确定性 fixture

每个测试文件包含：

```python
@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)
```

---

## 11. 军事级审查范围（15 维）

实现完成后，全面审查覆盖：

1. **数学正确性:** 算法对照参考文献（持久同调 Edelsbrunner；PC 算法 Spirtes-Glymour-Scheines + Meek 1995；LiNGAM Shimizu 等；HMC Betancourt；Lorenz 吸引子 Lorenz 1963）。
2. **数值稳定性:** NaN/Inf 守卫、条件数、收敛阈值、有限差分步长选择。
3. **边界情况:** 空输入、单点、退化分布、全零数据、极端量级。
4. **线程安全:** `ZeroDataModel` facade 的 `_lock` 一致性；模块内无共享可变状态。
5. **API 契约一致性:** 返回 dict 键与规范一致；类型与注解一致；`Optional` 返回值有文档。
6. **性能:** 算法复杂度（持久同调 n≤16 时 O(s³) ≈ 3.4e8，秒级可行；PC 算法 O(n_vars² * 2^max_cond_set)；HMC O(n_samples * n_leapfrog * dim)）；大输入退化路径（子采样、相关回退）。
7. **测试覆盖盲区:** 分支覆盖、变异测试、参数边界测试。
8. **集成正确性:** 引擎组合模块无隐藏耦合；`ZeroDataModel` facade 正确委托；`_N_COGNITIVE_MODULES` 未变。
9. **zero-data 哲学符合度:** 无预训练权重、无外部数据集、无重型 ML 依赖（明确禁止 scikit-learn、gudhi、ripser、pymc 等）；所有先验编码在 `EmergenceRules`。
10. **文档准确性:** docstring 与实现一致；docstring 中的示例可运行。
11. **确定性:** 同种子可复现；无全局状态突变。
12. **内存边界:** 有界 deque、无无界缓存、子采样限制内存。
13. **类型注解:** 所有公开 API 有注解；统一使用 `from __future__ import annotations`。
14. **边界错误处理:** 公开 API 对非有限输入抛 `ValueError`；可选 JIT kernel 导入用 `try/except`。
15. **对抗输入:** NaN、Inf、全零、全相同、极端大小、非连续数组、错误 dtype。

审查产出一份带严重性分级（CRITICAL / HIGH / MEDIUM / LOW / INFO）的书面报告与具体修复建议。

---

## 12. 实现阶段

实现按原始简报的四个阶段顺序执行，作为单一批次：

1. **阶段 1（基石）:** 模块 A（topology）+ 模块 B（causal_discovery）+ `EmergenceRules` + 部分引擎（仅 perceive + discover facade）。
2. **阶段 2（行动）:** 模块 C（differential）+ generate_trajectory facade。
3. **阶段 3（认知）:** 模块 D（hmc）+ 模块 E（chaotic_memory）+ sample_posterior + recall_memory facade。
4. **阶段 4（闭环）:** `emergence_cycle` 编排 + 涌现度计算 + ZeroDataModel 集成。

每个阶段以一次 commit 与一轮聚焦测试收尾。

---

## 13. 验收标准

- 5 个模块 + 引擎按本规范实现。
- 6 个测试文件 + 1 个性能测试文件全部通过（每模块 ≥ 30 测试，引擎 ≥ 15 测试，性能 ≥ 5 测试）。
- 现有 capability 测试仍通过（无回归）。
- `ruff check` 在所有新文件上无告警。
- 军事级审查完成并作为独立报告文档提交。
- `ZeroDataModel` 集成：`emergence_cycle` 在合成观测 `(50, 4)` 上端到端运行无错误。
- 性能：单次 `emergence_cycle` 在最大输入规模下 < 10 秒。

---

## 14. 范围之外

- 生产级持久同调（不引入 gudhi / ripser）。
- GPU 加速（不引入 cupy / numba CUDA kernel）。
- 分布式 / 并行采样（不引入 MPI / Dask）。
- Web API 端点（不引入 FastAPI 路由）。
- 预训练模型或外部数据集。
- 实时性性能保证。
- 持久层（不保存/加载引擎状态）。
- constraints 避障（DifferentialGenerator 的 constraints 参数为 placeholder，本期不实现）。

---

## 15. 修订日志

### v2 (2026-07-19)
基于第一轮超级军事级审查（27 项问题），修复：

- **C1（持久同调复杂度）:** max_points 从 64 降为 16，明确分层降级策略，删除"n ≤ 32 可行"的错误陈述。
- **C2（LiNGAM 依赖冲突）:** 明确依赖契约（禁止 scikit-learn），LiNGAM 改为自行实现简化版（SVD 白化 + 固定点 ICA），限制 n_vars ≤ 8。
- **C3（emergence_cycle 接口不匹配）:** 明确 observation 形状要求（2D (n_samples, n_features)），1D 输入直接返回零分。
- **C4（混沌记忆编码方案缺失）:** 完整定义 pattern → (target_y, target_z) 编码（前后两半均值降维），删除"理论无上限"陈述，明确 alpha=0.1。
- **H1（微分方程物理模型不符）:** 改用阻尼最小作用量（阻尼谐振子离散形式），避免振荡。
- **H2（target 未定义）:** 明确 target = end_state。
- **H3（反事实算法不完整）:** 改为线性高斯假设下的闭式 do-calculus，避免信念传播。
- **H4（reference_action 未定义）:** 明确为直线轨迹的作用量。
- **H5（posterior.std 形状不匹配）:** 改为 `mean(posterior.std)`。
- **H6（emergence_cycle 失败降级）:** 增加 warnings 字段记录失败原因。
- **H7（alpha 取值缺失）:** 增加 `chaotic_alpha: float = 0.1` 到 rules。
- **M1-M9:** 明确 Fisher z 检验、Meek R1-R3 定向规则、中心差分步长、初始单调序列 ESS、收敛判据改为 [0.5, 0.95]、轨迹发散度检测、delta 随机扰动形状、删除 seed 字段、锁粒度。
- **L1-L5:** max_dim 优先级、constraints placeholder、reference_action、异常捕获、ESS 局限性说明。
- **I1-I2:** 圆形 Betti 测试采样方法、性能测试文件。
