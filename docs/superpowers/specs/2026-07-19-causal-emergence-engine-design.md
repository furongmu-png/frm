# 因果涌现引擎 — 设计规范

> **状态:** 已批准 (2026-07-19)
> **作者:** Agent
> **实现计划:** 后续文档位于 `docs/superpowers/plans/`

---

## 1. 动机与公理

本规范定义**因果涌现引擎**（Causal Emergence Engine, CEE）：一个自包含、零数据的推理系统，将持久同调、因果发现、变分 PDE 求解、哈密顿蒙特卡洛与混沌吸引子记忆组合为单一递归循环。设计基于三条公理（按原始简报）：

- **公理 I（物理现实主义）:** 世界的本质是作用量与对称性，而非像素或文字。系统的核心任务是计算守恒量，而非分类或预测下一个 token。
- **公理 II（拓扑优先）:** 语义信息蕴含在拓扑不变量中（连通分量、环路、空洞）。变形、噪声、模态转换均不改变拓扑本质。
- **公理 III（因果优于相关）:** 真正的智能必须基于结构因果模型回答反事实查询（"如果...会怎样？"），而非仅输出概率分布。

本引擎是**零数据**的：无预训练权重、无外部数据集、无重型 ML 依赖（明确排除 gudhi / pymc / causal-learn / ripser）。所有算法仅用 `numpy` + `scipy` + Python 标准库实现，与现有 `zero_data_model` 代码库哲学一致。

---

## 2. 架构概览

### 2.1 包位置

```
src/zero_data_model/causal_emergence/
├── __init__.py            # 公开导出 + CausalEmergenceEngine 再导出
├── rules.py               # EmergenceRules dataclass（可配置先验）
├── topology.py            # 模块 A: PersistentHomologyPerceiver（持久同调感知器）
├── causal_discovery.py    # 模块 B: CausalInferenceEngine（PC + LiNGAM + do-calculus）
├── differential.py       # 模块 C: DifferentialGenerator（欧拉-拉格朗日 PDE 求解器）
├── hmc.py                 # 模块 D: HamiltonianSampler（蛙跳 HMC 采样器）
├── chaotic_memory.py      # 模块 E: ChaoticAssociativeMemory（Lorenz 吸引子盆记忆）
└── engine.py              # CausalEmergenceEngine（递归循环编排器）
```

### 2.2 模块模式

每个模块都是**纯 Python 类**（不继承 `CognitiveModule`、不进入 `self.modules`、不计入 `_N_COGNITIVE_MODULES`）。这与现有 `capabilities/` 模式一致：

- 构造函数签名：`(dim=64, <core_module>=None, rules=None)`。核心模块参数默认 `None`，每个构造函数惰性实例化回退（`if x is None: x = CoreModule(...)`）。
- 方法返回纯 dict / ndarray；所有数值输出经 `np.nan_to_num` 守卫。
- 适用处返回 L2 归一化向量。
- 确定性：每个类接受可选 `rng: np.random.Generator | None`。

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

# facade 方法
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

### 3.2 算法

1. **输入整形:** 将输入展平为 2D 点云 `(n_points, n_features)`。若输入为 1D，视为单点点云（退化情形：返回平凡不变量）。
2. **成对距离矩阵:** 欧氏距离 `D[i, j] = ||x_i - x_j||`。对称、零对角。
3. **Vietoris-Rips 过滤:** 对每个递增阈值 `eps`，构建单纯复形——当所有成对距离 ≤ `eps` 时单形出现。
4. **边界矩阵归约:** 标准从左到右列归约（GF(2)）——对低维 Betti 数，n ≤ 32 点时可行。对更大输入，若可用则回退到现有 `hardware/kernels._betti_numbers` JIT kernel；否则用 rng 做确定性子采样，将 n_points 上限设为 64。
5. **持久图:** 对每个同调维度 `d`（0, 1, 2），记录特征出现与消失的 `(birth, death)` 对。
6. **持久熵:** `H = -sum(p_i * log(p_i))`，其中 `p_i = (death_i - birth_i) / total_persistence`。
7. **欧拉示性数:** `chi = sum_d (-1)^d * betti_d`。

### 3.3 API

```python
class PersistentHomologyPerceiver:
    def __init__(
        self,
        dim: int = 64,
        math_universe=None,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ): ...

    def perceive(self, data: np.ndarray, max_dim: int = 2) -> dict:
        """计算输入点云的持久同调。

        返回:
            {
                'betti_numbers': list[int],          # [betti_0, betti_1, betti_2]
                'persistence_diagram': list[tuple],  # [(dim, birth, death), ...]
                'persistence_entropy': float,
                'euler_characteristic': int,
                'n_points': int,
                'max_eps': float,
            }
        """
```

### 3.4 边界情况

- 空输入：返回 `{betti_numbers: [0, 0, 0], persistence_diagram: [], persistence_entropy: 0.0, euler_characteristic: 0, n_points: 0, max_eps: 0.0}`。
- 单点：`betti_numbers = [1, 0, 0]`，持久图为空，熵为 0。
- 输入含 `NaN`/`Inf`：距离计算前用 `np.nan_to_num` 守卫。
- 大输入（> 64 点）：通过 `rng.choice` 做确定性子采样。

---

## 4. 模块 B：CausalInferenceEngine（`causal_discovery.py`）

### 4.1 目的

从多变量观测数据发现有向无环图（DAG），并回答干预 / 反事实查询。

### 4.2 算法

三种发现方法，由 `rules.causal_method` 选择：

1. **PC 算法（默认）:** 起始为完全无向图；对每对 `(i, j)`，若存在条件集 `S` 使 `i ⊥ j | S`（偏相关检验，阈值 `causal_significance`），则删除边。用 collider 检测（v-结构）和无环性约束定向边。边检验顺序确定以保证可复现。

2. **LiNGAM（非高斯 ICA）:** 对数据运行 FastICA 获得独立成分；混合矩阵 `W` 揭示因果方向（`W` 的非零项指示因果边）。将 `W` 经置换后转换为下三角 DAG。

3. **相关回退:** 当 `n_samples < 3` 或数据退化（任一列零方差）时，回退到现有 `CausalGraphBuilder` 的相关阈值法。

### 4.3 Do-calculus

给定已发现的 DAG，`intervene(var, value)` 估计干预效应：
- 置 `data[:, var] = value`（图割裂）。
- 重新计算边际均值。
- 效应 = 后 - 前。

反事实：对单条观测，通过 DAG 上的信念传播计算干预下的后验均值。

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
        """估计 do(X[intervention_var] = value) 的效应。"""

    def counterfactual(
        self,
        adjacency: np.ndarray,
        observed: np.ndarray,
        intervention_var: int,
        intervention_value: float,
    ) -> dict:
        """反事实查询：'如果 X[var] 当时是 value，会发生什么？'"""
```

### 4.5 边界情况

- `n_samples < 3`：回退到相关法。
- 常数列：在因果检验中跳过该列。
- ICA 不收敛：回退到相关法。
- LiNGAM 产生环（罕见）：贪心删边破环。

---

## 5. 模块 C：DifferentialGenerator（`differential.py`）

### 5.1 目的

通过求解离散化的欧拉-拉格朗日边值问题，生成连续且物理可行的轨迹。

### 5.2 算法

给定边界条件 `(start_state, end_state, n_steps)`：

1. 初始化 `start` 与 `end` 之间的线性插值轨迹 `q(t)`。
2. 定义拉格朗日量 `L(q, q_dot) = T(q_dot) - V(q)`，其中：
   - `T(q_dot) = 0.5 * ||q_dot||^2`（动能）
   - `V(q) = 0.5 * ||q - target||^2`（朝目标的二次势能）
3. 在均匀网格 `t = 0, 1, ..., n_steps` 上离散化，`dt = 1 / n_steps`。
4. 求解离散化的欧拉-拉格朗日方程：
   `d/dt(dL/dq_dot) - dL/dq = 0`
   离散形式：
   `(q[k+1] - 2*q[k] + q[k-1]) / dt^2 = -dV/dq = -(q[k] - target)`
5. 用 Gauss-Seidel 松弛迭代，直到 `max ||delta|| < tol` 或达到 `max_iter`。
6. 计算每步拉格朗日量、总作用量 `S = sum(L * dt)`。

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
        constraints: dict | None = None,  # 可选避障
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

- `start == end`：返回常数轨迹。
- `n_steps == 0`：返回单点轨迹。
- `max_iter` 内未收敛：返回最后一次迭代，`converged=False`。
- 所有输出 NaN 守卫。

---

## 6. 模块 D：HamiltonianSampler（`hmc.py`）

### 6.1 目的

用哈密顿蒙特卡洛（蛙跳积分）从后验分布采样。提供校准的不确定性估计。

### 6.2 算法

1. **输入:** `log_prob_fn(position) -> float`（对数后验）、`initial_position`、`n_samples`、`step_size`、`n_leapfrog`。
2. **势能:** `U(q) = -log_prob_fn(q)`。梯度用有限差分（或调用方提供的解析梯度）。
3. **动量重采样:** `p ~ N(0, M)`，`M` 为单位质量矩阵。
4. **蛙跳积分:**
   ```
   p_half = p - (step_size / 2) * grad_U(q)
   q_new = q + step_size * p_half
   p_new = p_half - (step_size / 2) * grad_U(q_new)
   ```
   重复 `n_leapfrog` 步。
5. **Metropolis 接受/拒绝:** 以概率 `min(1, exp(H_old - H_new))` 接受，`H = U + 0.5 * ||p||^2`。
6. 重复 `n_samples` 次收集样本。
7. 计算均值、标准差、ESS（有效样本量）和 `accept_rate`。

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

        返回:
            {
                'samples': np.ndarray,     # (n_samples, dim)
                'mean': np.ndarray,
                'std': np.ndarray,
                'accept_rate': float,
                'ess': float,             # 有效样本量
                'converged': bool,         # accept_rate 在 [0.2, 0.9] 内为 True
            }
        """
```

### 6.4 边界情况

- `log_prob_fn` 返回 `-inf` 或 NaN：拒绝该提议。
- `initial_position` 非有限：在边界处抛出 `ValueError`。
- `n_samples == 0`：返回空数组。
- 接受率 < 0.2 或 > 0.9：标记 `converged=False`（建议调参）。

---

## 7. 模块 E：ChaoticAssociativeMemory（`chaotic_memory.py`）

### 7.1 目的

将模式存储为 Lorenz 类动力系统的稳定吸引子盆。检索时收敛到最近的已存模式；落在盆边界上的查询产生混沌游走（定义为"灵感"或"涌现"）。

### 7.2 算法

1. **存储:** 每个模式编码为 Lorenz 类系统的目标平衡点：
   ```
   dx/dt = sigma * (y - x)
   dy/dt = x * (rho - z) - y - alpha * (y - target_y)
   dz/dt = x * y - beta * z - alpha * (z - target_z)
   ```
   其中 `(target_y, target_z)` 由模式推导。`alpha` 项将轨迹拉向已存模式的盆。
2. **检索:** 用 RK4 以查询为初值积分 Lorenz 系统 `n_steps` 步。收敛（或达 `n_steps`）后，按 L2 距离找最近的已存模式。
3. **涌现检测:** 若轨迹的 Lyapunov 指数全程为正（混沌游走）且未收敛，则标记 `emerged=True`。
4. **容量:** 理论无上限（每个模式为独立吸引子），但实际容量受 `rules.chaotic_memory_capacity`（默认 32）限制。

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

        返回: {'label': ..., 'n_stored': int, 'capacity': int}
        """

    def recall(self, query: np.ndarray, n_steps: int = 100) -> dict:
        """检索最近的已存模式。

        返回:
            {
                'label': int | str | None,
                'similarity': float,
                'emerged': bool,
                'trajectory': np.ndarray,    # (n_steps, 3) Lorenz 状态
                'converged': bool,
            }
        """

    def clear(self) -> None:
        """清除所有已存模式。"""
```

### 7.4 边界情况

- 空记忆：`recall` 返回 `label=None, similarity=0.0, emerged=True`（无盆的混沌游走）。
- 容量超限：驱逐最旧模式（FIFO）。
- 查询含 NaN：返回 `label=None, emerged=False, converged=False`。

---

## 8. CausalEmergenceEngine（`engine.py`）

### 8.1 目的

将五个模块编排为简报所述的递归循环：感知 → 因果锚定 → 反事实模拟 → 不确定性评估 → 记忆交互 → 涌现。

### 8.2 API

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

        步骤:
        1. perception = perceive_topology(observation)
        2. causal_graph = discover_causal_dynamics(observation)
        3. counterfactual = generate_trajectory({
               'start_state': observation,
               'end_state': observation + delta,  # 扰动
           })
        4. posterior = sample_posterior(
               log_prob_fn=lambda x: -0.5 * ||x - observation||^2,
               initial_position=observation,
               n_samples=rules.hmc_samples,
           )
        5. memory_response = recall_memory(observation)
        6. emergence_score = compute_emergence_score(
               perception, causal_graph, counterfactual, posterior, memory_response
           )

        返回: {
            'perception': dict,
            'causal_graph': dict,
            'counterfactual': dict,
            'posterior': dict,
            'memory': dict,
            'emergence_score': float,    # [0, 1]
        }
        """

    def compute_emergence_score(self, *outputs) -> float:
        """启发式涌现度: 高持久熵 + 高因果图密度 + 高记忆涌现 +
        低后验方差 → 高涌现度。"""
```

### 8.3 涌现度评分

涌现度是 `[0, 1]` 区间的启发式指标，组合：
- `0.3 * persistence_entropy`（拓扑复杂度）
- `0.2 * (n_edges / max_edges)`（因果图密度）
- `0.2 * memory.emerged`（若发生混沌游走取 1.0）
- `0.15 * (1 - posterior.std / dim)`（低不确定度 → 高分）
- `0.15 * (counterfactual.action / reference_action)`（作用量量级）

所有项在组合前归一化到 `[0, 1]`。权重总和 = 1.0。

---

## 9. EmergenceRules（`rules.py`）

```python
@dataclass
class EmergenceRules(DomainRules):
    """因果涌现引擎的先验。"""

    # 模块 A: 拓扑
    topology_max_points: int = 64          # 点云规模上限
    topology_max_dim: int = 2              # 最大同调维度
    topology_eps_steps: int = 50          # 过滤分辨率

    # 模块 B: 因果发现
    causal_method: str = "pc"              # 'pc' | 'lingam' | 'correlation'
    causal_significance: float = 0.05      # 独立性检验阈值
    causal_max_cond_set: int = 3          # PC 中条件集大小上限

    # 模块 C: 微分
    differential_max_iter: int = 100       # Gauss-Seidel 迭代上限
    differential_tol: float = 1e-6         # 收敛容差
    differential_dt: float = 0.01          # 时间步长

    # 模块 D: HMC
    hmc_step_size: float = 0.1
    hmc_n_leapfrog: int = 10
    hmc_samples: int = 100
    hmc_target_accept: float = 0.65        # Beskos 等的最优值

    # 模块 E: 混沌记忆
    chaotic_memory_capacity: int = 32
    chaotic_lorenz_sigma: float = 10.0
    chaotic_lorenz_rho: float = 28.0
    chaotic_lorenz_beta: float = 8.0 / 3.0
    chaotic_lyapunov_threshold: float = 0.9  # 涌现检测阈值

    # 引擎
    emergence_cycle_perturbation: float = 0.1  # 反事实扰动量
    seed: int | None = None
```

---

## 10. 测试策略

### 10.1 各模块测试

每个模块对应 `tests/test_causal_emergence_<module>.py`：

- **正确性:** 已知答案测试（如圆的 Betti 数 = [1, 1, 0]；HMC 恢复高斯均值）。
- **边界情况:** 空输入、单点、NaN、退化（零方差）、高维。
- **确定性:** 同种子产出相同输出。
- **API 契约:** 返回 dict 含期望键。

### 10.2 引擎集成测试

`tests/test_causal_emergence_engine.py`：
- 在合成观测上运行完整 `emergence_cycle`。
- 每个 facade 正确委托给底层模块。
- 线程安全：`self._lock` 下并发 `emergence_cycle` 调用。

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

### 10.4 确定性 fixture

每个测试文件包含：

```python
@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)
```

---

## 11. 军事级审查范围（15 维）

实现完成后，全面审查覆盖：

1. **数学正确性:** 算法对照参考文献（持久同调 Edelsbrunner；PC 算法 Spirtes-Glymour-Scheines；LiNGAM Shimizu 等；HMC Betancourt；Lorenz 吸引子 Lorenz 1963）。
2. **数值稳定性:** NaN/Inf 守卫、条件数、收敛阈值、有限差分步长选择。
3. **边界情况:** 空输入、单点、退化分布、全零数据、极端量级。
4. **线程安全:** `ZeroDataModel` facade 的 `_lock` 一致性；模块内无共享可变状态。
5. **API 契约一致性:** 返回 dict 键与规范一致；类型与注解一致；`Optional` 返回值有文档。
6. **性能:** 算法复杂度（如持久同调边界矩阵归约最坏 O(n^3)）；大输入退化路径（子采样、JIT kernel 回退）。
7. **测试覆盖盲区:** 分支覆盖、变异测试、参数边界测试。
8. **集成正确性:** 引擎组合模块无隐藏耦合；`ZeroDataModel` facade 正确委托；`_N_COGNITIVE_MODULES` 未变。
9. **zero-data 哲学符合度:** 无预训练权重、无外部数据集、无重型 ML 依赖；所有先验编码在 `EmergenceRules`。
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
- 6 个测试文件全部通过（每模块 ≥ 30 测试，引擎 ≥ 15 测试）。
- 现有 capability 测试仍通过（无回归）。
- `ruff check` 在所有新文件上无告警。
- 军事级审查完成并作为独立报告文档提交。
- `ZeroDataModel` 集成：`emergence_cycle` 在合成观测上端到端运行无错误。

---

## 14. 范围之外

- 生产级持久同调（不引入 gudhi / ripser）。
- GPU 加速（不引入 cupy / numba CUDA kernel）。
- 分布式 / 并行采样（不引入 MPI / Dask）。
- Web API 端点（不引入 FastAPI 路由）。
- 预训练模型或外部数据集。
- 实时性性能保证。
- 持久层（不保存/加载引擎状态）。
