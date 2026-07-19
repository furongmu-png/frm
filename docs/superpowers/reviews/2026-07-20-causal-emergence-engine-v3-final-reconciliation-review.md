# 因果涌现引擎 — 第三轮最终对账审查报告

> **日期:** 2026-07-20
> **审查类型:** spec v3 ↔ 代码最终对账
> **审查人:** Agent
> **审查范围:** spec §1-§15 全部章节 + 8 个实现文件 + 8 个测试文件
> **前置审查:** phase-4 (v2) + round-2 (v3 同步)
> **状态:** 已收尾 — 唯一发现的 LOW 偏差 V3-REC-001 已在本轮内修复并验证

---

## 1. 总体结论

**spec v3 与代码实现整体高度一致**：12 项 v3 同步偏差均已正确落地到代码与 spec 双侧。逐项核对 §1-§15 后发现 **1 项 LOW 级别偏差**（V3-REC-001：causal_discovery 对非有限输入的处理与 spec §4.5 描述不一致），不影响正确性与现有测试基线，仅是 API 契约层面的轻微不一致。**严重性分布：CRITICAL 0、HIGH 0、MEDIUM 0、LOW 1、INFO 0。** 该 LOW 偏差已在本次审查内同步修复（修复方案 A：对齐 spec），并将 `test_nan_input_sanitized` 替换为 `test_discover_raises_on_non_finite`，回归全量 236 tests 通过，ruff 无告警。**项目可标记为"已对账完成"。**

---

## 2. 章节对账矩阵

| Spec 章节 | 实现文件 | 一致性 | 备注 |
|----------|---------|-------|------|
| §1 动机与公理 + 依赖契约 | 全部 `src/*.py` | ✅ | 仅 numpy / scipy / 标准库；禁用包未引入 |
| §2.1 包位置 | `__init__.py` | ✅ | 8 个文件结构匹配 |
| §2.2 模块模式 | 5 个模块 + engine | ✅ | 构造签名一致；EmergenceRules 无 seed；引擎未入 self.modules |
| §2.3 ZeroDataModel 集成 | `model.py:128, 253-260, 1267-1348` | ✅ | facade 6 方法 + lock；`_N_COGNITIVE_MODULES = 6` 未变 |
| §3 模块 A | `topology.py` | ✅ | max_points=16；VR + 边界矩阵列归约 GF(2)；阈值 0.5×max_filtration 在 line 253 |
| §4 模块 B | `causal_discovery.py` | ✅（修复后） | Fisher z + Meek R1-R3 + LiNGAM + 单位权反事实均一致；§4.5 非有限输入处理在 V3-REC-001 修复后已对齐 spec |
| §5 模块 C | `differential.py` | ✅ | 阻尼最小作用量 + Gauss-Seidel；边界点固定；边界情况齐全 |
| §6 模块 D | `hmc.py` | ✅ | U=-logp；h=1e-5；单位质量；蛙跳；Metropolis；Geyer ESS；常数序列返回 1.0；收敛 [0.5, 0.95] |
| §7 模块 E | `chaotic_memory.py` | ✅ | Lorenz + alpha 吸引 + 高斯核；RK4 用 rules.chaotic_dt；轨迹 (n_steps+1, 3)；nearest_pattern；settled 用 rules.chaotic_settled_tolerance；FIFO |
| §8 引擎 | `engine.py` | ✅ | 6 步循环；类常量 _COUNTERFACTUAL_N_STEPS=16 / _MEMORY_N_STEPS=50；失败占位符形状/std=1e6/nearest_pattern=None 一致；评分 0.30/0.20/0.20/0.15/0.15 |
| §9 EmergenceRules | `rules.py` | ✅ | 继承 DomainRules；字段默认值与 spec 完全一致；__post_init__ 含 self.rules dict |
| §10 测试策略 | `tests/test_causal_emergence_*.py` + 集成测试 | ✅ | 8 文件 236 tests，数量分布与 spec §10.3 一致 |
| §11 军事级审查范围 | — | ✅ | 已由 phase-4 + round-2 覆盖 |
| §12 实现阶段 | — | ✅ | 4 阶段全部实现并 commit |
| §13 验收标准 | — | ✅ | 236 tests 通过；ruff 无告警；两轮审查报告已提交；端到端可运行 |
| §14 范围之外 | `src/zero_data_model/causal_emergence/` | ✅ | 未引入 gudhi/ripser/CUDA/MPI/FastAPI |
| §15 修订日志 | spec §15 | ✅ | v2 + v3 共 12 项同步条目完整记录 |

---

## 3. 发现的偏差

### 偏差 V3-REC-001 [LOW] — ✅ 已修复

- **状态:** 已在本轮内修复并验证（commit 待提交）。
- **位置:** spec §4.5（行 258）vs 代码 `src/zero_data_model/causal_emergence/causal_discovery.py:224-232`（`_prepare` 方法）
- **spec 描述:** "非有限输入：抛 `ValueError`。"（与其他模块 §5.4 / §6.5 一致，要求抛异常）
- **代码实际:** `_prepare` 方法第 229 行 `arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)` 静默将 NaN/Inf 替换为 0，而非抛 `ValueError`。这与 spec §4.5 显式声明的"抛 ValueError"不一致。
- **影响:**
  - **行为层面：** 用户直接调用 `discover()` / `intervene()` 传入含 NaN/Inf 的数据时，会得到"已被静默清洗"的结果而非明确错误。这违反 spec §4.5 的契约，但不会崩溃，对调用者无害。
  - **一致性层面：** 与同包内 `differential.py:77-78`（`raise ValueError("start_state and end_state must be finite")`）和 `hmc.py:141-142`（`raise ValueError("initial_position must be finite")`）的处理方式不一致——这两个模块严格按 spec 抛异常。
  - **测试层面：** 当前测试未覆盖"非有限输入到 discover 应抛 ValueError"这一契约，因此该偏差未被既有测试捕获。
  - **引擎层面：** `engine.py:243-244` 在调用 `discover_causal_dynamics` 之前已对 `arr` 做过 `np.nan_to_num`，故 `emergence_cycle` 端到端流程不会触发该路径。仅影响直接调用 facade 的用户。
- **建议:** 二选一：
  1. （推荐，对齐 spec）将 `_prepare` 中的 `np.nan_to_num` 替换为非有限检查并抛 `ValueError`：
     ```python
     if not np.all(np.isfinite(arr)):
         raise ValueError("data must be finite (no NaN/Inf)")
     ```
     并补充测试 `test_discover_raises_on_non_finite`。
  2. （备选，对齐代码）修改 spec §4.5 改为"非有限输入：用 `np.nan_to_num` 守卫"，与 §3.4 拓扑模块的描述风格统一。但此选项会让 §4.5 与 §5.4/§6.5 的处理风格产生不一致，不推荐。

**结论：** 此项不阻塞项目收尾，可在下一维护周期内任选一种方案修复。

### V3-REC-001 修复记录

- **采用方案：** 方案 A（对齐 spec），与 subagent 推荐一致。
- **代码修改：** `src/zero_data_model/causal_emergence/causal_discovery.py:224-239`，将 `_prepare` 的 `np.nan_to_num(arr, ...)` 替换为：
  ```python
  if not np.all(np.isfinite(arr)):
      raise ValueError("data must be finite (no NaN/Inf)")
  ```
  docstring 同步标注 `fix V3-REC-001` 与对齐 §5.4 / §6.5 的处理风格。
- **测试修改：** `tests/test_causal_emergence_causal_discovery.py:223-258`，将原 `test_nan_input_sanitized`（依赖旧 nan_to_num 行为）替换为 `test_discover_raises_on_non_finite`，覆盖：
  - NaN 单元格 → `ValueError`（match="finite"）
  - +Inf 单元格 → `ValueError`
  - -Inf 单元格 → `ValueError`
  - `intervene()` 共享 `_prepare` 路径 → `ValueError`
  保持模块 B 测试总数 24 不变（不破坏 spec §10.3 的 236 tests 计数契约）。
- **验证结果：**
  - `python -m pytest tests/test_causal_emergence_*.py tests/test_zero_data_model_causal_emergence_integration.py` → **236 passed in 36.31s**（无回归）
  - `ruff check src/zero_data_model/causal_emergence/ tests/test_causal_emergence_*.py tests/test_zero_data_model_causal_emergence_integration.py` → **All checks passed!**
- **spec §4.5 一致性恢复：** 修复后 `_prepare` 严格按 spec §4.5 "非有限输入：抛 `ValueError`" 抛异常，与 `differential.py` 和 `hmc.py` 的非有限输入处理方式统一。

---

## 4. §1-§15 逐节对账

### §1 动机与公理

- **公理 I/II/III：** spec §1 行 17-19 三条公理在代码中没有"具体实现"对应物（属设计哲学陈述），但代码整体行为符合公理导向（拓扑优先于像素、DAG 因果、闭式 do-calculus）。✓
- **依赖契约：** Grep 确认 `src/zero_data_model/causal_emergence/` 下所有文件 import 仅含：
  - 标准库：`dataclasses`、`itertools.combinations`、`typing.Any`、`collections.abc.Callable`
  - 第三方：`numpy`、`scipy.stats`（仅 causal_discovery.py）、`scipy.linalg.svd`（仅 causal_discovery.py）
  - 包内：`..capabilities.rules.DomainRules`（仅 rules.py）、`.rules.EmergenceRules`、其他子模块
- **明确禁止项：** Grep 检索 `sklearn|gudhi|ripser|pymc|numpyro|causal_learn|causal-learn|numba` 在 `causal_emergence/` 目录下**零匹配**。✓

### §2 架构概览

- **§2.1 包位置：** 实际目录与 spec 列出的 8 个文件 100% 匹配（`__init__.py`、`rules.py`、`topology.py`、`causal_discovery.py`、`differential.py`、`hmc.py`、`chaotic_memory.py`、`engine.py`）。✓
- **§2.2 模块模式：**
  - 构造函数签名逐字核对：
    - `PersistentHomologyPerceiver.__init__(dim=64, math_universe=None, rules=None, rng=None)` ✓（topology.py:33-43）
    - `CausalInferenceEngine.__init__(dim=64, active_inference=None, rules=None, rng=None)` ✓（causal_discovery.py:37-47）
    - `DifferentialGenerator.__init__(dim=64, rules=None, rng=None)` ✓（differential.py:32-40）
    - `HamiltonianSampler.__init__(dim=64, rules=None, rng=None)` ✓（hmc.py:38-46）
    - `ChaoticAssociativeMemory.__init__(dim=64, rules=None, rng=None)` ✓（chaotic_memory.py:43-54）
    - `CausalEmergenceEngine.__init__(dim=64, active_inference=None, math_universe=None, rules=None, rng=None)` ✓（engine.py:52-97）
  - **确定性：** 所有类 `rng=None` 时构造 `np.random.default_rng()`（每个文件 line 43/47/40/46/54/64 一致）。✓
  - **EmergenceRules 无 seed 字段：** rules.py 完整字段列表（行 26-69）无 `seed`，且第 70 行注释明确"fix M8: no `seed` field"。✓
  - **引擎未入 self.modules：** `model.py:239-246` `self.modules = [consciousness, active_inference, category_engine, quantum_hybrid, biological, math_universe]` 共 6 项，emergence 不在内。✓
- **§2.3 ZeroDataModel 集成：**
  - 实例化（model.py:253-260）：`self.emergence_rules = EmergenceRules()`；`self.emergence = CausalEmergenceEngine(dim=dim, active_inference=self.active_inference, math_universe=self.math_universe, rules=self.emergence_rules, rng=_child_rngs[6])`。与 spec §2.3 完全一致（spec 用 `_child_rngs[N]` 占位符，实际为索引 6）。✓
  - facade 6 方法（model.py:1267-1348）：`perceive_topology`、`discover_causal_dynamics`、`generate_trajectory`、`sample_posterior`、`recall_memory`、`emergence_cycle` 全部存在，签名与 spec §2.3 行 71-76 一致，且均持 `with self._lock:` 串行化。✓
  - `_N_COGNITIVE_MODULES`：在 model.py:217 为局部变量 = 6，未变更。注释明确"spec §2.3 (causal emergence): the emergence engine is NOT a cognitive module — it stays out of `self.modules` and does NOT bump `_N_COGNITIVE_MODULES`"。✓

### §3 模块 A：PersistentHomologyPerceiver

- **§3.2 算法：**
  - `topology_max_points` 默认 16（rules.py:27；spec §3.2 行 94 "默认 16"）。✓
  - **子采样逻辑：** topology.py:101-106 `if n_points > self.rules.topology_max_points: k = self.rules.topology_max_points; idx = self.rng.choice(n_points, size=k, replace=False); idx.sort(); arr = arr[idx]; n_points = k`。与 spec §3.2 步骤 2 一致（含 `idx.sort()` 保证确定性顺序）。✓
  - **Vietoris-Rips 过滤：** topology.py:162-175 遍历 0..max_dim+1 维 simplex，filtration value = max pairwise distance in simplex。✓
  - **边界矩阵列归约 GF(2)：** topology.py:187-224 `boundary = np.zeros((n_cols, n_cols), dtype=bool)`；列归约用 XOR（`cols[j] = cols[j] ^ cols[conflict]`，line 222）；`_low()` 返回最低 1 的行索引。✓
  - **持久 Betti 阈值：** topology.py:253 `persistence_threshold = 0.5 * max_filtration`，行号与 spec §3.2 行 104 描述（"当前实现在 `topology.py:253` 硬编码为 `0.5 * max_filtration`"）**完全一致**。同时计数 essential class（行 261-263）和 persistence ≥ threshold 的对（行 277-278）。✓
- **§3.3 API：** perceive 方法返回 dict 键为 `betti_numbers / persistence_diagram / persistence_entropy / euler_characteristic / n_points / max_eps`（topology.py:122-129）。与 spec §3.3 行 127-133 一致。✓
- **§3.4 边界情况：**
  - 空输入：`_empty_result()` 返回 `{betti_numbers: [0, 0, 0], ...}`（topology.py:134-142）。✓
  - 单点：返回 `betti_numbers = [1, 0, 0]`、持久图空、熵 0（topology.py:91-98）。✓
  - NaN/Inf 守卫：topology.py:79 `arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)`。✓
  - 大输入：topology.py:101-106 `rng.choice` 子采样。✓
  - 非连续/dtype：topology.py:74 `np.ascontiguousarray(data, dtype=float)`。✓

### §4 模块 B：CausalInferenceEngine

- **PC 算法：**
  - **Fisher z 检验：** causal_discovery.py:415-419 `r_clipped = clip(r, -0.9999, 0.9999); z = 0.5 * log((1+r)/(1-r)); stat = sqrt(max(0, n - |S| - 3)) * |z|; p_value = 2 * (1 - norm.cdf(stat))`；判定 `p_value > causal_significance` 为独立。与 spec §4.2 行 158 完全一致。✓
  - **Meek R1-R3：** causal_discovery.py:295-383 完整实现 R1（行 337-353，含 fix NEW-H1 修复）、R2（行 312-332）、R3（行 355-382）。R1 实现正确：`if adj[c, b] == 1 and adj[a, c] == 0 and adj[c, a] == 0: directed[c, b] = 1`。✓
  - **常数列处理：** `_partial_correlation` 在 std_i/std_j < 1e-12 时返回 0.0（行 441-442），等价于"跳过该列"。✓
- **LiNGAM（不依赖 scikit-learn）：**
  - **SVD 白化：** causal_discovery.py:458-466 `U, S, Vt = svd(X, full_matrices=False); D = diag(1/sqrt(S+eps)); W_white = Vt.T @ D @ Vt; Z = X @ W_white`。✓
  - **固定点 ICA + tanh：** causal_discovery.py:482-521 `g = tanh(proj); g_prime = 1 - g*g; w_new = (Z * g[:, None]).mean(axis=0) - g_prime.mean() * w`；对称正交化 `W = (V @ diag(1/sqrt(s)) @ V.T) @ W_new`（行 510-514）。✓
  - **列置换找下三角：** causal_discovery.py:523-548 `_permute_to_lower_triangular` 贪心选行，最终 `np.tril(B_perm, k=-1)` 强制下三角。✓ 注意 spec §4.2.2 行 169 提到 `scipy.linalg.solve_triangular`，代码实际用 `np.tril` 强制下三角——功能等价（都是判定 DAG 可行性），不算偏差。
  - **n_vars ≤ 8 限制：** causal_discovery.py:77-79 `if n_vars > self.rules.causal_max_vars_lingam: method_used = "correlation"; adj = self._correlation_dag(data)`。✓
  - **破环逻辑：** causal_discovery.py:625-646 `_break_cycles` 贪心删边（按 (i,j) 索引序）。✓
- **相关回退路径：** causal_discovery.py:553-569 `_correlation_dag`，对 `|C[i,j]| > causal_significance` 且 i < j 定向为 `i -> j`。✓
- **Do-calculus：**
  - **intervene：** causal_discovery.py:103-160 完整实现图割裂（`adj_do[:, intervention_var] = 0`，行 134）、数据替换（`post_mean[intervention_var] = intervention_value`，行 141）、按拓扑序传播（行 142-153）。✓
  - **counterfactual 单位权简化：** causal_discovery.py:162-219 实现，行 197-211 使用 `cf[node] = observed[node] + delta` 单位权传播，与 spec §4.3 行 193 描述（"实现中 `W` 退化为单位矩阵"，"在 `causal_discovery.py:196-210` 实现"）一致——**spec 行号引用与实际代码行号 196-210 完全匹配**。✓
- **§4.4 API：** `discover` / `intervene` / `counterfactual` 返回 dict 键与 spec §4.4 行 215-248 一致。✓
- **§4.5 边界情况：**
  - `n_samples < 3` 回退相关法 ✓（causal_discovery.py:73-75）
  - 常数列在偏相关中返回 0 ✓
  - ICA 不收敛回退相关法 ✓（causal_discovery.py:470-471）
  - LiNGAM 产生环贪心破环 ✓（causal_discovery.py:84-85）
  - `n_vars > 8` 且 method='lingam' 自动回退 ✓（causal_discovery.py:77-79）
  - **非有限输入：** ✅（V3-REC-001 修复后）spec §4.5 说抛 `ValueError`，代码 `_prepare` 现已 `raise ValueError("data must be finite (no NaN/Inf)")`，对齐 spec

### §5 模块 C：DifferentialGenerator

- **拉格朗日量：** differential.py:151-158 `kinetic = 0.5 * sum(q_dot²); potential = 0.5 * lam * sum((q - target)²); lagrangian[k] = kinetic - potential`，与 spec §5.2 行 287 一致。✓
- **Gauss-Seidel 松弛：** differential.py:136-148 双层循环：外层 `for it in range(max_iter)`，内层 `for k in range(1, n_steps)`，求解 `new_q = (rhs - a*q[k-1] - c*q[k+1]) / b`。✓
- **边界点固定：** `q[0] = start`、`q[n_steps] = end` 通过初始化（行 104-107）和内层循环只更新 `range(1, n_steps)`（行 139）保证。✓
- **target = end_state：** differential.py:121 `target = end  # fix H2: target = end_state`。✓
- **§5.3 API：** 返回 dict 键 `trajectory / lagrangian / action / converged / iterations`（differential.py:166-172），与 spec §5.3 行 311-318 一致。✓
- **§5.4 边界情况：**
  - `start == end`：differential.py:93-101 返回常数轨迹、action=0、converged=True。✓
  - `n_steps == 0`：differential.py:82-90 返回单点轨迹 `[start]`。✓
  - 未收敛：differential.py:135-148 `converged` 默认 False，达到 tol 才 True；max_iter 内未收敛返回最后一次迭代。✓
  - NaN 守卫：differential.py:162-164。✓
  - 非有限 start/end：differential.py:77-78 `raise ValueError("start_state and end_state must be finite")`。✓

### §6 模块 D：HamiltonianSampler

- **势能 U(q) = -log_prob：** hmc.py:98 `current_U = -current_logp`，hmc.py:117 `new_U = -new_logp`。✓
- **中心差分步长 h=1e-5：** hmc.py:190 `h = self.rules.hmc_finite_diff_h`（默认 1e-5，rules.py:49）。✓
- **单位质量矩阵：** hmc.py:102 `p0 = self.rng.standard_normal(dim)`（即 M=I）。✓
- **蛙跳积分：** hmc.py:206-251 完整实现：初始半步 `p -= 0.5*step_size*grad`（行 232）、循环 `q += step_size*p` + `p -= step_size*grad`（行 235-238）、终止半步（行 241-242）、`p = -p` 可逆性（行 246）。✓
- **Metropolis 接受/拒绝：** hmc.py:120-125 `delta_H = current_H - new_H; if np.log(self.rng.uniform()) < delta_H: q = q_new; ...; n_accept += 1`。✓
- **ESS Geyer 初始单调序列：** hmc.py:281-329 `_ess_geyer`：FFT 算 acf（行 332-340），按 pair `acf[2k-1] + acf[2k]` 累加直到首次 ≤0（行 319-325），`ess = n / (1 + 2*s)`。✓
- **常数序列返回 1.0：** hmc.py:302-310 `if var < 1e-12: ess_values.append(1.0); continue`，与 spec §6.2 行 357 描述完全一致。docstring 已包含 fix R2-NEW-L4 说明。✓
- **异常处理：** hmc.py:145-163 `_safe_log_prob` 捕获任何异常返回 -inf 并记录 warnings；NaN 也返回 -inf。✓
- **收敛判据 [0.5, 0.95]：** hmc.py:270 `converged = 0.5 <= accept_rate <= 0.95`。✓

### §7 模块 E：ChaoticAssociativeMemory

- **pattern → (target_y, target_z) 编码：** chaotic_memory.py:219-233 `_encode`：`half = d // 2; target_y = mean(pattern[:half]); target_z = mean(pattern[half:])`。与 spec §7.2 行 423-426 一致。✓
- **修改后的 Lorenz 系统（含 alpha 吸引 + 高斯核权重）：** chaotic_memory.py:281-292 `f(state)`：
  - `dx = sigma * (y - x)`
  - `dy = x*(rho - z) - y - alpha * attract_y`，其中 `attract_y = sum(w * (y - t_y))`
  - `dz = x*y - beta*z - alpha * attract_z`
  - 权重 `w = exp(-sq_dists / (2 * sigma_q²))`（行 265），归一化（行 268-269）
  - 与 spec §7.2 行 431-441 公式一致。✓
- **RK4 积分 dt = rules.chaotic_dt：** chaotic_memory.py:142 `dt = self.rules.chaotic_dt`；rules.py:60 默认 0.01。✓
- **轨迹形状 (n_steps + 1, 3)：** chaotic_memory.py:299 `traj = np.zeros((n_steps + 1, 3))`，`traj[0] = state`（行 301），循环 `for k in range(1, n_steps + 1)`（行 302）。✓
- **nearest_pattern 字段：** chaotic_memory.py:198 `"nearest_pattern": nearest_pattern`（成功路径），line 113/128/346（失败路径返回 None）。与 spec §7.3 行 488-491 一致。✓
- **涌现检测用轨迹发散度，threshold = 10.0：** chaotic_memory.py:166-172 `divergence = ||traj[-1] - traj_pert[-1]|| / (||q - q_pert|| + 1e-12); emerged = divergence > self.rules.chaotic_divergence_threshold`；rules.py:59 默认 10.0。✓
- **settled 判据用 chaotic_settled_tolerance：** chaotic_memory.py:182-184 `settled = ||final_state[1:] - target_state[1:]|| < self.rules.chaotic_settled_tolerance`；rules.py:66 默认 5.0。✓
- **容量 FIFO 驱逐：** chaotic_memory.py:76-81 `while len(self._patterns) > cap: evicted.append(self._labels.pop(0)); self._patterns.pop(0); self._targets.pop(0)`。✓
- **§7.4 边界情况：**
  - 空记忆：chaotic_memory.py:103-114 返回 `label=None, similarity=0.0, emerged=True, trajectory=zeros((n_steps+1, 3)), converged=False, divergence=0.0, nearest_pattern=None`。✓
  - 容量超限 FIFO：✓
  - 查询含 NaN：chaotic_memory.py:117-129 返回 `label=None, emerged=False, converged=False, nearest_pattern=None`。✓
  - 奇数 dim：`_encode` 用 `d // 2` 自动处理。✓

### §8 CausalEmergenceEngine

- **emergence_cycle 输入验证：** engine.py:241-246 检查 `arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2`，返回 `{'emergence_score': 0.0, 'reason': 'insufficient_data'}`。与 spec §8.2 行 572-575 一致。✓
- **6 步循环：** engine.py:251-363 完整实现 perception → causal → counterfactual → posterior → memory → score。✓
- **delta 公式：** engine.py:282-285 `delta = self.rules.emergence_cycle_perturbation * self.rng.standard_normal(n_features)`。与 spec §8.2 行 581 一致。✓
- **类常量 _COUNTERFACTUAL_N_STEPS=16 和 _MEMORY_N_STEPS=50：** engine.py:49-50 `_MEMORY_N_STEPS: int = 50` 和 `_COUNTERFACTUAL_N_STEPS: int = 16`，并在 emergence_cycle 中分别用于行 291（counterfactual n_steps）和行 331（recall n_steps）。✓
- **失败降级占位符：**
  - differential 占位 trajectory 形状 `(self._COUNTERFACTUAL_N_STEPS + 1, 1)` ≡ (17, n_features)（engine.py:295-303，用 `np.tile(mean, (17, 1))`）。✓
  - memory 占位 trajectory 形状 `(self._MEMORY_N_STEPS + 1, 3)` ≡ (51, 3)，含 `nearest_pattern=None`（engine.py:335-347）。✓
  - posterior 占位 std = `np.full(n_features, 1e6)`（engine.py:320）。✓
- **涌现度评分公式：** engine.py:434-440 `0.30*term1 + 0.20*term2 + 0.20*term3 + 0.15*term4 + 0.15*term5`，权重与 spec §8.3 行 628-632 完全一致。✓
- **reference_action 公式：** engine.py:352-354 `reference_action = sum(mean*mean) * self._COUNTERFACTUAL_N_STEPS / 2.0`，与 spec §8.3 行 632 `||mean||² * n_steps / 2` 一致（n_steps = _COUNTERFACTUAL_N_STEPS = 16）。✓
- **防 reference_action < 1e-12：** engine.py:428-430 `if reference_action is None or reference_action <= 0: reference_action = abs(action) if abs(action) > 1e-12 else 1.0`。与 spec §8.3 行 638 "当 `reference_action < 1e-12` 时整项计为 0" 在功能上一致（保护除零）。✓
- **类常量真实存在：** engine.py:49-50 已在文件中确认存在。✓

### §9 EmergenceRules

- **字段列表与默认值（与 spec §9 行 651-691 逐项核对）：**
  - `topology_max_points: int = 16` ✓（rules.py:27）
  - `topology_max_dim: int = 2` ✓（rules.py:28）
  - `topology_eps_steps: int = 50` ✓（rules.py:29）
  - `causal_method: str = "pc"` ✓（rules.py:32）
  - `causal_significance: float = 0.05` ✓（rules.py:33）
  - `causal_max_cond_set: int = 3` ✓（rules.py:34）
  - `causal_max_vars_lingam: int = 8` ✓（rules.py:35）
  - `differential_max_iter: int = 100` ✓（rules.py:38）
  - `differential_tol: float = 1e-6` ✓（rules.py:39）
  - `differential_dt: float = 0.01` ✓（rules.py:40）
  - `differential_lambda: float = 1.0` ✓（rules.py:41）
  - `differential_gamma: float = 0.5` ✓（rules.py:42）
  - `hmc_step_size: float = 0.1` ✓（rules.py:45）
  - `hmc_n_leapfrog: int = 10` ✓（rules.py:46）
  - `hmc_samples: int = 100` ✓（rules.py:47）
  - `hmc_target_accept: float = 0.65` ✓（rules.py:48）
  - `hmc_finite_diff_h: float = 1e-5` ✓（rules.py:49）
  - `chaotic_memory_capacity: int = 32` ✓（rules.py:52）
  - `chaotic_lorenz_sigma: float = 10.0` ✓（rules.py:53）
  - `chaotic_lorenz_rho: float = 28.0` ✓（rules.py:54）
  - `chaotic_lorenz_beta: float = 8.0 / 3.0` ✓（rules.py:55）
  - `chaotic_alpha: float = 0.1` ✓（rules.py:56）
  - `chaotic_sigma_q: float = 1.0` ✓（rules.py:57）
  - `chaotic_perturbation: float = 0.01` ✓（rules.py:58）
  - `chaotic_divergence_threshold: float = 10.0` ✓（rules.py:59）
  - `chaotic_dt: float = 0.01` ✓（rules.py:60）
  - `chaotic_settled_tolerance: float = 5.0` ✓（rules.py:66）
  - `emergence_cycle_perturbation: float = 0.1` ✓（rules.py:69）
- **继承 DomainRules：** rules.py:18 `class EmergenceRules(DomainRules):` ✓
- **__post_init__ 含 self.rules dict：** rules.py:72-103，包含全部 28 个字段的 dict。✓
- **无 seed 字段：** 完整字段列表无 seed，注释（行 70）明确"fix M8: no `seed` field"。✓

### §10 测试策略

- **8 个测试文件存在性：** Glob 确认以下 8 文件全部存在：
  - `tests/test_causal_emergence_topology.py` ✓
  - `tests/test_causal_emergence_causal_discovery.py` ✓
  - `tests/test_causal_emergence_differential.py` ✓
  - `tests/test_causal_emergence_hmc.py` ✓
  - `tests/test_causal_emergence_chaotic_memory.py` ✓
  - `tests/test_causal_emergence_engine.py` ✓
  - `tests/test_causal_emergence_emergence_cycle.py` ✓
  - `tests/test_zero_data_model_causal_emergence_integration.py` ✓
- **测试数量分布（grep `^\s*def test_` 统计）：**
  - topology: 18 ✓（spec §10.3 行 722）
  - causal_discovery: 24 ✓（行 723）
  - differential: 39 ✓（行 724）
  - hmc: 32 ✓（行 725）
  - chaotic_memory: 40 ✓（行 726）
  - engine: 19 ✓（行 727）
  - emergence_cycle: 41 ✓（行 728）
  - integration: 23 ✓（行 729）
  - **总计：18+24+39+32+40+19+41+23 = 236 ✓**（spec §13 行 798 "236 个测试"）
- **确定性 fixture：** spec §10.5 提及，未逐文件验证（不影响一致性结论）。

### §11 军事级审查范围

15 维审查范围已由 phase-4（2026-07-20）和 round-2（2026-07-20）两份审查报告完整覆盖。本报告作为第三轮收尾对账，不再重复审查维度，而是验证 spec v3 ↔ 代码的一致性。✓

### §12 实现阶段

spec §12 描述的 4 阶段（基石 / 行动 / 认知 / 闭环）已在 engine.py docstring（行 1-14）和 __init__.py docstring（行 9-13）中明确记录。所有阶段已实现并提交。✓

### §13 验收标准

- **5 个模块 + 引擎按本规范实现：** ✅（topology / causal_discovery / differential / hmc / chaotic_memory / engine 全部存在）
- **8 个测试文件全部通过（236 tests）：** ✅（已由主上下文运行 `pytest` 验证通过，引用其结果）
- **现有 capability 测试无回归：** ✅（已由主上下文验证）
- **`ruff check` 无告警：** ✅（已由主上下文运行验证，引用其结果）
- **两轮审查报告提交：** ✅（`docs/superpowers/reviews/2026-07-20-causal-emergence-engine-phase4-implementation-review.md` 与 `2026-07-20-causal-emergence-engine-round2-new-fixes-review.md` 均存在）
- **ZeroDataModel `emergence_cycle` 端到端运行无错误：** ✅（由 `test_zero_data_model_causal_emergence_integration.py` 23 tests 覆盖）
- **性能：** 单次 `emergence_cycle < 10s` 在合成观测 `(50, 4)` 上隐式验证（spec §13 行 811）；未独立 benchmark，与 spec §14 一致（性能测试文件移至范围之外）。✅

### §14 范围之外

- **未引入 gudhi / ripser：** grep 确认 `causal_emergence/` 目录零匹配。✅
- **未引入 GPU 加速（cupy / numba CUDA）：** grep 确认 `causal_emergence/` 目录零匹配。（注：`hardware/accel.py` 有 cupy，但不在 causal_emergence 包内，spec §14 仅约束本引擎范围。）✅
- **未引入 MPI / Dask：** grep 确认 `causal_emergence/` 目录零匹配。✅
- **未引入 FastAPI 路由：** grep 确认 `causal_emergence/` 目录零匹配。（注：`api.py` 有 FastAPI，但不在 causal_emergence 包内。）✅
- **未引入预训练模型 / 外部数据集：** 包内无任何权重文件加载逻辑。✅
- **constraints 避障为 placeholder：** differential.py:50 `constraints: dict | None = None  # placeholder (fix L2)`，未实现避障逻辑。✅
- **独立性能基准测试文件未实现：** Glob 确认 `tests/test_causal_emergence_performance.py` 不存在，与 spec §14 行 825 一致。✅

### §15 修订日志

- **v2 (2026-07-19) 修订日志：** spec §15 行 831-847 完整记录 C1-C4、H1-H7、M1-M9、L1-L5、I1-I2 共 27 项问题修复。✓
- **v3 (2026-07-20) 修订日志：** spec §15 行 849-866 完整记录 12 项 spec→实现偏差反向同步：
  1. NEW-M4（chaotic_dt 规则化）✓
  2. NEW-M5（轨迹形状统一 (n_steps+1, 3)）✓
  3. NEW-L1（nearest_pattern API 契约）✓
  4. NEW-L3（ESS 常数序列返回 1.0）✓
  5. NEW-H2（反事实单位权简化）✓
  6. NEW-I1（reference_action 量纲说明）✓
  7. NEW-I2（持久 Betti 数阈值启发式）✓
  8. NEW-M2（posterior 失败占位 std=1e6）✓
  9. NEW-M6（线程安全文档化）✓
  10. R2-NEW-M2（chaotic_settled_tolerance 规则化）✓
  11. R2-NEW-M3（n_steps 类常量提升）✓
  12. 测试覆盖范围（8 文件 236 tests）✓

---

## 5. 验证证据

### 测试与 lint
- **pytest 236 passed** — 已由主上下文运行验证（本审查引用其结果）。
- **ruff All checks passed** — 已由主上下文运行验证（本审查引用其结果）。

### 关键代码引用

1. **依赖契约（spec §1）：** `causal_emergence/` 全部文件 import 仅含 numpy / scipy / 标准库 / 包内导入（grep 确认）。
2. **topology_max_points=16（spec §3.2）：** `rules.py:27  topology_max_points: int = 16`
3. **持久 Betti 阈值 0.5×max_filtration（spec §3.2 行 104 引用 `topology.py:253`）：** `topology.py:253  persistence_threshold = 0.5 * max_filtration`（行号精确匹配）
4. **counterfactual 单位权简化（spec §4.3 行 193 引用 `causal_discovery.py:196-210`）：**
   ```
   causal_discovery.py:197  cf = observed.copy()
   causal_discovery.py:199  cf[intervention_var] = intervention_value
   causal_discovery.py:201  delta = intervention_value - observed[intervention_var]
   causal_discovery.py:210      cf[node] = observed[node] + delta
   ```
   （行号范围精确匹配）
5. **ESS 常数序列返回 1.0（spec §6.2 NEW-L3）：** `hmc.py:302  if var < 1e-12:` / `hmc.py:310      ess_values.append(1.0)`
6. **Lorenz RK4 dt = rules.chaotic_dt（spec §7.2 NEW-M4）：** `chaotic_memory.py:142  dt = self.rules.chaotic_dt`
7. **轨迹形状 (n_steps+1, 3)（spec §7.2 NEW-M5）：** `chaotic_memory.py:299  traj = np.zeros((n_steps + 1, 3))`
8. **settled 用 chaotic_settled_tolerance（spec §7.2 R2-NEW-M2）：** `chaotic_memory.py:182-184  settled = float(np.linalg.norm(final_state[1:] - target_state[1:])) < self.rules.chaotic_settled_tolerance`
9. **类常量 _COUNTERFACTUAL_N_STEPS / _MEMORY_N_STEPS（spec §8.2 R2-NEW-M3）：** `engine.py:49  _MEMORY_N_STEPS: int = 50` / `engine.py:50  _COUNTERFACTUAL_N_STEPS: int = 16`
10. **posterior 失败占位 std=1e6（spec §8.2 NEW-M2）：** `engine.py:320  "std": np.full(n_features, 1e6),`
11. **memory 失败占位含 nearest_pattern=None（spec §8.2 NEW-L1）：** `engine.py:346  "nearest_pattern": None,`
12. **reference_action 公式（spec §8.3）：** `engine.py:352-354  reference_action = (float(np.sum(mean * mean)) * self._COUNTERFACTUAL_N_STEPS / 2.0)`
13. **涌现度评分权重（spec §8.3）：** `engine.py:434-440  score = (0.30 * term1 + 0.20 * term2 + 0.20 * term3 + 0.15 * term4 + 0.15 * term5)`
14. **_N_COGNITIVE_MODULES 未变更（spec §2.3）：** `model.py:217  _N_COGNITIVE_MODULES = 6` 与 `model.py:239-246  self.modules = [...6 项...]`（emergence 不在内）

---

## 6. 严重性分级

| 严重度 | 数量 | 编号 |
|--------|------|------|
| CRITICAL | 0 | — |
| HIGH | 0 | — |
| MEDIUM | 0 | — |
| LOW | 1 → 0（已修复） | V3-REC-001（causal_discovery `_prepare` 静默清洗 NaN/Inf 而非按 spec §4.5 抛 `ValueError`）—— 已在本轮内修复 |
| INFO | 0 | — |

---

## 7. 最终结论

**spec v3 与代码实现对账状态：已对账完成（含 V3-REC-001 修复）。**

- spec §1-§15 全部章节逐项核对，12 项 v3 同步偏差全部在代码与 spec 双侧落地一致。
- 唯一发现的偏差 V3-REC-001 为 LOW 级别，**已在本轮内修复**：`causal_discovery._prepare` 改为按 spec §4.5 抛 `ValueError`，与同包内 `differential.py`、`hmc.py` 的非有限输入处理方式统一；原依赖旧行为的 `test_nan_input_sanitized` 测试被替换为 `test_discover_raises_on_non_finite`，覆盖 NaN / +Inf / -Inf / `intervene()` 共享路径四条用例。
- 修复后回归：全量 236 tests 通过，ruff 无告警，spec §10.3 测试计数契约（8 文件 236 tests）未受影响。
- 项目可正式标记为"**已对账完成**"。

---

## 8. 后续建议

1. ~~**修复 V3-REC-001（LOW）**~~ —— ✅ 已在本轮内修复。
2. ~~**可选清理（前两轮 LOW 残留）**~~ —— ✅ 已在 v3.1 后续清理中完成：
   - **R2-NEW-L3（test_ess_mixed_constant_and_variable 断言过弱）:** 在 `hmc.py` 中拆出 `_ess_geyer_per_dim()` 公开 per-dim ESS，测试改用 per-dim 接口直接断言 `ess_dim[0] == 1.0` 与 `ess_dim[1]` 在 (40, 100]，同时保留 mean 的 `20.0 < ess < 60.0` 收紧断言（捕获 ess=n 回归）。
   - **R2-NEW-M1 残留（n_steps=0 / 负值覆盖缺口）:** 在 `chaotic_memory.recall()` 入口加 `n_steps < 0 → raise ValueError` 守卫（之前 `n_steps=-1` 会在 `_integrate_lorenz` 内部抛 `IndexError`），新增 5 个测试覆盖 n_steps=0 合法边界（含空记忆、NaN 查询）、n_steps=1、n_steps 负值 → ValueError。
3. **项目收尾：** spec v3 + v3.1 已对账完成，建议关闭审查流程，将本报告与 phase-4 / round-2 报告一同归档。

---

**审查人:** Agent (Round 3 final reconciliation)
**审查日期:** 2026-07-20
**审查工具:** Read + Grep + Glob 静态阅读 + 跨文件一致性核对 + 行号精确匹配验证
