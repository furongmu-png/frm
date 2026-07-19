# 因果涌现引擎设计规范 — 超级军事级审查报告

> **审查对象:** [docs/superpowers/specs/2026-07-19-causal-emergence-engine-design.md](file:///workspace/docs/superpowers/specs/2026-07-19-causal-emergence-engine-design.md)
> **审查日期:** 2026-07-19
> **审查者:** Agent
> **审查阶段:** 设计阶段（实现前）
> **审查范围:** spec 第 11 节定义的 15 维度，外加跨节一致性、可行性、可实现性

---

## 执行摘要

对设计规范进行了 15 维 + 跨节一致性的全面审查，共发现 **27 项问题**：

| 严重性 | 数量 |
|--------|------|
| CRITICAL | 4 |
| HIGH | 7 |
| MEDIUM | 9 |
| LOW | 5 |
| INFO | 2 |

**结论：spec 不可直接进入实现。** 4 项 CRITICAL 问题涉及算法不可行、依赖冲突、接口不匹配，必须在编码前修复。建议修订 spec 后再次审查，然后进入实现。

---

## CRITICAL 级别问题（必须修复才能进入实现）

### C1. 持久同调算法复杂度不可行

**位置:** §3.2 第 3-4 步、§9 `topology_max_points: int = 64`

**问题:** 边界矩阵列归约的复杂度是 O(s³)，其中 s 是**单形数**而非点数。Vietoris-Rips 复形在 n 个点上的单形数为：
- 0 单形：C(n,1) = n
- 1 单形：C(n,2)
- 2 单形：C(n,3)
- 3 单形：C(n,4)

对 n=64 点、计算到 2 维同调（需要 3 单形），单形数 = 64 + 2016 + 41664 = 43744。O(s³) ≈ 8.4e13 次运算。即便对 n=32，单形数仍达 32 + 496 + 4960 = 5488，O(s³) ≈ 1.65e14。在合理时间内不可能完成。

**修复建议:**
1. 将 `topology_max_points` 降为 **16**（n=16 时单形数 16+120+560=696，O(s³)≈3.4e8，秒级可行）。
2. 对 16 < n ≤ 64 的输入，**只计算 betti_0**（连通分量，O(n²) union-find）+ **betti_1 的近似**（环数 = edges - vertices + components，O(n²)）。
3. 对 n > 64 的输入，确定性子采样到 16 点。
4. 在 spec §3.2 明确写"分层降级策略"，删除"n ≤ 32 可行"的错误陈述。

---

### C2. LiNGAM 依赖冲突

**位置:** §4.2.2、§1（zero-data 哲学）

**问题:** spec 第 4.2.2 节说"对数据运行 FastICA"。FastICA 的标准实现是 `sklearn.decomposition.FastICA`，但：
- §1 明确说"无重型 ML 依赖（明确排除 gudhi / pymc / causal-learn / ripser）"——但没明确提及 scikit-learn。
- spec 同时声称"仅用 numpy + scipy + 标准库"。
- 自实现 FastICA 需要数百行代码（白化 + 牛顿迭代 + 收敛判定），且很容易出错。

这是依赖契约的根本性冲突：要么放弃 LiNGAM，要么明确允许 scikit-learn。

**修复建议:** 在 §4.2.2 改为：
> LiNGAM 方法需要独立成分分析。本实现采用**简化版 LiNGAM**：用 `scipy.linalg.svd` 做白化，用 numpy 实现固定点 ICA（Hyvärinen 1999），仅支持 4 个变量的标准情形。当变量数 > 4 或 ICA 不收敛时，回退到相关法。

或更激进：**完全移除 LiNGAM**，只保留 PC + 相关法两种方法。后者更符合 zero-data 极简哲学。

---

### C3. emergence_cycle 接口不匹配

**位置:** §8.2 步骤 1-2

**问题:** emergence_cycle 内部调用：
1. `perceive_topology(observation)` — 但 §3.4 说 1D 输入视为单点点云，返回 `betti=[1,0,0]`，零信息量。
2. `discover_causal_dynamics(observation)` — 但 §4.4 说输入是 `(n_samples, n_vars)`。单条 1D observation 无法构成数据矩阵。

**根本性问题:** spec 没有定义 `emergence_cycle` 期望的 `observation` 形状。如果 observation 是 2D `(n_samples, n_features)`，那么 §3.4 的"1D 视为单点"规则会让 cycle 在感知阶段退化。

**修复建议:**
1. 在 §8.2 开头明确：`observation: np.ndarray (n_samples, n_features)`，要求 `n_samples >= 2, n_features >= 2`。
2. 对 1D 输入，emergence_cycle 直接返回零分：`{'emergence_score': 0.0, 'reason': 'insufficient_data'}`。
3. 在 §8.2 步骤 2 显式处理单样本情形：跳过因果发现，返回空 DAG。

---

### C4. 混沌记忆编码方案缺失

**位置:** §7.2 第 1 步、§7.2 第 4 步

**问题:** 两个关键缺陷：

1. **编码方案未定义:** "`(target_y, target_z)` 由模式推导"——但 pattern 是 `dim=64` 维向量，target 只有 2 维。映射函数完全缺失。没有这个映射，store 方法无法实现。

2. **容量声明错误:** "理论无上限（每个模式为独立吸引子）"——这是错误的。修改后的 Lorenz 系统只有一个奇异吸引子，多个 target 点不会各自产生独立吸引盆，而是互相干扰，最终只有一个或少数几个吸引子存活。实际容量远低于"无限"。

3. **混沌可能消失:** 加了 `alpha * (y - target_y)` 项后，当 alpha 较大时系统会塌缩到 target 附近的固定点，丧失混沌特性。spec 没给 alpha 取值。

**修复建议:**
1. 在 §7.2 第 1 步明确定义编码：`target_y = mean(pattern[:dim//2]), target_z = mean(pattern[dim//2:])`（或类似的降维）。
2. 删除"理论无上限"陈述，改为"实际容量约等于 `rules.chaotic_memory_capacity`，超出时 FIFO 驱逐"。
3. 明确 `alpha = 0.1`（小到不破坏混沌，又能提供弱吸引）。
4. 在 §7.2 增加稳定性说明：alpha 太大会消除混沌，太小则记忆无效。

---

## HIGH 级别问题（算法不完整或物理不正确）

### H1. 微分方程物理模型与目的不符

**位置:** §5.2 第 2-4 步

**问题:** 拉格朗日量定义为 `L = 0.5*||q_dot||² - 0.5*||q - target||²`，对应欧拉-拉格朗日方程：
```
q'' = -(q - target)
```
这是**谐振子方程**，解是正弦波 `q(t) = target + A*cos(t) + B*sin(t)`。轨迹会在 start 和 end 之间**振荡**，而不是单调收敛到 end。这与"生成最短路径轨迹"的目的完全不符。

**修复建议:** 改用**阻尼拉格朗日量**或**测地线**：
- 方案 A（推荐）：用测地线（最小作用量 = 最短路径），L = `0.5*||q_dot||²`，无势能。解是线性插值（平凡但正确）。
- 方案 B：加阻尼项，L = `0.5*||q_dot||² - 0.5*||q - target||² - gamma*||q_dot||²`，让轨迹单调收敛。
- 方案 C：用样条插值作为初值，然后做最小作用量修正。

明确选择一种并写入 spec。

---

### H2. 微分方程的 target 未定义

**位置:** §5.2 第 4 步

**问题:** 离散方程 `(q[k+1] - 2*q[k] + q[k-1]) / dt² = -(q[k] - target)` 中的 `target` 没有定义。是 `end_state`？是某个中间点？

**修复建议:** 明确 `target = end_state`，并在方程中替换变量名。

---

### H3. 反事实算法不完整

**位置:** §4.3

**问题:** "对单条观测，通过 DAG 上的信念传播计算干预下的后验均值"——信念传播（belief propagation）需要：
1. 每个节点的条件概率分布（CPD）。spec 没说怎么从数据估计 CPD。
2. DAG 的因子分解结构。
3. 消息传递调度顺序。

spec 完全没给这些细节。

**修复建议:** 简化为**前门调整 / 后门调整的线性近似**：
> 对线性高斯 DAG（权重 W = 从数据回归得到的邻接系数），反事实 = `observed - W[:, var] * (observed[var] - intervention_value)`。

这是 do-calculus 在线性情形下的闭式解，避开了信念传播的复杂性。

---

### H4. 涌现度评分的归一化基准未定义

**位置:** §8.3

**问题:** `0.15 * (counterfactual.action / reference_action)` 中的 `reference_action` 没有定义。是常数？是从 observation 推导？是 `||observation||²`？

**修复建议:** 明确 `reference_action = ||observation||² * n_steps / 2`（即直线轨迹的作用量），作为归一化基准。

---

### H5. 后验方差的归一化方式错误

**位置:** §8.3

**问题:** `0.15 * (1 - posterior.std / dim)` 中 `posterior.std` 是 `ndarray`（dim 维），不是标量。`posterior.std / dim` 形状不匹配。

**修复建议:** 改为 `0.15 * (1 - mean(posterior.std) / dim)` 或 `0.15 * (1 - norm(posterior.std) / sqrt(dim))`。明确选择一种。

---

### H6. emergence_cycle 缺乏失败降级路径

**位置:** §8.2

**问题:** cycle 内部串联 5 个模块调用。任一模块失败（如模块 B 回退到相关法、模块 D 不收敛、模块 E 空记忆）时，emergence_score 还能算吗？spec 没说。

**修复建议:** 在 §8.2 增加：
> 任一模块失败时，对应项计为 0，并在返回 dict 中增加 `'warnings': list[str]` 字段记录失败原因。emergence_score 仍可计算但会偏低。

---

### H7. 混沌记忆的 alpha 取值缺失

**位置:** §7.2 第 1 步、§9

**问题:** 修改后的 Lorenz 方程引入了 `alpha` 参数，但 §9 的 `EmergenceRules` 没有 `chaotic_alpha` 字段。alpha 取值完全未定义。

**修复建议:** 在 §9 增加 `chaotic_alpha: float = 0.1`，并在 §7.2 说明：alpha 控制吸引强度，太小则记忆无效，太大则消除混沌。

---

## MEDIUM 级别问题（实现细节模糊，会导致返工）

### M1. PC 算法的独立性检验未指定

**位置:** §4.2.1

**问题:** "偏相关检验，阈值 causal_significance"——但没说用什么检验统计量。标准做法是 Fisher z 变换：`z = 0.5 * log((1+r)/(1-r))`，然后与正态分布的 p 值比较。

**修复建议:** 明确：用偏相关系数的 Fisher z 变换，p 值 < `causal_significance` 则拒绝独立性假设。

---

### M2. PC 算法的定向规则未指定

**位置:** §4.2.1

**问题:** "collider 检测（v-结构）和无环性约束定向边"——但具体规则没写。标准是 Meek (1995) 的 4 条定向规则（R1-R4）。

**修复建议:** 明确：使用 Meek 规则 R1、R2、R3（R4 需要额外的邻接信息，可省略）。

---

### M3. HMC 有限差分步长未指定

**位置:** §6.2 第 2 步

**问题:** "梯度用有限差分"——但没说步长、前向/中心差分。步长选择对 HMC 性能影响极大。

**修复建议:** 明确：中心差分，步长 `h = 1e-5`（数值最优），即 `grad[i] = (U(q + h*e_i) - U(q - h*e_i)) / (2h)`。

---

### M4. ESS 计算方法未指定

**位置:** §6.2 第 7 步

**问题:** "计算 ESS"——但没说方法。单链 ESS 估计常用自相关法（初始单调序列法）或 Gelman-Rubin（需多链）。

**修复建议:** 明确：用初始单调序列法（Geyer 1992），从 lag-1 自相关开始累加直到自相关变负。单链，注明估计不确定性。

---

### M5. HMC 接受率收敛判据过于宽松

**位置:** §6.4

**问题:** "接受率在 [0.2, 0.9] 内为 converged"——但接受率 > 0.9 在高斯后验上是正常的（HMC 对高斯几乎不拒绝）。这个判据会误判好的采样为"未收敛"。

**修复建议:** 改为 `[0.5, 0.95]`，并注明这是启发式判据，真正收敛需要 R-hat < 1.01（多链）或 ESS/samples > 0.4。

---

### M6. Lyapunov 指数检测逻辑错误

**位置:** §7.2 第 3 步、§9 `chaotic_lyapunov_threshold: float = 0.9`

**问题:** "Lyapunov 指数全程为正"——但 Lyapunov 指数是渐近量（t→∞），不是"全程"。且标准 Lorenz 系统的最大 Lyapunov 指数约 0.906，阈值 0.9 几乎总是触发，会误判所有查询为"涌现"。

**修复建议:** 改用**轨迹发散度**：`divergence = ||trajectory[-1] - trajectory_perturbed[-1]|| / ||trajectory[0] - trajectory_perturbed[0]||`，与阈值比较。或用局部 Lyapunov 指数：`log(||delta(t_end)|| / ||delta(t_0)||) / t_end`。

---

### M7. emergence_cycle 扰动量形状不明

**位置:** §8.2 步骤 3、§9 `emergence_cycle_perturbation: float = 0.1`

**问题:** `end_state = observation + delta`——observation 是 ndarray，delta 是标量 0.1？这会广播为 `observation + 0.1`（每维加 0.1）。语义模糊。

**修复建议:** 明确：`delta = rules.emergence_cycle_perturbation * rng.standard_normal(observation.shape)`（随机扰动方向）。

---

### M8. rules.seed 与 rng 双种子源冲突

**位置:** §9 `seed: int | None = None`、§2.2 `rng: np.random.Generator | None`

**问题:** rules 里有 seed，构造函数又接受 rng。两个种子源会冲突：rules.seed 优先还是 rng 优先？

**修复建议:** 从 EmergenceRules 删除 `seed` 字段。只保留 rng 参数。如需可复现，调用方传 `rng = np.random.default_rng(seed)`。

---

### M9. 线程安全锁粒度不明

**位置:** §10.2

**问题:** "self._lock 下并发 emergence_cycle 调用"——但 emergence_cycle 内部调多个模块，每个模块有状态（chaotic_memory 的存储、hmc 的采样缓存）。锁粒度是整个 cycle 还是每个模块？

**修复建议:** 明确：锁粒度是整个 emergence_cycle（粗粒度）。模块内部不持有可变状态（store 是 append-only，recall 是只读）。

---

## LOW 级别问题（文档清晰度）

### L1. topology_max_dim 与 perceive max_dim 参数优先级

**位置:** §3.3

**问题:** API 有 `max_dim: int = 2` 参数，rules 也有 `topology_max_dim`。优先级未说明。

**修复建议:** 明确：方法参数优先，rules 作为默认值。

---

### L2. constraints 参数结构未定义

**位置:** §5.3

**问题:** `constraints: dict | None = None  # 可选避障`——但没说 dict 的结构。

**修复建议:** 明确：`{'obstacles': list[np.ndarray], 'safe_radius': float}` 或直接标注"placeholder，本期不实现"。

---

### L3. reference_action 的归一化基准

**位置:** §8.3（与 H4 相关）

**问题:** 见 H4。

**修复建议:** 见 H4。

---

### L4. HMC log_prob_fn 异常处理

**位置:** §6.4

**问题:** 没说 log_prob_fn 抛异常（非 ValueError）怎么办。

**修复建议:** 明确：捕获所有异常，拒绝提议，记录到 warnings。

---

### L5. 单链 ESS 估计的局限性

**位置:** §6.3

**问题:** `ess: float` 没注明单链估计的不可靠性。

**修复建议:** 在 docstring 中注明："单链 ESS 估计有较大不确定性，建议多链运行后用 Gelman-Rubin 综合。"

---

## INFO 级别（建议）

### I1. 测试构造建议

**位置:** §10.1

**建议:** "圆的 Betti 数 = [1, 1, 0]" 的测试构造：在单位圆上采样 16 个点（n=16，符合 C1 修复后的 max_points），VR 过滤后应得 `betti_0=1, betti_1=1, betti_2=0`。在测试 docstring 中说明采样方法。

---

### I2. 性能测试缺失

**位置:** §10

**建议:** 增加一个 `test_performance.py`，对每个模块在最大输入规模（n_points=16, n_vars=8, dim=64, n_samples=100）下测时延，确保单次调用 < 1 秒。

---

## 跨节一致性问题

### X1. max_points 矛盾

**位置:** §3.2 第 4 步 vs §9

§3.2 说"n ≤ 32 点时可行"，§9 说 `topology_max_points: int = 64`。两者矛盾。

**修复:** 见 C1。

### X2. zero-data 依赖清单不完整

**位置:** §1 vs §4.2.2

§1 说"numpy + scipy + 标准库"，但 LiNGAM 的 FastICA 通常需要 scikit-learn。

**修复:** 见 C2。

### X3. emergence_cycle observation 形状未定义

**位置:** §8.2

整个 cycle 的输入形状没有定义，导致与模块 A、B 的接口不匹配。

**修复:** 见 C3。

---

## 修复后的 spec 修订清单

实现前必须完成以下 spec 修订：

1. **C1:** 降 max_points 到 16，明确分层降级策略。
2. **C2:** 简化 LiNGAM 或移除，明确依赖边界。
3. **C3:** 定义 emergence_cycle 的 observation 形状与降级路径。
4. **C4:** 完整定义混沌记忆的编码方案、alpha 取值、容量声明。
5. **H1-H7:** 修正物理模型、补充未定义变量、增加降级路径。
6. **M1-M9:** 明确检验统计量、定向规则、差分步长、ESS 方法、收敛判据、Lyapunov 检测、扰动形状、删除双种子源、明确锁粒度。

修订后建议进行第二轮快速审查，确认所有 CRITICAL/HIGH 问题已解决，然后进入实现。

---

## 不受影响的部分

以下部分审查通过，无需修改：
- §2.1 包位置与文件组织
- §2.2 模块模式（纯 Python 类、构造函数签名）
- §2.3 集成方式（不进 self.modules、不改 _N_COGNITIVE_MODULES）
- §3.1 模块 A 目的
- §6.1 模块 D 目的
- §7.1 模块 E 目的
- §8.1 引擎目的
- §10.4 确定性 fixture
- §12 实现阶段划分（4 阶段合理）
- §13 验收标准（除测试数量可能调整）
- §14 范围之外

---

## 结论

spec 在概念层面（公理、模块划分、集成方式）是健全的，但在**算法可行性**和**接口一致性**两个维度存在严重问题。4 项 CRITICAL 问题中：
- C1（持久同调复杂度）会让模块 A 在合理时间内无法运行。
- C2（LiNGAM 依赖）会让 zero-data 哲学破裂。
- C3（emergence_cycle 接口）会让引擎编排层无法实现。
- C4（混沌记忆编码）会让模块 E 无法实现。

建议：**先修订 spec，再进入实现。** 不要在 spec 未修订的状态下开始编码，否则会产生大量返工。

修订完成后，按 §11 的 15 维度对**实现代码**进行再次审查。
