# 因果涌现引擎 spec v2 修订验证报告

> **验证对象:** [docs/superpowers/specs/2026-07-19-causal-emergence-engine-design.md](file:///workspace/docs/superpowers/specs/2026-07-19-causal-emergence-engine-design.md) (v2)
> **验证基准:** [docs/superpowers/reviews/2026-07-19-causal-emergence-engine-spec-review.md](file:///workspace/docs/superpowers/reviews/2026-07-19-causal-emergence-engine-spec-review.md)
> **验证日期:** 2026-07-19
> **验证者:** Agent

---

## 验证方法

逐条对照审查报告的 27 项问题，检查 v2 spec 是否完整修复。每项标记：
- ✅ 已修复
- ⚠️ 部分修复（需说明）
- ❌ 未修复

---

## CRITICAL 级别（4 项）

### C1. 持久同调算法复杂度不可行 — ✅ 已修复

**v2 §3.2:** 明确"分层降级策略"，max_points 从 64 降为 **16**，n=16 单形数 ≈ 696，O(s³) ≈ 3.4e8 秒级可行。明确删除"n ≤ 32 可行"的错误陈述（§3.2 末段）。`topology_max_points: int = 16` 在 §9 已落实。

**验证:** spec §3.2 步骤 2 子采样规则明确；§9 rules 字段已更新；§3.2 末段显式声明删除错误陈述。

---

### C2. LiNGAM 依赖冲突 — ✅ 已修复

**v2 §1 依赖契约:** 明确禁止 scikit-learn（含 FastICA、LinearRegression）、gudhi、ripser、pymc、numpyro、causal-learn、numba。
**v2 §4.2.2 简化版 LiNGAM:** 用 `scipy.linalg.svd` 白化 + 自实现固定点 ICA（Hyvärinen 1999），不引入 scikit-learn。限制 n_vars ≤ 8。`causal_max_vars_lingam: int = 8` 在 §9 已落实。

**验证:** §1 依赖边界清晰；§4.2.2 算法步骤完整；§9 rules 字段已新增。

---

### C3. emergence_cycle 接口不匹配 — ✅ 已修复

**v2 §8.2:** 明确 observation 形状要求：
- 2D `(n_samples, n_features)`，要求 `n_samples >= 2 且 n_features >= 2`。
- 1D 输入直接返回零分：`{'emergence_score': 0.0, 'reason': 'insufficient_data'}`。

**验证:** §8.2 docstring 显式声明形状要求与降级路径。

---

### C4. 混沌记忆编码方案缺失 — ✅ 已修复

**v2 §7.2 第 1 步:** 完整定义编码 `target_y = mean(pattern[:dim//2]), target_z = mean(pattern[dim//2:])`。
**v2 §7.2 第 5 步:** 删除"理论无上限"陈述，改为"实际容量受 rules.chaotic_memory_capacity 限制，超出 FIFO 驱逐"。
**v2 §7.2 第 2 步:** alpha=0.1 默认值明确，并增加稳定性说明（"alpha 太大会消除混沌，太小则记忆无效"）。
**v2 §9:** `chaotic_alpha: float = 0.1` 已加入 rules。

**验证:** 编码方案完整；容量声明正确；alpha 取值已落实；稳定性说明已加。

---

## HIGH 级别（7 项）

### H1. 微分方程物理模型与目的不符 — ✅ 已修复

**v2 §5.2:** 改用阻尼最小作用量，离散方程为"二阶时间导数 + 一阶时间导数（阻尼）+ 弹性恢复力 = 0"。明确"阻尼项让轨迹单调收敛到 end_state，避免谐振子的振荡。当 lambda=0, gamma=0 时退化为最短路径（线性插值）"。

**验证:** §5.2 算法修订完整，物理意义明确。

---

### H2. 微分方程的 target 未定义 — ✅ 已修复

**v2 §5.2 第 2 步:** 明确 `target = end_state`。

**验证:** 变量名已替换。

---

### H3. 反事实算法不完整 — ✅ 已修复

**v2 §4.3:** 改为线性高斯假设下的闭式 do-calculus。完整给出：
1. 估计线性权重（每个节点回归）
2. intervene(var, value) 三步：图割裂、数据替换、按拓扑序传播
3. counterfactual(var, value, observed) 闭式解：`observed - W[:, var] * (observed[var] - value)`

**验证:** 算法完整、可实施，避免信念传播的复杂性。

---

### H4. 涌现度评分的归一化基准未定义 — ✅ 已修复

**v2 §8.3:** 明确 `reference_action = ||observation.mean(axis=0)||² * n_steps / 2`（直线轨迹的作用量）。

**验证:** 基准已定义。

---

### H5. 后验方差的归一化方式错误 — ✅ 已修复

**v2 §8.3:** 改为 `mean(posterior.std) / dim`，明确"用 `mean(posterior.std)` 而非 `posterior.std / dim`"。

**验证:** 形状匹配。

---

### H6. emergence_cycle 缺乏失败降级路径 — ✅ 已修复

**v2 §8.2 失败降级:** 明确"任一模块失败时，对应项计为 0，warnings 列表记录失败原因"。返回 dict 增加 `'warnings': list[str]` 字段。

**验证:** 降级路径与 warnings 字段已加。

---

### H7. 混沌记忆的 alpha 取值缺失 — ✅ 已修复

**v2 §9:** `chaotic_alpha: float = 0.1` 已加入 EmergenceRules。
**v2 §7.2 第 2 步:** alpha 在 Lorenz 方程中显式出现，并附稳定性说明。

**验证:** rules 字段与算法均落实。

---

## MEDIUM 级别（9 项）

### M1. PC 算法的独立性检验未指定 — ✅ 已修复

**v2 §4.2.1:** 明确"用偏相关系数的 Fisher z 变换。给定偏相关系数 r，`z = 0.5 * log((1+r)/(1-r))`，统计量 `sqrt(n - |S| - 3) * |z|` 服从标准正态分布。p 值 < causal_significance 则拒绝独立性"。

**验证:** 检验统计量与判据完整。

---

### M2. PC 算法的定向规则未指定 — ✅ 已修复

**v2 §4.2.1:** 明确"应用 Meek (1995) 规则 R1、R2、R3（R4 需要额外的邻接信息，本实现省略）"，并列出 R1/R2/R3 的具体定向条件。

**验证:** 三条规则完整列出。

---

### M3. HMC 有限差分步长未指定 — ✅ 已修复

**v2 §6.2 第 3 步:** 明确"中心差分，步长 `h = 1e-5`"，给出公式 `grad_U[i] = (U(q + h*e_i) - U(q - h*e_i)) / (2*h)`。
**v2 §9:** `hmc_finite_diff_h: float = 1e-5` 已加入 rules。

**验证:** 步长与差分方式明确。

---

### M4. ESS 计算方法未指定 — ✅ 已修复

**v2 §6.2 第 8 步:** 明确"用初始单调序列法（Geyer 1992）：从 lag-1 自相关开始，按 lag 累加直到自相关之和首次变负，ESS = `n_samples / (1 + 2 * sum)`"。

**验证:** 方法明确，公式给出。

---

### M5. HMC 接受率收敛判据过于宽松 — ✅ 已修复

**v2 §6.4:** 改为 `0.5 <= accept_rate <= 0.95`，并附修订说明"原 [0.2, 0.9] 过严，对高斯后验 HMC 几乎不拒绝"。

**验证:** 判据修订。

---

### M6. Lyapunov 指数检测逻辑错误 — ✅ 已修复

**v2 §7.2 第 4 步:** 改用"轨迹发散度"而非 Lyapunov 指数。给出完整公式：
```
perturbed = query + rules.chaotic_perturbation * rng.standard_normal(query.shape)
trajectory_perturbed = integrate(perturbed, n_steps)
divergence = ||trajectory[-1] - trajectory_perturbed[-1]|| / (||query - perturbed|| + 1e-12)
emerged = (divergence > rules.chaotic_divergence_threshold)
```
**v2 §9:** `chaotic_perturbation: float = 0.01` 与 `chaotic_divergence_threshold: float = 10.0` 已加入 rules。

**验证:** 检测方法替换为发散度，阈值合理（10.0 远大于 1）。

---

### M7. emergence_cycle 扰动量形状不明 — ✅ 已修复

**v2 §8.2 步骤 3:** 明确 `delta = rules.emergence_cycle_perturbation * rng.standard_normal(n_features)`，"修复 M7：明确 delta 形状为随机扰动方向"。

**验证:** 形状明确。

---

### M8. rules.seed 与 rng 双种子源冲突 — ✅ 已修复

**v2 §2.2:** 明确"EmergenceRules **不**含 `seed` 字段，避免双种子源冲突"。
**v2 §9:** rules 定义中显式注释"修复 M8：删除 seed 字段，仅由 rng 参数控制确定性"。

**验证:** seed 字段已删除。

---

### M9. 线程安全锁粒度不明 — ✅ 已修复

**v2 §2.3:** 明确"facade 方法（均持锁 with self._lock: ...，锁粒度为整个方法体，修复 M9）"。
**v2 §10.2:** 明确"self._lock 锁粒度为整个 facade 方法体。模块内部不持有可变状态（store 是 append-only deque，recall 是只读）。并发 emergence_cycle 调用串行化执行"。

**验证:** 锁粒度与状态可变性明确。

---

## LOW 级别（5 项）

### L1. topology_max_dim 与 perceive max_dim 参数优先级 — ✅ 已修复

**v2 §3.3:** API 改为 `max_dim: int | None = None`，docstring 明确"参数 max_dim 优先于 rules.topology_max_dim；None 时使用 rules 默认值（修复 L1）"。

**验证:** 优先级明确。

---

### L2. constraints 参数结构未定义 — ✅ 已修复

**v2 §5.3:** 注释改为"placeholder，本期不实现（修复 L2）"。
**v2 §14:** 范围之外明确列出"constraints 避障（DifferentialGenerator 的 constraints 参数为 placeholder，本期不实现）"。

**验证:** 明确为 placeholder，本期不实现。

---

### L3. reference_action 的归一化基准 — ✅ 已修复

**与 H4 同步修复。** v2 §8.3 明确 `reference_action = ||observation.mean(axis=0)||² * n_steps / 2`。

**验证:** 基准已定义。

---

### L4. HMC log_prob_fn 异常处理 — ✅ 已修复

**v2 §6.2 第 7 步:** 明确"log_prob_fn 抛任何异常时，捕获并拒绝该提议，记录到 warnings 列表。返回 -inf 或 NaN 时同样拒绝"。
**v2 §6.5 边界情况:** 重申"log_prob_fn 抛异常：捕获并拒绝，记录到 warnings"。

**验证:** 异常处理完整。

---

### L5. 单链 ESS 估计的局限性 — ✅ 已修复

**v2 §6.2 第 8 步:** 明确"单链 ESS 估计有较大不确定性（修复 L5），建议多链运行后用 Gelman-Rubin 综合"。
**v2 §6.3 docstring:** 重申"单链 ESS 估计有较大不确定性，建议多链运行后用 Gelman-Rubin 综合"。

**验证:** 局限性已说明。

---

## INFO 级别（2 项）

### I1. 测试构造建议 — ✅ 已修复

**v2 §10.1:** 明确"在单位圆上采样 16 个点（n=16，符合 max_points），VR 过滤后应得 betti_0=1, betti_1=1, betti_2=0。采样方法：`angles = linspace(0, 2*pi, 16, endpoint=False); points = column_stack([cos(angles), sin(angles)])`"。

**验证:** 采样方法已给出。

---

### I2. 性能测试缺失 — ✅ 已修复

**v2 §10.4:** 新增"性能测试（修复 I2）"小节，定义 `tests/test_causal_emergence_performance.py`，对每个模块在最大输入规模下测时延，给出具体阈值。

**验证:** 性能测试已加入验收标准。

---

## 跨节一致性问题（3 项）

### X1. max_points 矛盾 — ✅ 已修复

**v2 §3.2 与 §9 一致:** 都是 `topology_max_points: int = 16`。删除了"n ≤ 32 可行"的错误陈述。

---

### X2. zero-data 依赖清单不完整 — ✅ 已修复

**v2 §1 依赖契约:** 明确允许与禁止的依赖清单。LiNGAM 的 ICA 改为自实现（§4.2.2）。

---

### X3. emergence_cycle observation 形状未定义 — ✅ 已修复

**v2 §8.2:** 明确 observation 形状要求。

---

## 验证总结

| 类别 | 总数 | ✅ 已修复 | ⚠️ 部分修复 | ❌ 未修复 |
|------|------|----------|------------|----------|
| CRITICAL | 4 | 4 | 0 | 0 |
| HIGH | 7 | 7 | 0 | 0 |
| MEDIUM | 9 | 9 | 0 | 0 |
| LOW | 5 | 5 | 0 | 0 |
| INFO | 2 | 2 | 0 | 0 |
| 跨节一致性 | 3 | 3 | 0 | 0 |
| **合计** | **30** | **30** | **0** | **0** |

**结论:** 第一轮审查的 27 项问题 + 3 项跨节一致性问题 = 30 项，**全部已修复**。spec v2 可进入第二轮审查。

---

## 第二轮审查：检查 v2 是否引入新问题

对 v2 进行快速二次审查，检查修订是否引入新隐患：

### 新增项检查

1. **§4.2.2 固定点 ICA 迭代公式:** `W <- (Z * g(W^T Z)).mean(axis=1) - g'(W^T Z).mean(axis=1)` — 标准的 Hyvärinen FastICA 公式，正确。
2. **§4.2.2 对称正交化:** `W <- (W @ W^T)^{-1/2} @ W` — 标准做法，正确。
3. **§5.2 离散方程:** `(q[k+1] - 2*q[k] + q[k-1]) / dt² + gamma * (q[k+1] - q[k-1]) / (2*dt) + lambda * (q[k] - target) = 0` — 这是阻尼谐振子的标准中心差分离散，正确。
4. **§7.2 Lorenz 修改:** 引入 `alpha * sum_i w_i * (y - target_y_i)` 项，w_i 为高斯核权重。这在物理上合理——多个吸引子盆的加权叠加，符合"最近吸引子主导"的直觉。
5. **§8.3 涌现度评分权重:** 0.3 + 0.2 + 0.2 + 0.15 + 0.15 = 1.0 ✓
6. **§9 rules 字段:** 所有新增字段（differential_lambda, differential_gamma, hmc_finite_diff_h, chaotic_alpha, chaotic_sigma_q, chaotic_perturbation, chaotic_divergence_threshold, causal_max_vars_lingam）类型注解正确，默认值合理。

### 新发现的潜在问题（轻微）

**N1（INFO）:** §7.2 第 2 步的修改 Lorenz 方程中，`w_i` 依赖 `query`（"查询对第 i 个模式的亲和权重"），但 `w_i` 在积分过程中应该是固定的（基于初始 query 计算）还是随当前状态更新？spec 没明确。

**建议:** 在实现时明确为"基于初始 query 计算一次 w_i，积分过程中保持不变"（否则系统变成非自治，分析困难）。这条不算 spec 缺陷，可在实现时处理。

**N2（INFO）:** §4.2.1 PC 算法的"检验顺序按 (i, j, |S|) 字典序"——但 PC 标准做法是先按 |S|=0 全部检验，再 |S|=1，再 |S|=2...，每轮删除边后再进入下一轮（因为邻接集会变）。spec 的"字典序"描述不够明确。

**建议:** 实现时采用标准的"层序"：先 |S|=0 全检验，再 |S|=1（基于更新后的邻接集），以此类推。spec 的"字典序"可解读为"同一 |S| 内按 (i, j) 字典序"，这是正确的。这条也不算 spec 缺陷。

### 第二轮审查结论

v2 没有引入新的 CRITICAL / HIGH / MEDIUM 问题。仅有 2 项 INFO 级别的实现细节可在编码时处理。**spec v2 可进入实现阶段。**

---

## 最终结论

| 阶段 | 结果 |
|------|------|
| 第一轮审查 | 发现 27 项问题（4 CRITICAL + 7 HIGH + 9 MEDIUM + 5 LOW + 2 INFO） |
| spec 修订 | 全部 30 项问题（含 3 项跨节一致性）已修复 |
| 修订验证 | 30/30 ✅，0 ⚠️，0 ❌ |
| 第二轮审查 | 未引入新 CRITICAL/HIGH/MEDIUM 问题，仅 2 项 INFO 实现细节 |
| **可进入实现** | **是** |

spec v2 已通过超级军事级审查的全部修复与验证，可进入实现阶段。
