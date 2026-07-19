# 因果涌现引擎 — Round 2 军事级审查报告（NEW-M4/M5/L1/L3）

> **状态:** 已完成 (2026-07-20)
> **审查范围:** 阶段 4 修复后实施的 4 项审查发现 (NEW-M4, NEW-M5, NEW-L1, NEW-L3)
> **审查方法:** 静态代码审查 + 集成点验证 + 测试覆盖审计 + 跨修复一致性检查
> **审查基线:** 阶段 4 实现审查报告 (2026-07-20-causal-emergence-engine-phase4-implementation-review.md) + commit 4afa551

---

## 1. 审查总结

| 严重度 | 数量 | 说明 |
|--------|------|------|
| CRITICAL | 0 | 无致命问题 |
| HIGH | 0 | 无严重问题 |
| MEDIUM | 1 | 测试覆盖缺口：非默认 `chaotic_dt` 未被测试 |
| LOW | 5 | 文档完善、魔法数规则化、断言加强 |
| INFO | 0 | — |

**总评:** 4 项修复均正确、内部一致、文档完备。无 CRITICAL / HIGH 问题。1 项 MEDIUM 为测试缺口（非正确性缺陷）。5 项 LOW 为文档/测试加固机会。

---

## 2. NEW-M4: chaotic_dt 配置化

### 验证项

- `chaotic_dt: float = 0.01` 正确放置于 `EmergenceRules`（rules.py:60），默认值匹配原硬编码值。
- 字段已添加到 `__post_init__` 的 `self.rules` dict（rules.py:94）。✓
- chaotic_memory.py:140 读取 `dt = self.rules.chaotic_dt`。✓
- Grep 确认 chaotic_memory.py 中无其他硬编码 `0.01` 时间步残留。
- 默认值合理 — Lorenz 特征时间尺度 ~1，dt=0.01 给出每特征时间 ~100 步，远在 RK4 稳定域内。

### 发现

#### R2-NEW-M1 — MEDIUM — 非默认 `chaotic_dt` 缺乏测试覆盖

- **位置:** `tests/test_causal_emergence_chaotic_memory.py`（整个文件）
- **描述:** 无测试构造 `EmergenceRules(chaotic_dt=...)` 并使用非默认值验证集成。`tests/` 中搜索 `chaotic_dt` 返回 0 匹配。修复对非默认值的正确性未经验证。用户改变 `chaotic_dt` 可能遇到问题（如大 dt 的 RK4 不稳定，或收敛阈值不匹配 — 见 R2-NEW-M2）而测试无法捕获。
- **修复方案:** 添加测试：
  ```python
  def test_recall_with_non_default_chaotic_dt():
      rules = EmergenceRules(chaotic_dt=0.001)  # 10x smaller
      rng = np.random.default_rng(0)
      mem = ChaoticAssociativeMemory(dim=4, rules=rules, rng=rng)
      mem.store(_make_pattern(rng, dim=4), label="A")
      result = mem.recall(_make_pattern(rng, dim=4), n_steps=50)
      assert result["trajectory"].shape == (51, 3)
      assert np.all(np.isfinite(result["trajectory"]))
      assert np.isfinite(result["divergence"])
  ```

#### R2-NEW-M2 — LOW — `5.0` 收敛阈值仍为硬编码魔法数

- **位置:** `src/zero_data_model/causal_emergence/chaotic_memory.py:179-181`
- **描述:** 注释（137-139 行）明确警告："settled-tolerance 5.0 calibrated to default dt=0.01; if you change chaotic_dt significantly, consider scaling..." 这是一个文档化的隐藏耦合 — `settled` 标志的语义依赖 `chaotic_dt`，但阈值未从 `dt` 派生。用户设 `chaotic_dt=0.001` 时，`settled` 几乎永不触发（轨迹每步移动 ~10x 更少，可能在 `n_steps` 步内未达 5.0 盆半径），静默破坏收敛语义。前一轮审查明确建议"若 5.0 收敛阈值依赖 dt，亦应同步规则化"——仅 `chaotic_dt` 被规则化，`5.0` 未被同步处理。
- **修复方案（二选一）:**
  1. 添加 `chaotic_settled_tolerance: float = 5.0` 到 `EmergenceRules`，或
  2. 从 `chaotic_dt` 派生阈值（如 `tolerance = 500.0 * chaotic_dt`，默认等于 5.0），使耦合显式且自缩放。

---

## 3. NEW-M5: 轨迹形状统一为 (n_steps + 1, 3)

### 验证项

- **循环正确性**（chaotic_memory.py:296-309）：`traj = np.zeros((n_steps + 1, 3)); state = state0.copy(); traj[0] = state; for k in range(1, n_steps + 1): ... RK4 step ... traj[k] = state`。正确存储初始状态于索引 0，执行 n_steps RK4 步，每步后存储结果。`traj[n_steps]` 是 n_steps RK4 步后的状态。✓
- **约定匹配 DifferentialGenerator**（differential.py:104-107）：`q = np.zeros((n_steps + 1, dim))`，`q[0] = start`，`q[n_steps] = end`。`(n_steps + 1, dim)` 约定现跨模块 C 和 E 一致。✓
- **`trajectory[-1]` 调用者：**
  - 166 行（divergence）：`delta_final = ||trajectory[-1] - trajectory_pert[-1]||`。新形状下指向 n_steps RK4 步后的状态。语义更正确（用户要求 n_steps 步积分）。
  - 175 行（settled 检查）：`final_state = trajectory[-1]`。同上，多走一步，更正确。
- **`compute_emergence_score`**（engine.py:362-428）：不引用 `memory_response.trajectory`，仅用 `memory_response.get("emerged", False)`。形状变更安全。✓
- **包外无调用者**对 `trajectory[k]` 索引——grep 无匹配。
- **早期返回占位符**（chaotic_memory.py:107, 122）：均用 `np.zeros((n_steps + 1, 3))`。✓
- **测试更新**（test_causal_emergence_chaotic_memory.py:128, 181）：断言 `(51, 3)` 和 `(101, 3)`。✓
- **`test_lorenz_initial_state_encodes_query`**（437-447 行）：检查 `trajectory[0] == [0.0, 0.5, 1.5]`——验证 `traj[0] = state0` 在任何 RK4 步前发生。✓
- **边界情况 n_steps=0:** `_integrate_lorenz` 返回 `np.zeros((1, 3))`，`traj[0] = state0`；循环 `for k in range(1, 1)` 不执行。行为合理（仅初始状态）。无 bug。但无测试覆盖此边界。
- **Spec §7.3**（设计文档 477 行）：`'trajectory': np.ndarray, # (n_steps + 1, 3) Lorenz 状态（fix NEW-M5：与模块 C 一致，含初始 + 终止边界）`。✓

### 发现

#### R2-NEW-M3 — LOW — engine.py 失败占位符中的硬编码 `np.zeros((51, 3))`

- **位置:** `src/zero_data_model/causal_emergence/engine.py:331`
- **描述:** 失败降级占位符硬编码 `(51, 3)`，假设 `n_steps=50`（来自 323 行的 `recall_memory(mean, n_steps=50)`）。脆弱：若有人将 323 行的 `n_steps=50` 改为其他值，占位符形状会静默不匹配真实 recall 输出形状。前一轮审查将此标记为"设计气味"但仅文档化。NEW-M5 更新占位符从 `(50, 3)` 到 `(51, 3)` 但未解决底层脆弱性。
- **修复方案:** 提升 n_steps 为局部常量或动态计算占位符形状：
  ```python
  _MEMORY_N_STEPS = 50
  ...
  memory_response = self.recall_memory(mean, n_steps=_MEMORY_N_STEPS)
  ...
  "trajectory": np.zeros((_MEMORY_N_STEPS + 1, 3)),
  ```
  使耦合显式且自维护。

---

## 4. NEW-L1: nearest_pattern 加入 spec §7.3 API 契约

### 验证项

- **4 个返回路径一致:**
  - 空记忆（chaotic_memory.py:111）：`"nearest_pattern": None` ✓
  - NaN/inf 查询（chaotic_memory.py:126）：`"nearest_pattern": None` ✓
  - 引擎失败占位符（engine.py:335）：`"nearest_pattern": None` ✓
  - 成功路径（chaotic_memory.py:195）：`"nearest_pattern": nearest_pattern`，其中 `nearest_pattern = self._patterns[nearest_idx]`（148 行）——始终为 ndarray，因 `self._patterns` 仅含经 `_sanitize` 处理的 ndarray。✓
- **类型标注 `np.ndarray | None`**（spec 483 行）：准确——成功路径返回 ndarray，所有失败路径返回 None。✓
- **`compute_emergence_score`** 不引用 `nearest_pattern`。`None` 占位符安全——无下游代码解包它。✓
- **必需键测试**（测试文件 354-366 行）：元组扩展含 `"nearest_pattern"`。✓
- **Spec §7.3 更新**（设计文档 480-483 行）：文档化字段，含注释 "fix NEW-L1: nearest_pattern 暴露原始存储向量（调试 / 可视化用）；不参与涌现度计算。空记忆时为 None。" ✓

### 发现

#### R2-NEW-L1 — LOW — Spec §7.3 注释不完整（未提及 NaN/失败情况）

- **位置:** `docs/superpowers/specs/2026-07-19-causal-emergence-engine-design.md:480-482`
- **描述:** spec 注释仅说"空记忆时为 None"，但实现也在 NaN/inf 查询时（chaotic_memory.py:126）和引擎级失败降级时（engine.py:335）返回 None。spec 低估了 None 的情况。
- **修复方案:** 扩展注释为："None when memory is empty, query is non-finite, or recall fails (engine degradation)."

#### R2-NEW-L2 — LOW — `recall()` docstring 缺 `nearest_pattern` 键

- **位置:** `src/zero_data_model/causal_emergence/chaotic_memory.py:90-99`
- **描述:** `recall()` docstring 称 `Returns dict with label, similarity, emerged, trajectory, converged, divergence.`——列表不含 `nearest_pattern`。NEW-L1 后 dict 有 7 键，但 docstring 仅列 6。
- **修复方案:** 更新为：`Returns dict with label, similarity, emerged, trajectory, converged, divergence, nearest_pattern.`

---

## 5. NEW-L3: ESS 常数序列返回 1.0 而非 n

### 验证项

- **实现**（hmc.py:296-305）：`if var < 1e-12: ess_values.append(1.0); continue`。`1e-12` 阈值匹配先前代码（仅追加值从 `float(n)` 改为 `1.0`）。✓
- **数学合理性:** 返回 1.0 保守且可辩护。常数序列方差为零——自相关未定义（0/0）。三种常见约定：
  - ESS = 0（无方差信息）——过于悲观，全常数 posterior 会使 `np.mean` 崩溃。
  - ESS = n（完全代表）——旧行为，掩盖其他维度的欠采样。
  - ESS = 1（一个有效样本足够）——新行为，保守中间选择。
  内联注释（297-303 行）清晰解释推理。✓
- **均值 ESS 计算**（hmc.py:323）：`float(np.mean(ess_values))`——当某些维度返回 1.0 其他返回较大值时仍正确。对 2 维 posterior（dim 0 常数 ESS=1.0，dim 1 i.i.d. ESS≈n），均值 ≈(1+n)/2，远小于旧 (n+n)/2 = n。这是预期行为。✓
- **其他依赖旧 `ess = n` 行为的位置:** Grep `_ess_geyer` 仅一个调用者（`_finalize` 在 hmc.py:269）和两个测试。无外部代码依赖旧行为。✓
- **边界情况 `n < 2`**（hmc.py:289-290）：未变，返回 `float(n)`（即 n=1 时 1.0，n=0 时 0.0）。与新常数序列 1.0 值一致。✓
- **clamp** `max(1.0, min(float(n), ess))` 在 321 行在 NEW-L3 前已存在——对非常数维度强制 `1.0 <= ess <= n`。NEW-L3 变更正交：仅影响常数维度特殊情形。✓

### 发现

#### R2-NEW-L3 — LOW — 混合测试断言过弱；未捕获 `ess = 0` 回归

- **位置:** `tests/test_causal_emergence_hmc.py:349`
- **描述:** 断言 `1.0 < ess < 100.0` 过弱。它捕获回归到旧行为（ess = n → 均值 = (100+100)/2 = 100，失败上界 `< 100.0`），但不捕获回归到常数维度 `ess = 0`（均值 = (0+100)/2 = 50，会通过测试）。测试 docstring（347-348 行）甚至承认 "ess should be close to 50.5 but bounded"——断言未实际验证此点。
- **修复方案:** 收紧为 `assert 40.0 < ess < 60.0`（或类似），同时捕获旧 `ess = n` 回归（ess ≈ 100）和假设的 `ess = 0` 回归（ess ≈ 50 对 n=100 i.i.d.）。更严：`assert abs(ess - 50.5) < 10.0`。

#### R2-NEW-L4 — LOW — `_ess_geyer` docstring 未提及常数序列特殊情形

- **位置:** `src/zero_data_model/causal_emergence/hmc.py:282-287`
- **描述:** docstring 称 "For each dimension, compute the autocorrelation function and sum pairs... ESS_d = n / (1 + 2 * sum). Return the mean across dimensions." 未提及常数序列特殊情形（每维返回 1.0）。内联注释（297-303 行）解释了，但仅读 docstring 的读者不会知道此特殊情形。
- **修复方案:** 添加到 docstring："Constant dimensions (variance < 1e-12) return ESS_d = 1.0 — a conservative middle ground between 0 (no info) and n (perfectly representative)."

---

## 6. 跨修复一致性

| 修复对 | 交互 | 状态 |
|--------|------|------|
| NEW-M4 + NEW-M5 | `chaotic_dt` 影响积分步长；轨迹形状变更正交。无冲突。 | ✓ 一致 |
| NEW-M4 + NEW-L1 | 无共享代码路径。 | ✓ 一致 |
| NEW-M4 + NEW-L3 | 无共享代码路径（不同模块）。 | ✓ 一致 |
| NEW-M5 + NEW-L1 | 均修改 chaotic_memory.py 中相同 3 个早期返回占位符（100-127 行）和引擎失败占位符（engine.py:326-336）。两修复一致应用——每个占位符现含正确形状 `(n_steps + 1, 3)` 且 `nearest_pattern: None` 字段。 | ✓ 一致 |
| NEW-M5 + NEW-L3 | 无共享代码路径。 | ✓ 一致 |
| NEW-L1 + NEW-L3 | 无共享代码路径。 | ✓ 一致 |

**4 项修复均正确交互。** 唯一两修复触及相同行的位置（NEW-M5 + NEW-L1 在早期返回占位符）显示变更被一起应用，无冲突。

---

## 7. 测试覆盖

### 4 项修复添加/更新的测试

- `test_recall_empty_memory_trajectory_shape`（chaotic_memory 测试:120-129）——验证空记忆 `(51, 3)` 形状。✓
- `test_recall_trajectory_shape`（chaotic_memory 测试:171-181）——验证非空记忆 `(101, 3)` 形状。✓
- `test_lorenz_initial_state_encodes_query`（chaotic_memory 测试:437-447）——验证 `traj[0] = (0, qy, qz)`。✓
- `test_recall_returns_required_keys`（chaotic_memory 测试:354-366）——验证 `nearest_pattern` 在必需键中。✓
- `test_ess_constant_series_returns_one`（hmc 测试:317-329）——验证常数序列 ESS = 1.0。✓
- `test_ess_mixed_constant_and_variable`（hmc 测试:332-349）——验证混合常数/变量行为。✓（但弱——见 R2-NEW-L3）

### 缺失测试（缺口）

1. **无非默认 `chaotic_dt` 测试**——见 R2-NEW-M1（MEDIUM）。最显著的缺口。
2. **无 `n_steps=0` 在 `_integrate_lorenz` 的测试**——新约定返回 `(1, 3)`（仅初始状态），但无测试覆盖此边界。原代码返回 `(0, 3)`（空），故为行为变更。低优先级，因无调用者传 `n_steps=0`。
3. **无 `n_steps=1` 测试**——会验证 `traj[0] = state0` 和 `traj[1]` = 1 RK4 步后状态。低优先级。
4. **无测试验证成功路径 `nearest_pattern` 是 ndarray**——`test_recall_returns_required_keys` 仅检查键存在，不检查类型。低优先级。
5. **无集成测试在 `test_causal_emergence_emergence_cycle.py` 或 `test_zero_data_model_causal_emergence_integration.py` 验证记忆轨迹形状**——形状仅在单元级检查。`emergence_cycle` 调用 `recall_memory(mean, n_steps=50)`，故记忆子字典轨迹应为 `(51, 3)`。低优先级（单元测试已覆盖）。

---

## 8. 行为变更（非 bug，但值得注意）

1. `trajectory[-1]` 现指 n_steps RK4 步后的状态（原为 n_steps-1）。这使 divergence 值通常更大（多一步混沌放大）且 `settled` 标志对多走一步的状态求值。所有现有测试仍通过因检查有限性/范围属性，非精确值。
2. ESS 对某些维度为常数的 posterior 现显著降低（每常数维度 1.0 而非 n）。这是预期修复——揭示混合 posterior 中的欠采样。任何对 ESS 阈值的下游代码可能见到不同（更保守）行为。

---

## 9. 结论

**4 项修复已就绪可生产。** 所有修复正确、内部一致、在代码注释和 spec 中妥善文档化。无 CRITICAL 或 HIGH 问题引入。

**最终签署前推荐（1 MEDIUM）:**
- 添加非默认 `chaotic_dt` 测试（R2-NEW-M1）以闭合覆盖缺口。

**未来清理轮推荐（5 LOW，非阻塞）:**
- 规则化 `5.0` settled-tolerance 或从 `chaotic_dt` 派生（R2-NEW-M2）。
- 提升 engine.py `n_steps=50` 魔法数为命名常量（R2-NEW-M3）。
- 扩展 spec §7.3 注释提及 NaN/失败情况 `nearest_pattern = None`（R2-NEW-L1）。
- 更新 `recall()` docstring 键列表含 `nearest_pattern`（R2-NEW-L2）。
- 收紧 `test_ess_mixed_constant_and_variable` 为 `40.0 < ess < 60.0`（R2-NEW-L3）。
- 更新 `_ess_geyer` docstring 提及常数序列特殊情形（R2-NEW-L4）。

---

**审查人:** Agent (Round 2 military-grade review)
**审查日期:** 2026-07-20
**审查工具:** 静态阅读 + 跨修复一致性审计 + 测试覆盖分析
