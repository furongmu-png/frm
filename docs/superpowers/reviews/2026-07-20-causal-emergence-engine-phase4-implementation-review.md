# 因果涌现引擎 — 阶段 4 实现军事级审查报告

> **状态:** 已完成 (2026-07-20)
> **审查范围:** 阶段 1-4 全部实现代码（`src/zero_data_model/causal_emergence/` 8 个文件 + 7 个测试文件，共 200 个测试）
> **审查方法:** 静态代码审查 + 集成点验证 + 测试覆盖审计
> **审查基线:** spec v2 + 第一轮审查修复 (C1-C4, H1-H7, M3, M6-M8)

---

## 1. 审查总结

| 严重度 | 数量 | 说明 |
|--------|------|------|
| CRITICAL | 0 | 无致命问题 |
| HIGH | 2 | Meek R1 死代码；反事实单位权传播偏离 spec |
| MEDIUM | 6 | 死代码、占位符不一致、配置缺失、线程安全 |
| LOW | 3 | API 表面扩展、注释误导、ESS 定义争议 |
| INFO | 7 | 4 项正向确认 + 3 项观察 |

**总评:** 阶段 4 的 `emergence_cycle` 编排与 `compute_emergence_score` 启发式实现正确，spec v2 修复 (C1-C4, H1-H7, M3, M6-M8) 均已落地。无 CRITICAL 问题。2 项 HIGH 涉及因果发现模块（阶段 1）的正确性，应修复。其余多为死代码、注释误导、占位符一致性问题。

---

## 2. CRITICAL

无。

---

## 3. HIGH

### NEW-H1 — Meek R1 定向规则为死代码，永不定向任何边

- **位置:** [src/zero_data_model/causal_emergence/causal_discovery.py:324-353](file:///workspace/src/zero_data_model/causal_emergence/causal_discovery.py#L324-L353)
- **描述:** R1 规则块包含三重嵌套循环，但每个分支都以 `pass` 结尾（第 340、343、353 行）。第 341 行的条件 `directed[a, b] == 1 and adj[a, c] == 1` 检查 `a` 与 `c` **相邻**，与 R1 要求相反（R1 需要 `a` 与 `c` **不相邻**）。第 333-340 行存在自相矛盾条件（`adj[a, c] == 1` 与 `adj[a, c] == 0` 同时出现）。第 346-353 行重复同样的错误。结果：R1 永不定向任何边，CPDAG→DAG 转换仅依赖 R2、R3 和第 386-391 行的索引顺序兜底，许多 v-结构 (`a -> c <- b` with `a` not adj `b`) 将被错误定向或保持未定向，导致因果 DAG 不正确。
- **修复方案:** 用正确的 R1 实现替换死代码块：
  ```python
  # R1: a -> b, a - c, c - b, and a NOT adjacent to c  =>  c -> b
  for a in range(n_vars):
      for b in range(n_vars):
          if a == b or directed[a, b] != 1:
              continue
          for c in range(n_vars):
              if c in (a, b) or directed[c, b] == 1:
                  continue
              if adj[c, b] == 1 and adj[a, c] == 0 and adj[c, a] == 0:
                  directed[c, b] = 1
                  adj[c, b] = 0
                  adj[b, c] = 0
                  changed = True
  ```
  并添加针对 R1 必要场景的单元测试。

### NEW-H2 — 反事实使用单位权传播，偏离 spec §4.3 加权公式

- **位置:** [src/zero_data_model/causal_emergence/causal_discovery.py:181-209](file:///workspace/src/zero_data_model/causal_emergence/causal_discovery.py#L181-L209)
- **描述:** Spec §4.3 指定线性高斯反事实为 `cf[node] = observed[node] - W[:, var] * (observed[var] - value)`（对后代节点），其中 `W` 为估计的权重矩阵。实现采用 `cf[node] = observed[node] + delta`（第 200 行），其中 `delta` 为标量，对所有后代统一偏移。这意味着：(a) 每个后代接收相同偏移，与边强度无关；(b) 权重矩阵 `W`（`_estimate_weights` 估算）未被使用；(c) `counterfactual` 方法签名 `(adjacency, observed, intervention_var, intervention_value)` 无 `data` 参数，无法估算权重。返回的 `shift` 字段对任何非单位权 DAG 都是错误的。
- **修复方案:** 因 `counterfactual` 无法从单观测估算权重，采用以下二选一：
  - (a) 在方法签名添加 `weights: np.ndarray | None = None` 参数，由调用方提供（如先调用 `discover` 然后 `_estimate_weights`）；
  - (b) 更新 spec §4.3 承认此简化，更新 docstring 明确标注 "unit-weight simplified counterfactual — pass weights externally for full spec compliance"。
  推荐 (b) 作为最小变更，避免破坏 API。

---

## 4. MEDIUM

### NEW-M1 — 持久同调 pair 循环中的死表达式

- **位置:** [src/zero_data_model/causal_emergence/topology.py:268-270](file:///workspace/src/zero_data_model/causal_emergence/topology.py#L268-L270)
- **描述:** 第 270 行 `len(simplices[j][0]) - 1` 是一个无赋值目标的裸表达式，显然是 `d_death = len(simplices[j][0]) - 1` 的不完整重构残留。
- **修复:** 删除该行（`d_death` 当前未被使用），或正确赋值并断言 `d_death == d_birth + 1`。

### NEW-M2 — 后验失败占位符 `std=1.0` 与"large std"注释自相矛盾

- **位置:** [src/zero_data_model/causal_emergence/engine.py:301-309](file:///workspace/src/zero_data_model/causal_emergence/engine.py#L301-L309)（第 304 行）
- **描述:** 当 HMC 采样器抛异常时，降级占位符设置 `"std": np.ones(n_features)` 并注释 `# large std -> low score term`。但 `1.0` 相对于 `dim=n_features` 并不大。在 `compute_emergence_score` 中 `term4 = 1.0 - mean_std / dim = 1.0 - 1.0 / n_features`，对 `n_features ∈ [2, 64]` 得到 `term4 ∈ [0.5, 0.984]` —— 这是高分而非低分，违反了注释意图，会部分掩盖后验失败。
- **修复:** 将占位符改为 `"std": np.full(n_features, 1e6)` 使 term4 接近 0，并添加针对后验失败的回归测试。

### NEW-M3 — chaotic_memory.store() 中的死代码静默忽略维度不匹配

- **位置:** [src/zero_data_model/causal_emergence/chaotic_memory.py:66-69](file:///workspace/src/zero_data_model/causal_emergence/chaotic_memory.py#L66-L69)
- **描述:** `store` 方法包含一个 `if p.shape[0] != self.dim and self.dim != 64:` 块，但块体只有 `pass`。条件被求值但无任何动作，维度不匹配被静默忽略。注释描述了"首次存储时覆盖 dim"的意图，但代码未实现。
- **修复:** 删除整个 `if` 块（dim 不匹配由下游 `_encode`/`_integrate_lorenz` 自然处理），或实现文档所述行为（首次存储时更新 `self.dim`）。

### NEW-M4 — chaotic_memory Lorenz 积分中 `dt=0.01` 硬编码，不可配置

- **位置:** [src/zero_data_model/causal_emergence/chaotic_memory.py:129](file:///workspace/src/zero_data_model/causal_emergence/chaotic_memory.py#L129)
- **描述:** `recall` 方法硬编码 `dt = 0.01`，与 `differential.py` 读取 `self.rules.differential_dt` 不一致。`EmergenceRules` 已暴露 `chaotic_lorenz_sigma/rho/beta` 和 `chaotic_alpha` 但缺 `chaotic_dt` 字段。
- **修复:** 添加 `chaotic_dt: float = 0.01` 到 `EmergenceRules`，在 `recall` 中读取；若 `5.0` 收敛阈值依赖 `dt`，亦应同步规则化。**（本次仅文档化，不改实现以避免破坏现有测试基线）**

### NEW-M5 — 模块 C 与模块 E 轨迹形状约定不一致

- **位置:** [src/zero_data_model/causal_emergence/differential.py:104](file:///workspace/src/zero_data_model/causal_emergence/differential.py#L104) vs [src/zero_data_model/causal_emergence/chaotic_memory.py:280](file:///workspace/src/zero_data_model/causal_emergence/chaotic_memory.py#L280)
- **描述:** 模块 C 返回 `(n_steps + 1, dim)`（含初始和终止边界），模块 E 返回 `(n_steps, 3)`（不含独立初始行）。引擎占位符已正确镜像各自约定（counterfactual 用 `(17, 1)`，memory 用 `(50, 3)`），但跨模块形状不一致是设计气味。两套测试均已锁定各自约定，属有意但未文档化的偏离。
- **修复:** **（本次仅文档化）** 在 spec §5/§7 明确两种约定的语义差异；未来重构时统一到 `(n_steps + 1, dim)`。

### NEW-M6 — 非线程安全：共享可变 RNG + 未同步的内存列表

- **位置:** [src/zero_data_model/causal_emergence/engine.py:50-51](file:///workspace/src/zero_data_model/causal_emergence/engine.py#L50-L51) 与 [src/zero_data_model/causal_emergence/chaotic_memory.py:71-80](file:///workspace/src/zero_data_model/causal_emergence/chaotic_memory.py#L71-L80)
- **描述:** 引擎共享单一 `np.random.Generator` 跨 5 个模块（`Generator` 内部状态可变，并发 `emergence_cycle` 会交叉抽取破坏可复现性）；`ChaoticAssociativeMemory.store` 无锁修改 `self._patterns/_labels/_targets`，并发 `store` + `recall` 可能产生撕裂读（`_targets` 已 N+1 项但 `_patterns` 仍 N 项，引发 IndexError）。
- **修复:** spec 未要求线程安全；本次仅在类 docstring 中明确标注 "NOT thread-safe — single-threaded use only"。

---

## 5. LOW

### NEW-L1 — `nearest_pattern` 字段不在 spec §7.3 API 契约中

- **位置:** [src/zero_data_model/causal_emergence/chaotic_memory.py:184](file:///workspace/src/zero_data_model/causal_emergence/chaotic_memory.py#L184)
- **描述:** `recall` 返回 dict 含 `nearest_pattern`（原始存储向量），spec §7.3 仅列出 `label, similarity, emerged, trajectory, converged, divergence` 6 项。该字段未在 API 契约测试中断言，可移除而测试不破。
- **修复:** 保留字段（对调试有用），但在 spec §7.3 补充声明，并加入 API 契约测试的 required-keys 元组。

### NEW-L2 — `_break_cycles` 按二元权排序（恒为 1.0），非按幅值

- **位置:** [src/zero_data_model/causal_emergence/causal_discovery.py:625-640](file:///workspace/src/zero_data_model/causal_emergence/causal_discovery.py#L625-L640)
- **描述:** 注释称"按权值升序排序，移除直到无环"，但 `adj` 是二元矩阵（条目恒为 0 或 1），`float(adj[i, j])` 恒为 1.0，排序实际退化为按 `(i, j)` 索引顺序。
- **修复:** 更新注释为"按索引顺序确定性兜底（二元邻接矩阵）"，移除"magnitude"误导措辞。

### NEW-L3 — 常数序列 ESS = n 的定义有争议

- **位置:** [src/zero_data_model/causal_emergence/hmc.py:296-298](file:///workspace/src/zero_data_model/causal_emergence/hmc.py#L296-L298)
- **描述:** 当样本某维 `var < 1e-12` 时，代码令 `ess = n`，注释"完全信息性，ESS = n"。可论证常数序列携带零后验信息，ESS 应为 0 或 1。当前实现会上偏均值 ESS，可能掩盖欠采样。
- **修复:** **（保留当前实现）** 常数序列在贝叶斯后验中通常意味着先验-似然主导，ESS = n 作为"完全采样"是合理工程选择；仅添加注释说明争议。

---

## 6. INFO

### NEW-I1 — `reference_action` 公式与 `action` 量纲不一致（spec 公式问题）

- **位置:** [src/zero_data_model/causal_emergence/engine.py:327-329](file:///workspace/src/zero_data_model/causal_emergence/engine.py#L327-L329)
- **描述:** 实现 `reference_action = ||mean||² * 16 / 2.0` 严格匹配 spec §8.3，但实际 `action` 单位为 `1/dt`（含 `differential_dt` 因子），导致 term5 大多饱和到 1.0。这是 spec 公式问题，实现忠实于 spec。
- **修复:** 无代码改动；若 spec 修订，建议改为 `reference_action = ||mean||² / (2 * n_steps * dt)`。

### NEW-I2 — `persistence_threshold = 0.5 * max_filtration` 为启发式，非标准持久 Betti 定义

- **位置:** [src/zero_data_model/causal_emergence/topology.py:252-253](file:///workspace/src/zero_data_model/causal_emergence/topology.py#L252-L253)
- **描述:** 0.5×max_filtration 阈值是合理的工程启发式，但非标准持久 Betti 定义（标准为固定尺度 ε 或仅 essential classes）。spec §3 未文档化此阈值。
- **修复:** 无代码改动；spec §3 增补阈值选择说明。

### NEW-I3 — Phase 4 测试覆盖缺口

- **位置:** [tests/test_causal_emergence_emergence_cycle.py](file:///workspace/tests/test_causal_emergence_emergence_cycle.py)
- **描述:** 当前 36 个测试覆盖基础场景，但以下未覆盖：
  - 大 `dim > 64` 行为
  - `n_features > dim` 场景
  - 强制模块失败（monkeypatch）的降级路径
  - warnings 字符串格式断言
  - 并发调用（线程安全）
  - `reference_action = 0` 回退分支
  - `test_higher_persistence_increases_score` 名不副实（仅比较 persistence_entropy）
- **修复:** 增补 7 项测试覆盖以上场景。

### NEW-I4 — 导入图无环（正向确认）

- 仅 `engine.py` 和 `__init__.py` 导入兄弟模块；5 个叶子模块仅从 `.rules` 导入。无循环导入风险。

### NEW-I5 — `__init__.py` 导出与 spec §10 API 表面一致（正向确认）

- `__all__` 列出 7 个符号：`EmergenceRules`, `PersistentHomologyPerceiver`, `CausalInferenceEngine`, `DifferentialGenerator`, `HamiltonianSampler`, `ChaoticAssociativeMemory`, `CausalEmergenceEngine`。无私有符号泄漏。

### NEW-I6 — 评分权重、裁剪、Python 浮点类型均正确（正向确认）

- 权重 `0.30 + 0.20 + 0.20 + 0.15 + 0.15 = 1.00` ✓
- 5 项分别 `np.clip` 到 `[0, 1]` ✓
- 最终 `float(np.clip(score, 0.0, 1.0))` 返回 Python 浮点 ✓
- `emergence_score` 在返回 dict 中用 `float(...)` 包装 ✓
- `n_edges`、`iterations` 等用 `int(...)` ✓

### NEW-I7 — RNG 共享正确接线（正向确认，但见 NEW-M6 线程安全）

- 引擎构造函数将同一 `Generator` 实例传递给 5 个模块，确保可复现性。测试 `test_causal_emergence_engine.py` 验证共享 RNG 传播。单线程使用下正确。

---

## 7. 修复优先级与执行计划

### 立即修复（本次提交）

| ID | 严重度 | 修复内容 |
|----|--------|----------|
| NEW-H1 | HIGH | 重写 Meek R1 实现为正确规则，添加单元测试 |
| NEW-M1 | MEDIUM | 删除 topology.py:270 的死表达式 |
| NEW-M2 | MEDIUM | 将 posterior 占位符 std 改为 `np.full(n_features, 1e6)` |
| NEW-M3 | MEDIUM | 删除 chaotic_memory.store() 的死 if 块 |
| NEW-L2 | LOW | 更新 `_break_cycles` 注释移除"magnitude"误导 |

### 文档化（仅改 docstring，不改实现）

| ID | 严重度 | 处理方式 |
|----|--------|----------|
| NEW-H2 | HIGH | 在 `counterfactual` docstring 明确标注 "unit-weight simplified" 偏离 spec §4.3 |
| NEW-M4 | MEDIUM | 在 chaotic_memory docstring 标注 `dt=0.01` 硬编码 |
| NEW-M5 | MEDIUM | 在 spec §5/§7 文档化轨迹形状差异 |
| NEW-M6 | MEDIUM | 在引擎类 docstring 标注 "NOT thread-safe" |
| NEW-L1 | LOW | 在 spec §7.3 补充 `nearest_pattern` 字段 |
| NEW-L3 | LOW | 在 hmc.py 添加注释说明 ESS = n 的争议 |
| NEW-I1 | INFO | 在 spec §8.3 增补 `reference_action` 量纲说明 |
| NEW-I2 | INFO | 在 spec §3 增补 persistence_threshold 说明 |

### 测试增补

| ID | 内容 |
|----|------|
| NEW-I3-a | 大 dim 行为测试 |
| NEW-I3-c | 强制模块失败的降级路径测试（monkeypatch） |
| NEW-I3-f | `reference_action = 0` 回退分支测试 |
| NEW-I3-g | 重命名 `test_higher_persistence_increases_score` 或扩展断言 |

---

## 8. 验证清单

修复完成后必须验证：

- [ ] `ruff check src/zero_data_model/causal_emergence/ tests/test_causal_emergence_*.py` clean
- [ ] `python -m pytest tests/test_causal_emergence_*.py` 全部通过（当前基线 200）
- [ ] 新增 Meek R1 单元测试通过
- [ ] 新增 monkeypatch 降级测试通过
- [ ] 全量回归（除 `test_api.py`/`test_property_invariants.py` 已知缺失 fastapi/hypothesis）无新增失败
- [ ] git commit 修复

---

**审查人:** Agent (military-grade review)
**审查日期:** 2026-07-20
**审查工具:** 静态阅读 + 集成点审计 + 测试覆盖分析
