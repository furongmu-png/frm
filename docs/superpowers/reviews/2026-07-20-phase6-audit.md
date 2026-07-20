# Phase 6 实现代码审查报告

审查日期: 2026-07-20
审查对象: zero_data_model Phase 6 (能力层 + 外观/CLI/API/MCP 四入口)
审查基准: spec `2026-07-20-phase6-capabilities-design.md`(已批准)
审查方法: 静态阅读 + Grep 模式扫描 + 测试套件核对

---

## 1. 审查范围

本次审查覆盖 Phase 6 全部代码层与四类入口,具体文件如下:

| 层 | 文件 | 行数 | 关键内容 |
|---|---|---|---|
| 能力层 | `src/zero_data_model/capabilities/memory.py` | 438 | EpisodicMemory / WorkingMemory / ContextMemory / MemoryConsolidator |
| 能力层 | `src/zero_data_model/capabilities/memory_advanced.py` | 474 | HierarchicalMemory / SpreadingActivationMemory / ForgetfulMemory / MemoryIndexer |
| 能力层 | `src/zero_data_model/capabilities/planning.py` | 416 | TrajectoryPlanner / GoalDecomposer / ActionSequencer / HierarchicalPlanner |
| 能力层 | `src/zero_data_model/capabilities/planning_advanced.py` | 411 | MonteCarloTreePlanner / SymbolicPlanner / PolicyGradientPlanner / ContingencyPlanner |
| 能力层 | `src/zero_data_model/capabilities/multimodal.py` | 403 | CrossModalAligner / SharedLatentSpace / ModalityFuser / ModalityEncoder |
| 能力层 | `src/zero_data_model/capabilities/multimodal_advanced.py` | 263 | AttentionBasedFuser / ContrastiveAligner / MultimodalRetriever |
| 能力层 | `src/zero_data_model/capabilities/rl.py` | 411 | SyntheticMDP / QLearner / PolicyOptimizer / ValueFunction |
| 能力层 | `src/zero_data_model/capabilities/rl_advanced.py` | 393 | DynaQ / MonteCarloTreeSearch / PosteriorSampling |
| 规则层 | `src/zero_data_model/capabilities/rules.py` | 442 | MemoryRules / PlanningRules / MultimodalRules / RLRules |
| 入口-外观 | `src/zero_data_model/model.py` | 1950 | Phase 6 facade 方法 (约 1546–1950 行) |
| 入口-CLI | `src/zero_data_model/__main__.py` | 622 | 12 个 Phase 6 子命令 |
| 入口-API | `src/zero_data_model/api.py` | 1877 | 12 个 Phase 6 FastAPI 端点 |
| 入口-MCP | `src/zero_data_model/mcp_server.py` | 983 | 12 个 Phase 6 MCP 工具 (总 34) |

共计 30 个 Phase 6 能力类、34 个 MCP 工具、12 个 CLI 子命令、12 个 FastAPI 端点,已与 spec 中"四入口全对称"目标一致。

---

## 2. 测试覆盖矩阵

`tests/test_phase6_integration.py` 共 612 行,9 个测试类,覆盖四入口的等价性。`conftest.py` 中 `autouse` fixture 在每个测试前 `np.random.seed(42)`,保证使用全局 RNG 的测试也确定。

| 能力域 | Facade 测试 | CLI 测试 | API 测试 | MCP 测试 | 跨层一致性测试 |
|---|---|---|---|---|---|
| Memory | encode / retrieve / consolidate | encode / retrieve / consolidate | encode(含 label+NaN) / retrieve / consolidate | 同 Facade 签名 | 通过 |
| Planning | plan_trajectory / decompose_goal / sequence_actions | trajectory / decompose / sequence | trajectory(含形状不匹配) / decompose / sequence | 通过 | 通过 |
| Multimodal | fit / align / fuse / contrastive_loss | (CLI 未全展开) | align / fuse / contrastive | 通过 | 通过 |
| RL | step_mdp / train_q_learner / search_rl_mcts | step / train-q / search-mcts | step(含负值) / train-q / search-mcts | 通过 | 通过 |
| 工具总数 | — | — | — | `test_total_tool_count` 断言 == 34 | 通过 |

测试统计:Phase 6 共 369 项测试 + 68 项回归测试全部通过(用户上下文已确认)。

**测试覆盖缺口**(非阻塞):
- Multimodal CLI 子命令(`multimodal align/fuse/contrastive`)未在 `TestCLIPhase6` 中以独立用例覆盖,仅由 Facade 与 API 路径间接验证。建议后续补充。
- `test_multimodal_align_endpoint` 使用 `default_rng(0)` 生成的随机高斯矩阵作为输入。CCA 在协方差奇异时会进入回退路径(单位向量),虽已被实现处理,但若输入维度/样本数极端化,理论仍可能命中退化分支。建议增加一个明确测试奇异协方差的用例。
- `test_rl_step_negative_state_returns_400` 实际接受 400 或 422 两种状态码,见测试内注释,属已知灵活性,已在代码注释中记录。

---

## 3. 代码质量审查

总体代码质量良好:模块文件顶部均有简明的 docstring,辅助函数 `_sanitize_vector` / `_sanitize_state` / `_sanitize_obstacles` / `_ensure_finite` 统一在入口处拒绝非有限输入,避免 NaN/Inf 在算法内部扩散。

| 维度 | 评价 | 备注 |
|---|---|---|
| 模块划分 | 优 | 按"基础/高级"两文件拆分,类边界清晰,30 个类全部由 `capabilities/__init__.py` 用 `contextlib.suppress(ImportError)` 导出 |
| 文档字符串 | 良 | 每个类与公共方法均有 docstring;部分公式未引用来源(Ebbinghaus、Collins & Loftus 在 memory_advanced.py 顶部已注明) |
| 类型注解 | 良 | 函数签名均有注解,部分内部局部变量省略,可接受 |
| 错误处理 | 良 | 能力层抛 ValueError,API 层转 400,MCP 层经 `@_error_to_dict` 转 `{"error": ...}` |
| 复杂度 | 中 | `MemoryConsolidator.consolidate`、`HierarchicalPlanner.plan` 内嵌套较深,但行数可控 |
| 命名一致性 | 良 | `_sanitize_*` / `_cosine_similarity` / `_to_py` 等跨文件同名,语义统一 |

**轻度代码异味**(非阻塞,不要求本次修改):

- `rl_advanced.py:362` `PosteriorSampling.value_iteration` 末行
  ```python
  policy = np.argmax(Q, axis=1) if 'Q' in dir() else np.zeros(self.n_states, dtype=int)
  ```
  `'Q' in dir()` 在 `Q = R + gamma * ...` 赋值之后恒为 True,`else` 分支为不可达死代码。该写法同时削弱可读性,建议改为
  ```python
  policy = np.argmax(Q, axis=1)
  ```
  并依赖循环至少执行 1 次(已由 `for _ in range(int(n_iters))` 保证)或在 `n_iters == 0` 时显式处理。

- `multimodal.py:178 / 198` `SharedLatentSpace` 在 `fit` / `_init_attention` 内部以 `np.random.default_rng(0)` 创建 RNG,绕过了构造器 `rng` 参数链路。详见第 8 节。

- `api.py:111` `datetime.utcnow().strftime(...)` 在 Python 3.12+ 已被弃用,建议改为 `datetime.now(timezone.utc)`。当前不影响功能,但会在高版本 Python 上发出 DeprecationWarning。

---

## 4. 跨层一致性

四入口均委托到底层能力类的同一方法,无独立分支实现。下表抽样核对 12 个 Phase 6 调用链:

| 能力 | Facade (model.py) | CLI (__main__.py) | API (api.py) | MCP (mcp_server.py) | 底层方法 |
|---|---|---|---|---|---|
| encode_memory | `EpisodicMemory.encode` | `memory encode` | `POST /memory/encode` | `memory_encode` | 同 |
| retrieve_memory | `EpisodicMemory.retrieve` | `memory retrieve` | `POST /memory/retrieve` | `memory_retrieve` | 同 |
| consolidate_memory | `MemoryConsolidator.consolidate` | `memory consolidate` | `POST /memory/consolidate` | `memory_consolidate` | 同 |
| plan_trajectory | `TrajectoryPlanner.plan` | `planning trajectory` | `POST /planning/trajectory` | `planning_trajectory` | 同 |
| decompose_goal | `GoalDecomposer.decompose` | `planning decompose` | `POST /planning/decompose` | `planning_decompose` | 同 |
| sequence_actions | `ActionSequencer.sequence` | `planning sequence` | `POST /planning/sequence` | `planning_sequence` | 同 |
| fit_cross_modal | `CrossModalAligner.fit` | (CLI 未覆盖) | `POST /multimodal/align` | `multimodal_align` | 同 |
| fuse_modalities | `ModalityFuser.fuse` | (CLI 未覆盖) | `POST /multimodal/fuse` | `multimodal_fuse` | 同 |
| contrastive_loss | `ContrastiveAligner.loss` | (CLI 未覆盖) | `POST /multimodal/contrastive` | `multimodal_contrastive` | 同 |
| step_mdp | `SyntheticMDP.step` | `rl step` | `POST /rl/step` | `rl_step` | 同 |
| train_q_learner | `QLearner.train` | `rl train-q` | `POST /rl/train-q` | `rl_train_q` | 同 |
| search_rl_mcts | `MonteCarloTreeSearch.search` | `rl search-mcts` | `POST /rl/search-mcts` | `rl_search_mcts` | 同 |

**发现一处真实签名不一致(BUG,需修复)**:

`model.py:1610-1615` 的 `index_memory` 外观方法:
```python
def index_memory(self, observations: np.ndarray, labels: list | None = None) -> dict:
    with self._lock:
        return self.memory_indexer.build_from(observations, labels=labels)
```
但 `MemoryIndexer.build_from` 的签名(`memory_advanced.py:458`)为:
```python
def build_from(self, observations: np.ndarray) -> dict:
```
`build_from` **不接受 `labels` 参数**。任何对 `index_memory(observations, labels=...)` 的调用都会抛 `TypeError: unexpected keyword argument 'labels'`,并经 `@_error_to_dict` / API 异常处理转为错误响应。

`tests/test_phase6_integration.py` 未直接调用 `index_memory`,因此该路径未被回归覆盖。**建议二选一修复**:
1. 在 `MemoryIndexer.build_from` 增加可选 `labels: list | None = None` 参数并落盘索引元数据;或
2. 从 `index_memory` 外观签名移除 `labels`,与底层保持一致。

其它 11 条调用链签名核对一致,无类似问题。

---

## 5. 安全审查

按用户要求核对四类安全维度:

### 5.1 API 速率限制

| 端点 | 限制 | 备注 |
|---|---|---|
| `/memory/encode` | `30/minute` | 默认 |
| `/memory/retrieve` | `30/minute` | |
| `/memory/consolidate` | `10/minute` | 较重负载,收紧 |
| `/planning/trajectory` | `30/minute` | |
| `/planning/decompose` | `30/minute` | |
| `/planning/sequence` | `30/minute` | |
| `/multimodal/align` | `30/minute` | |
| `/multimodal/fuse` | `30/minute` | |
| `/multimodal/contrastive` | `30/minute` | |
| `/rl/step` | `30/minute` | |
| `/rl/train-q` | `5/minute` | 最重负载,最严 |
| `/rl/search-mcts` | `10/minute` | 重负载 |

`slowapi` 缺失时降级为不限流(测试 fixture 会显式 `limiter.enabled = False`)。生产部署需确保 `slowapi` 已安装。

### 5.2 Pydantic 字段边界

| 字段 | 约束 | 防御目标 |
|---|---|---|
| `MemoryEncodeRequest.observation` | `min_length=1, max_length=4096` | CWE-400 |
| `MemoryRetrieveRequest.top_k` | `ge=1, le=100` | 资源耗尽 |
| `PlanningTrajectoryRequest.n_steps` | `ge=0, le=1000` | 资源耗尽 |
| `PlanningSequenceRequest.adjacency` | `min_length=1, max_length=200` | 矩阵爆炸 |
| `MultimodalAlignRequest.observations_a/b` | `min_length=2, max_length=1000` | SVD 复杂度 |
| `RLStepRequest.state/action` | `ge=0, le=4096` | 越界 |
| `RLTrainQRequest.n_episodes` | `ge=1, le=2000` | 训练时长 |
| `RLSearchMCTSRequest.n_simulations` | `ge=1, le=2000` | 搜索时长 |
| `RLSearchMCTSRequest.max_depth` | `ge=1, le=200` | 树深 |

### 5.3 输入体积上限

`MAX_BODY = 4 * 1024 * 1024` (4 MiB) 在请求体读取前强制;`_ensure_finite` 在端点入口拒绝 NaN/Inf。`test_memory_encode_nan_returns_400` 验证之(实际返回 400 或 422,见 §2)。

### 5.4 持久化根隔离

`persistence.set_persistence_root` 在测试 fixture 与生产部署中将所有写入沙箱化到指定根目录;路径名经严格正则校验,拒绝绝对路径、`..` 遍历与符号链接(见 `api.py` 模块 docstring)。Phase 6 端点未引入新的持久化路径,沿用既有沙箱。

### 5.5 MCP 层额外保护

`_to_py` 将 `np.floating` 中的 NaN/Inf 映射为 `None`,保证严格 JSON 兼容(避免 `json.loads` 在标准客户端上因非有限浮点抛错);`_error_to_dict` 将任意异常转为 `{"error": str}` 并经 `logging.exception` 落盘完整 traceback,无信息丢失。

**安全审查结论**:Phase 6 各入口在速率、字段、体积、路径四维均有显式防御,无新增安全风险。

---

## 6. 性能审查

| 算法 | 复杂度 | 风险点 | 实际处理 |
|---|---|---|---|
| EpisodicMemory.retrieve | O(N·d) 线性扫描 | N 受 `memory_capacity=128` 限制 | 安全 |
| MemoryIndexer.search | O(N·d) 矩阵乘 | L2 归一化后单次 matmul | 已用 `np.linalg.norm` 向量化 |
| TrajectoryPlanner.plan | O(n_steps·d) | n_steps ≤ 1000 (API 边界) | 安全 |
| ActionSequencer.sequence | O(V+E) Kahn | adjacency ≤ 200 | 安全 |
| MonteCarloTreePlanner | O(n_sim·depth) | n_sim ≤ 2000, depth ≤ 200 | 安全 |
| CrossModalAligner.fit (CCA) | O(d³) Cholesky+SVD | d 受 `MultimodalAlignRequest` ≤ 1000 限制 | 大维度时偏慢,但被速率限制 |
| AttentionBasedFuser | O(n²·d) 注意力 | n 受 `MultimodalAlignRequest` ≤ 1000 | 注意力矩阵 10⁶ 元素,可接受 |
| ContrastiveAligner.loss | O(n²·d) 相似度矩阵 | n ≤ 1000 | 同上 |
| QLearner.train | O(ep·steps·S·A) | ep ≤ 2000, steps 默认 100 | 已 5/min 限流 |
| MCTS.search | O(n_sim·depth·A) | n_sim ≤ 2000, depth ≤ 200 | 已 10/min 限流 |
| PosteriorSampling.value_iteration | O(iters·S·A·S) | iters 默认 50 | 安全 |

**性能审查结论**:所有算法复杂度均与 spec 一致,关键负载通过 API 字段上限 + 速率限制双重防御;无 O(2^N) 或未限界递归。

---

## 7. 数值稳定性审查

使用 Grep 对 8 个 Phase 6 源文件扫描 `np.(exp|log|linalg.norm|sqrt|std|var)` 模式,逐项核对保护措施:

### 7.1 `np.exp` 出现 5 处

| 位置 | 上下文 | 保护 | 评价 |
|---|---|---|---|
| `memory.py:256` | softmax over similarities | `exp / max(exp.sum(), 1e-12)` | 通过 |
| `memory_advanced.py:316` | Ebbinghaus `R = exp(-age/S)` | `S = max(S, 1e-6)` 上游保护 | 通过 |
| `rl.py:236` | policy softmax | `logits -= logits.max()` 后 exp,`max(exp.sum(), 1e-12)` | 通过 |
| `planning_advanced.py:319` | policy gradient softmax | 同上 max-subtraction + `max(exp.sum(), 1e-12)` | 通过 |
| `multimodal_advanced.py:95` | 多头注意力 softmax | `scores -= scores_max`,`np.maximum(exp.sum(...), 1e-12)` | 通过 |
| `multimodal_advanced.py:165` | InfoNCE 分子 softmax | `sim -= sim_max`,`np.maximum(denom, 1e-12)` | 通过 |

### 7.2 `np.log` 出现 2 处

| 位置 | 上下文 | 保护 | 评价 |
|---|---|---|---|
| `multimodal.py:347` | IDF `log((N+1)/(df+1)+1)` | `N ≥ 1, df ≥ 0` ⇒ 参数 ≥ 2.0,隐式安全 | 通过 |
| `multimodal_advanced.py:167` | InfoNCE `log(denom)` | `np.maximum(denom, 1e-12)` | 通过 |

### 7.3 `np.linalg.norm` 出现 13 处

| 位置 | 保护 | 评价 |
|---|---|---|
| `memory.py:57-58` | `na, nb`; `_cosine_similarity` 在零范数时返回 0.0 | 通过 |
| `memory_advanced.py:404 / 428 / 467` | `np.where(norms < 1e-12, 1.0, norms)` 归一化 | 通过 |
| `planning.py:139` | `np.linalg.norm(traj[-1] - goal) <= tol`,无除法 | 通过 |
| `multimodal.py:120-121` | `max(np.linalg.norm(...), 1e-12)` | 通过 |
| `multimodal_advanced.py:158-159` | `np.maximum(np.linalg.norm(...), 1e-12)` | 通过 |
| `multimodal_advanced.py:238 / 243` | `MultimodalRetriever` 归一化检索 | 通过 |
| `rl.py:261` | `logits_norm` 仅作输出,无除法 | 通过 |
| `rl_advanced.py:358` | 收敛判定 `norm(new_V - V) < 1e-6`,无除法 | 通过 |
| `planning_advanced.py:347` | `weight_norm` 仅作输出 | 通过 |

### 7.4 `np.sqrt` 出现 1 处

| 位置 | 上下文 | 保护 | 评价 |
|---|---|---|---|
| `rl_advanced.py:338` | `std = sqrt(posterior_var)` | `np.maximum(self.posterior_var, 0.0)` 防 NaN | 通过 |

### 7.5 除法与 matmul

- 所有除法均经 `max(..., 1e-12)` / `np.maximum(..., 1e-12)` 保护。
- `multimodal.py` CCA 路径使用 `np.errstate(divide="ignore", invalid="ignore")` + `np.nan_to_num`,并在 S[0] < 1e-12 时回退到第一单位向量。
- `multimodal_advanced.py:92` 注意力缩放 `1/math.sqrt(max(self.dk, 1))`,防 `dk=0`。
- `multimodal_advanced.py:167` InfoNCE `sim / max(temperature, 1e-6)`,防温度为 0。
- `multimodal.py:120-121` CCA 投影向量归一化 `max(np.linalg.norm(...), 1e-12)`。

**数值稳定性审查结论**:Phase 6 全部 24 处数值敏感模式均有显式保护,无未护栏除法/对数/指数/开方。CCA 在协方差奇异时走 `np.errstate` + 单位向量回退,InfoNCE 与 softmax 均使用 max-subtraction 稳定化。

---

## 8. 确定性审查

`conftest.py` 在每项测试前 `np.random.seed(42)`,但生产代码路径仍需满足"同种子同输出"的强不变量。本次发现 3 处违反"per-module RNG 经构造器注入"约定的代码:

| 位置 | 问题 | 影响 | 严重度 |
|---|---|---|---|
| `rl.py:74` `SyntheticMDP.step` | `np.random.choice(self.n_states, p=self.P[s, a])` 使用全局 RNG,而非构造器接收的 `rng` | 同种子下重复运行结果受全局 RNG 状态影响,且与 `QLearner`/`PolicyOptimizer` 共享同一全局流,破坏可复现性 | 中 |
| `rl.py:89` `SyntheticMDP.reset` | `np.random.randint(len(non_terminal))` 同上 | 同上 | 中 |
| `multimodal.py:178 / 198` `SharedLatentSpace` | `np.random.default_rng(0)` 硬编码种子 0,不接受构造器 `rng` | 跨实例同种子(0)产生相同初始化;无法由 facade `_child_rngs[6]` 控制 | 中 |
| `planning_advanced.py:178` `MonteCarloTreePlanner._rollout` | `actions[np.random.randint(len(actions))]` 使用全局 RNG,未用 `self.rng` | MCTS 模拟结果受全局 RNG 流影响;与同实例内 `PolicyGradientPlanner` 等共享全局流 | 中 |
| `multimodal_advanced.py:52` `AttentionBasedFuser` | `self.rng = rng or np.random.default_rng(0)` | 与构造器约定一致(缺省时回退),但缺省种子硬编码 0 | 低 |
| `rl_advanced.py:46 / 167 / 303` | `self.rng = rng or np.random.default_rng()` | 缺省时不传种子,非确定;但 facade 路径已显式注入 `_child_rngs[6]` | 低 |

**facade 注入路径核对(model.py)**:

| facade 方法 | 注入的 RNG | 评价 |
|---|---|---|
| `planning_policy_gradient` | `self._child_rngs[6]` | 通过 |
| `multimodal_attention_fuser` | `self._child_rngs[6]` | 通过(但 SharedLatentSpace 内部仍硬编码) |
| `rl_q_learner` | `self._child_rngs[6]` | 通过 |
| `rl_policy_optimizer` | `self._child_rngs[6]` | 通过 |
| `rl_value_function` | 无 RNG 需求 | 通过 |
| `rl_dyna_q` | `self._child_rngs[6]` | 通过 |
| `rl_mcts` | `self._child_rngs[6]` | 通过(但 `_rollout` 内部仍走全局) |
| `rl_posterior` | `self._child_rngs[6]` | 通过 |
| `step_mdp` | `SyntheticMDP(seed=seed)` 构造器种子 | **未注入实例 rng**,见上表 rl.py:74/89 |

**确定性审查结论**:facade 层 RNG 注入约定已建立并覆盖大部分类,但 `SyntheticMDP.step/reset`、`SharedLatentSpace.fit`、`MonteCarloTreePlanner._rollout` 三处绕过约定,直接影响"同种子同输出"的强不变量。建议作为 P1 修复(详见 §9)。

---

## 9. 已知问题与风险

按严重度排序:

| 编号 | 严重度 | 位置 | 描述 | 建议修复 |
|---|---|---|---|---|
| P6-AUDIT-001 | 高 (BUG) | `model.py:1614` + `memory_advanced.py:458` | `index_memory` 外观传 `labels=`,但 `MemoryIndexer.build_from` 不接受该参数。任何调用都会抛 `TypeError`,且未被测试覆盖。 | 二选一:(a) 在 `build_from` 增加可选 `labels` 参数;(b) 从外观签名移除 `labels`。 |
| P6-AUDIT-002 | 中 | `rl.py:74, 89` | `SyntheticMDP.step/reset` 使用全局 `np.random.*`,绕过构造器 `rng`,破坏可复现性。 | 改为 `self.rng.choice(...)` / `self.rng.integers(...)`;构造器需确保 `self.rng` 已落盘。 |
| P6-AUDIT-003 | 中 | `multimodal.py:178, 198` | `SharedLatentSpace` 硬编码 `default_rng(0)`,不接受 `rng` 参数。 | 在构造器增加 `rng` 参数并贯穿到 `fit` / `_init_attention`。 |
| P6-AUDIT-004 | 中 | `planning_advanced.py:178` | `MonteCarloTreePlanner._rollout` 使用全局 `np.random.randint`,未用 `self.rng`。 | 改为 `self.rng.integers(len(actions))`。 |
| P6-AUDIT-005 | 低 | `rl_advanced.py:362` | `'Q' in dir()` 恒为 True,死代码;可读性差。 | 简化为 `policy = np.argmax(Q, axis=1)`,并在 `n_iters == 0` 时显式处理。 |
| P6-AUDIT-006 | 低 | `api.py:111` | `datetime.utcnow()` 在 Python 3.12+ 已弃用。 | 改为 `datetime.now(timezone.utc)`。 |
| P6-AUDIT-007 | 低 | `tests/test_phase6_integration.py` | Multimodal CLI 子命令缺独立用例;`test_multimodal_align_endpoint` 未显式覆盖 CCA 退化分支。 | 补充 CLI 用例与奇异协方差用例。 |
| P6-AUDIT-008 | 低 | `multimodal_advanced.py:52`、`rl_advanced.py:46/167/303` | 缺省 `rng` 回退到 `default_rng()` 或 `default_rng(0)`,非确定或硬编码。 | facade 已显式注入,影响有限;可考虑统一回退种子常量。 |

**风险展望**:
- 若 `slowapi` 在生产未安装,所有速率限制静默失效。建议启动时检查并发出 WARNING。
- CCA 在样本数 < 维度数时协方差必奇异,当前回退路径已处理但未单测;若未来扩大 API 输入维度上限,需同步增加退化测试。

---

## 10. 审查结论

**总体裁定: PASS WITH NOTES(通过,带修复建议)**

Phase 6 实现在功能完整性、跨层对称性、数值稳定性、安全防御、性能边界五个维度均达到 spec 要求:

- 30 个能力类、12 个 CLI 子命令、12 个 FastAPI 端点、12 个 MCP 工具四入口全对称,签名与底层一致(除 §4 所述 `index_memory` 一处);
- 数值稳定性 24/24 处显式保护,无未护栏的 exp/log/norm/sqrt/除法;
- 安全四维(速率/字段/体积/路径)显式防御,无新增风险;
- 369 项 Phase 6 测试 + 68 项回归测试全部通过;
- 性能复杂度与 spec 一致,关键负载有速率限制兜底。

**必须在合入前修复**:
- P6-AUDIT-001(`index_memory` 签名不一致,BUG)。

**建议在下一迭代修复**:
- P6-AUDIT-002 / 003 / 004(三处 RNG 确定性问题,影响"同种子同输出"强不变量);
- P6-AUDIT-005 / 006(死代码与弃用 API,代码卫生)。

**可在后续补充**:
- P6-AUDIT-007 / 008(测试覆盖与回退种子统一)。

本次审查未发现阻塞合入的安全漏洞、数据竞争或崩溃级缺陷;P6-AUDIT-001 为唯一功能性 BUG 且影响面有限(单一外观方法未被测试覆盖),修复后即可宣告 Phase 6 完成。

---

## 11. 修复回执(2026-07-20 同日)

审查后立即对前述 P1/P2 项进行了就地修复并复跑全套测试,结果如下:

| ID | 状态 | 修复方式 |
|---|---|---|
| P6-AUDIT-001 | **已修复** | `memory_advanced.py:458` 的 `MemoryIndexer.build_from` 增加可选 `labels: list \| None = None` 参数;长度校验后写入 `self._labels`。外观 `model.py:1614` 不需改动。 |
| P6-AUDIT-002 | **已修复** | `rl.py:44` 的 `SyntheticMDP.__init__` 增加 `rng: np.random.Generator \| None = None` 参数并存为 `self.rng`;`step()` 改为 `self.rng.choice(...)`、`reset()` 改为 `self.rng.integers(...)`。外观 `model.py:602` 注入 `_child_rngs[6]`。 |
| P6-AUDIT-003 | **已修复** | `multimodal.py:169` 的 `SharedLatentSpace.__init__` 增加 `rng` 参数;构造与 `update()` 中 `dim_modal` 自适应分支均改用 `self.rng.standard_normal(...)`。外观 `model.py:575` 注入 `_child_rngs[6]`。 |
| P6-AUDIT-004 | **已修复** | `planning_advanced.py:66` 的 `MonteCarloTreePlanner.__init__` 增加 `rng` 参数;`_rollout` 改为 `self.rng.integers(len(actions))`。外观 `model.py:553` 注入 `_child_rngs[6]`。 |
| P6-AUDIT-005 | **已修复** | `rl_advanced.py:344` 的 `value_iteration` 在循环前初始化 `Q = R + gamma * (self.mdp.P @ V)`,删除 `'Q' in dir()` 死代码分支,简化为 `policy = np.argmax(Q, axis=1)`。 |
| P6-AUDIT-006 | 待修复 | `api.py:111` 的 `datetime.utcnow()` 弃用警告 — 非阻塞,后续迭代统一替换为 `datetime.now(datetime.UTC)`。 |
| P6-AUDIT-007 | 待补充 | Multimodal CLI 测试覆盖较薄 — 非阻塞。 |
| P6-AUDIT-008 | 待补充 | 部分 RNG 回退种子与 facade `_child_rngs` 约定的统一 — 非阻塞。 |

**修复后测试结果**:

- Phase 6 套件:369 / 369 通过
- v4 回归套件 (`test_api_emergence` + `test_mcp_emergence` + `test_cli_emergence`):68 / 68 通过
- 合计:437 / 437 通过,无回归

**最终裁定**:**PASS**(原 PASS WITH NOTES 中所列 P1/P2 阻塞项已全部就地修复并经测试验证)。可宣告 Phase 6 完成,准予合入主分支。

---

## 12. 第二轮回执(2026-07-20 同日)

针对 §11 中列为"待修复 / 待补充"的 P6-AUDIT-006 / 007 / 008 三项非阻塞遗留就地修复:

| ID | 状态 | 修复方式 |
|---|---|---|
| P6-AUDIT-006 | **已修复** | `api.py:38` 引入 `timezone`;`api.py:111` 由 `datetime.utcnow()` 改为 `datetime.now(timezone.utc)`,消除 `DeprecationWarning`。全仓 Grep 确认无残留 `datetime.utcnow()` 调用。 |
| P6-AUDIT-007 | **已修复** | `tests/test_phase6_integration.py::TestCLIPhase6` 新增 3 个 Multimodal CLI 测试:`test_multimodal_align_cli`、`test_multimodal_fuse_cli`、`test_multimodal_contrastive_cli`,覆盖 align / fuse / contrastive 三个子命令,验证 `fit`/`aligned`/`fused`/`loss` 字段。Multimodal CLI 测试覆盖由 0 提升到 3。 |
| P6-AUDIT-008 | **已修复** | `rl.py` (3 处) + `rl_advanced.py` (3 处) + `planning_advanced.py` (1 处) + `multimodal_advanced.py` (1 处) 共 4 文件 8 处 `rng or np.random.default_rng(...)` 全部统一为 `rng if rng is not None else np.random.default_rng(...)` 形式,语义更明确(仅在 `rng is None` 时回退,不会替换可能为 falsy 的 Generator 实例),代码风格与 §11 修复一致。 |

**第二轮测试结果**:

- Phase 6 套件:372 / 372 通过(新增 3 个 Multimodal CLI 测试)
- v4 回归套件:68 / 68 通过
- 合计:**440 / 440 通过,无回归**

**最终裁定**:**PASS**。所有审查发现的 8 项问题(P1×1 + P2×4 + P3×3)均已就地修复并经测试验证。Phase 6 实现完整,可宣告收尾。


