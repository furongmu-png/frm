# Phase 6 — 4 个新 Capability 设计规范（v1）

> **状态:** 草案 (2026-07-20)
> **作者:** Agent
> **范围:** 在 `src/zero_data_model/capabilities/` 下新增 Memory / Planning / Multimodal / RL 共 4 类 capability（8 个文件，约 16-20 个 capability 类），并接入 `ZeroDataModel` facade、`__main__.py` CLI、`api.py` Web API、`mcp_server.py` MCP 三层入口。
> **依赖契约:** 沿用 §1 公理 — 仅 numpy / scipy / 标准库；不引入 PyTorch / TensorFlow / JAX / gym / stable-baselines3 / opencv / librosa / transformers。
> **关联文档:**
> - `docs/superpowers/specs/2026-07-19-causal-emergence-engine-design.md` — causal_emergence 子包（与本 spec 紧密集成）
> - `docs/superpowers/reviews/2026-07-20-causal-emergence-engine-v4-integration-entry-points-review.md` — v4 审查报告（本 spec 沿用其入口字段一致性约束）

---

## 1. 动机与范围

v4 已完成因果涌现引擎与三层集成入口。Phase 6 在此之上引入 4 类 capability，将引擎能力封装为可复用的领域适配器：

| 类别 | 文件 | 类数 | 与 causal_emergence 的集成 |
|------|------|------|---------------------------|
| Memory | `memory.py` + `memory_advanced.py` | 4 + 4 = 8 | `ChaoticAssociativeMemory` 提供联想检索底座 |
| Planning | `planning.py` + `planning_advanced.py` | 4 + 4 = 8 | `DifferentialGenerator` 提供 trajectory；`CausalInferenceEngine` 提供干预 |
| Multimodal | `multimodal.py` + `multimodal_advanced.py` | 4 + 3 = 7 | `emergence_cycle` 提供跨模态融合循环 |
| RL | `rl.py` + `rl_advanced.py` | 4 + 3 = 7 | `HamiltonianSampler` 提供策略采样；`DifferentialGenerator` 提供动作轨迹 |

总计约 30 个 capability 类、4 个 Rules dataclass、6 个 facade 挂载点、12 个 CLI 子命令、12 个 API 端点、12 个 MCP 工具。

### 1.1 不在范围内

- 训练 / 微调任何神经网络（仍坚持零数据）。
- 接入外部 RL gym 环境（用合成 MDP）。
- 跨进程 / 跨机器分布式记忆（单进程）。
- 多 agent 协同（单 agent）。
- 视觉 / 音频原始模态处理（沿用 v3 的 `vision_*` / `audio_*` capability，本 spec 只做"跨模态对齐"层）。

---

## 2. 通用模式（沿用现有 capability 约定）

每个新 capability 类遵循以下契约（来自现有 `causal.py` / `reasoning.py` 模式）：

1. **构造函数**：`(dim=64, <core_deps>=None, rules: XxxRules | None = None)`。核心依赖惰性实例化回退。
2. **不继承 `CognitiveModule`**，不进入 `self.modules`，不影响 `_N_COGNITIVE_MODULES = 6`。
3. **方法返回 `dict`**，numpy 标量转 `float(...)` / `int(...)`，numpy 数组保留原类型。
4. **空输入守卫**：方法首段处理空数组/形状不匹配，返回同 schema 的零值字典。
5. **数值健壮性**：涉及除法 / log / sqrt 用 `np.errstate(invalid="ignore", divide="ignore")` + `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)`。
6. **文件首行**：`# src/zero_data_model/capabilities/<name>.py` 路径注释 + 模块 docstring 强调"无外部 ML 库依赖"。
7. **测试组织**：`tests/test_capabilities_<domain>.py` 与 `tests/test_capabilities_<domain>_advanced.py`；顶部 `np.random.seed(42)` autouse fixture；按类分块，每类先测 Rules 默认值，再测方法 basic / empty / invalid / structure。

### 2.1 Rules dataclass 契约

```python
@dataclass
class MemoryRules(DomainRules):
    # 字段命名后缀：_threshold / _depth / _max_iter / _epsilon / _horizon
    memory_capacity: int = 128
    memory_decay: float = 0.95
    memory_similarity_threshold: float = 0.7
    # ...

    def __post_init__(self):
        super().__init__()
        self.rules = {
            "memory_capacity": self.memory_capacity,
            # ... 重复字段名作为 key
        }
```

---

## 3. Memory Capability（`memory.py` + `memory_advanced.py`）

### 3.1 目的

将 `causal_emergence.chaotic_memory.ChaoticAssociativeMemory`（Lorenz 吸引子盆记忆）封装为可复用的领域记忆层，提供情景记忆、工作记忆、对话上下文记忆三类高层抽象。

### 3.2 类列表

#### `memory.py`（base，4 类）

1. **`EpisodicMemory`** — 情景记忆：存储 `(observation, label, timestamp)` 三元组；按相似度检索 top-k；与 `ChaoticAssociativeMemory.recall()` 集成做检索时的 Lorenz 盆增强。

2. **`WorkingMemory`** — 工作记忆：维护最近 N 条观测的滑动窗口 + 注意力权重；支持 push / pop / peek；权重由与"焦点"向量的相似度决定。

3. **`ContextMemory`** — 对话上下文记忆：存储 `(user_msg, agent_msg, turn_id)` 序列；提供"摘要"方法（取最近 N 条 embedding 均值）；与 `nlp.SentenceEncoder` 集成（可选）。

4. **`MemoryConsolidator`** — 记忆固化：将 WorkingMemory 中权重高的项固化为 EpisodicMemory 项；删除相似度 > 阈值的冗余项；触发时机由调用者决定。

#### `memory_advanced.py`（advanced，4 类）

1. **`HierarchicalMemory`** — 层级记忆：三层（短期 / 中期 / 长期），固化时间阈值由 `MemoryRules.hierarchy_consolidation_times` 控制；与 `MemoryConsolidator` 集成。

2. **`SpreadingActivationMemory`** — 扩散激活记忆：基于 Collins & Loftus (1975) 模型，检索时沿相似度图扩散激活；扩散深度 `memory_spreading_depth`。

3. **`ForgetfulMemory`** — 遗忘记忆：基于 Ebbinghaus 遗忘曲线 `R = exp(-t / S)`，旧项权重按时间衰减；可恢复（未真正删除）。

4. **`MemoryIndexer`** — 记忆索引器：构建 Faiss-lite 风格的倒排索引（纯 numpy），加速 top-k 检索；支持增量更新。

### 3.3 关键 API

```python
class EpisodicMemory:
    def __init__(self, dim: int = 64, chaotic_memory=None, rules: MemoryRules | None = None): ...

    def encode(self, observation: np.ndarray, label: str | int | None = None) -> dict:
        """Store an observation. Returns {'id': int, 'label': ..., 'timestamp': float}."""

    def retrieve(self, query: np.ndarray, top_k: int = 5) -> dict:
        """Top-k retrieval. Returns {'items': list[dict], 'similarities': list[float]}."""

    def forget(self, ids: list[int]) -> dict:
        """Remove items by id. Returns {'forgotten': int}."""

    def clear(self) -> dict:
        """Clear all. Returns {'cleared': int}."""
```

### 3.4 Rules 字段

```python
memory_capacity: int = 128
memory_decay: float = 0.95
memory_similarity_threshold: float = 0.7
memory_top_k: int = 5
memory_consolidation_weight: float = 0.8
memory_spreading_depth: int = 2
memory_ebbinghaus_S: float = 1.0  # 遗忘曲线稳定性常数
memory_hierarchy_consolidation_times: tuple = (60, 3600)  # (短期→中期, 中期→长期) 秒
```

---

## 4. Planning Capability（`planning.py` + `planning_advanced.py`）

### 4.1 目的

将 `causal_emergence.DifferentialGenerator`（阻尼最小作用量轨迹）与 `CausalInferenceEngine`（干预推断）封装为可复用的规划层。

### 4.2 类列表

#### `planning.py`（base，4 类）

1. **`HierarchicalPlanner`** — 层级规划：将抽象目标分解为子目标序列；与 `causal_emergence.discover_causal_dynamics()` 集成做"动作依赖图"分解。

2. **`GoalDecomposer`** — 目标分解：基于规则模板（AND/OR 树）将目标分解为原子动作；模板来自 `PlanningRules.goal_templates`。

3. **`TrajectoryPlanner`** — 轨迹规划：包装 `DifferentialGenerator.generate()`，输入 `(start_state, goal_state, obstacles)`，输出 `(actions, trajectory, action_cost)`。

4. **`ActionSequencer`** — 动作序列化：将 DAG 形式的动作集（来自 `CausalInferenceEngine.discover()`）拓扑排序为线性序列；检测循环并贪心破环。

#### `planning_advanced.py`（advanced，4 类）

1. **`MonteCarloTreePlanner`** — MCTS：经典 UCT 算法，4 阶段（选择 / 扩展 / 模拟 / 回传）；与 `HamiltonianSampler` 集成做 rollout 采样。

2. **`SymbolicPlanner`** — 符号规划：STRIPS 风格，输入 `(initial_state, goal_state, operators)`，返回动作序列；用图搜索（A* + 启发式 = goal_distance）。

3. **`PolicyGradientPlanner`** — 策略梯度规划：将规划问题形式化为策略优化，softmax over actions；与 `rl.PolicyOptimizer` 集成。

4. **`ContingencyPlanner`** — 应急规划：生成多套 plan，每套标注前置条件；当环境变化触发 fallback。

### 4.3 关键 API

```python
class TrajectoryPlanner:
    def __init__(self, dim: int = 64, differential_generator=None, rules: PlanningRules | None = None): ...

    def plan(self, start_state: np.ndarray, goal_state: np.ndarray,
             obstacles: np.ndarray | None = None, n_steps: int = 32) -> dict:
        """Generate trajectory. Returns {'trajectory', 'actions', 'action_cost', 'converged', 'obstacle_violations'}."""
```

### 4.4 Rules 字段

```python
planning_horizon: int = 32
planning_max_depth: int = 5
planning_mcts_simulations: int = 100
planning_mcts_ucb_c: float = 1.414
planning_goal_tolerance: float = 1e-3
planning_operator_set: tuple = ("move", "rotate", "grasp", "release")
```

---

## 5. Multimodal Capability（`multimodal.py` + `multimodal_advanced.py`）

### 5.1 目的

提供跨模态对齐层 — 将不同模态（文本 / 图像 / 音频 embedding）映射到共享潜空间，与 `causal_emergence.emergence_cycle()` 集成做跨模态融合。

### 5.2 类列表

#### `multimodal.py`（base，4 类）

1. **`CrossModalAligner`** — 跨模态对齐：基于 CCA（Canonical Correlation Analysis，纯 numpy 实现），将两个模态的 embedding 矩阵对齐到共享子空间。

2. **`SharedLatentSpace`** — 共享潜空间：维护一个 `(dim_modal, dim_shared)` 投影矩阵；增量更新（在线 CCA）。

3. **`ModalityFuser`** — 模态融合器：输入多个模态的 embedding，输出融合 embedding；融合策略 = "mean" / "concat" / "weighted"（权重来自 `MultimodalRules.fusion_weights`）。

4. **`ModalityEncoder`** — 模态编码器：将原始模态（如文本 token id 序列、图像 patch embedding）映射到固定维度向量；用纯 numpy 的 BoW / TF-IDF / PCA。

#### `multimodal_advanced.py`（advanced，3 类）

1. **`AttentionBasedFuser`** — 注意力融合：scaled dot-product attention，纯 numpy；输入 `(Q, K, V)`，输出 attention 加权融合。

2. **`ContrastiveAligner`** — 对比对齐：InfoNCE 损失（纯 numpy），拉近正样本对、推开负样本对；与 `HamiltonianSampler` 集成做负样本采样。

3. **`MultimodalRetriever`** — 多模态检索：跨模态 top-k 检索（如"以图搜文"），基于 `SharedLatentSpace` 的投影。

### 5.3 关键 API

```python
class CrossModalAligner:
    def __init__(self, dim: int = 64, rules: MultimodalRules | None = None): ...

    def fit(self, modality_a: np.ndarray, modality_b: np.ndarray) -> dict:
        """CCA fit. Returns {'projection_a', 'projection_b', 'correlation'}."""

    def align(self, embedding: np.ndarray, source: str) -> dict:
        """Project embedding to shared space. source ∈ {'a', 'b'}. Returns {'aligned': np.ndarray}."""
```

### 5.4 Rules 字段

```python
multimodal_shared_dim: int = 32
multimodal_alignment_method: str = "cca"  # 'cca' | 'pls' | 'random'
multimodal_fusion_strategy: str = "mean"  # 'mean' | 'concat' | 'weighted'
multimodal_fusion_weights: tuple = (0.5, 0.5)
multimodal_attention_heads: int = 4
multimodal_contrastive_temperature: float = 0.07
```

---

## 6. RL Capability（`rl.py` + `rl_advanced.py`）

### 6.1 目的

零数据 RL：在合成 MDP（SyntheticMDP）上跑经典算法；与 `HamiltonianSampler` 集成做策略后验采样，与 `DifferentialGenerator` 集成做动作轨迹生成。

### 6.2 类列表

#### `rl.py`（base，4 类）

1. **`SyntheticMDP`** — 合成 MDP：随机生成 transition `(S, A, P, R, gamma)`；提供 `step(state, action) -> (next_state, reward, done, info)`；确定性 seed。

2. **`QLearner`** — Q-learning：经典 tabular Q-learning，`Q[s, a] += alpha * (r + gamma * max_a' Q[s', a'] - Q[s, a])`；epsilon-greedy 探索。

3. **`PolicyOptimizer`** — 策略优化：softmax 策略 + REINFORCE 风格梯度估计（纯 numpy）；与 `HamiltonianSampler` 集成做策略后验。

4. **`ValueFunction`** — 值函数：维护 V[s] 表，支持 TD(0) / TD(λ) 更新；Monte Carlo 评估。

#### `rl_advanced.py`（advanced，3 类）

1. **`DynaQ`** — Dyna-Q：Q-learning + 模型学习 + 想象 rollout；planning steps 来自 `RLRules.dyna_planning_steps`。

2. **`MonteCarloTreeSearch`** — MCTS：与 `planning_advanced.MonteCarloTreePlanner` 共享代码，但 RL 版本聚焦"环境模型未知，从经验学习"。

3. **`PosteriorSampling`** — 后验采样：PSRL（Posterior Sampling for Reinforcement Learning），与 `HamiltonianSampler` 集成做后维护一个 posterior over R。

### 6.3 关键 API

```python
class QLearner:
    def __init__(self, n_states: int = 16, n_actions: int = 4,
                 mdp: SyntheticMDP | None = None, rules: RLRules | None = None): ...

    def update(self, state: int, action: int, reward: float, next_state: int) -> dict:
        """TD update. Returns {'td_error': float, 'q_value': float}."""

    def select_action(self, state: int) -> dict:
        """Epsilon-greedy. Returns {'action': int, 'q_values': np.ndarray, 'greedy': bool}."""

    def train(self, n_episodes: int = 100) -> dict:
        """Run full training. Returns {'episode_rewards': list, 'final_policy': np.ndarray}."""
```

### 6.4 Rules 字段

```python
rl_gamma: float = 0.95  # 折扣因子
rl_alpha: float = 0.1  # 学习率
rl_epsilon: float = 0.1  # 探索率
rl_n_states: int = 16
rl_n_actions: int = 4
rl_dyna_planning_steps: int = 10
rl_mcts_simulations: int = 100
rl_posterior_burn_in: int = 50
```

---

## 7. ZeroDataModel facade 集成

### 7.1 `model.py` 顶部 import

```python
from .capabilities.memory import EpisodicMemory, WorkingMemory, ContextMemory, MemoryConsolidator
from .capabilities.memory_advanced import HierarchicalMemory, SpreadingActivationMemory, ForgetfulMemory, MemoryIndexer
from .capabilities.planning import HierarchicalPlanner, GoalDecomposer, TrajectoryPlanner, ActionSequencer
from .capabilities.planning_advanced import MonteCarloTreePlanner, SymbolicPlanner, PolicyGradientPlanner, ContingencyPlanner
from .capabilities.multimodal import CrossModalAligner, SharedLatentSpace, ModalityFuser, ModalityEncoder
from .capabilities.multimodal_advanced import AttentionBasedFuser, ContrastiveAligner, MultimodalRetriever
from .capabilities.rl import SyntheticMDP, QLearner, PolicyOptimizer, ValueFunction
from .capabilities.rl_advanced import DynaQ, MonteCarloTreeSearch, PosteriorSampling
```

### 7.2 `__init__` 挂载

```python
self.memory_rules = MemoryRules()
self.memory_episodic = EpisodicMemory(dim=dim, chaotic_memory=self.emergence.chaotic_memory, rules=self.memory_rules)
self.memory_working = WorkingMemory(dim=dim, rules=self.memory_rules)
# ... 共 8 个 memory_*
self.planning_rules = PlanningRules()
self.planning_trajectory = TrajectoryPlanner(dim=dim, differential_generator=self.emergence.differential, rules=self.planning_rules)
# ... 共 8 个 planning_*
self.multimodal_rules = MultimodalRules()
self.multimodal_aligner = CrossModalAligner(dim=dim, rules=self.multimodal_rules)
# ... 共 7 个 multimodal_*
self.rl_rules = RLRules()
self.rl_q_learner = QLearner(n_states=rl_rules.rl_n_states, n_actions=rl_rules.rl_n_actions, rules=self.rl_rules)
# ... 共 7 个 rl_*
```

### 7.3 facade 方法

约 30 个 facade 方法（每个 capability 至少 1 个），命名约定 `<verb>_<noun>`：
- `encode_episodic_memory(observation, label=None) -> dict`
- `retrieve_episodic_memory(query, top_k=5) -> dict`
- `plan_trajectory(start, goal, obstacles=None, n_steps=32) -> dict`
- `align_cross_modal(modality_a, modality_b) -> dict`
- `train_q_learner(n_episodes=100) -> dict`
- ...

所有方法持 `with self._lock:` 守卫。

### 7.4 `__init__.py` 登记

新增 8 个 `contextlib.suppress(ImportError)` 块（4 base + 4 advanced），`__all__` 列表追加约 30 个新类名。

---

## 8. 三层入口扩展

### 8.1 CLI（`__main__.py`）

新增 4 个 subparser group：`memory` / `planning` / `multimodal` / `rl`，每组 3 个子命令。总计 12 个新子命令：

- `memory {encode, retrieve, consolidate}`
- `planning {plan, decompose, sequence}`
- `multimodal {align, fuse, retrieve}`
- `rl {train, evaluate, plan_action}`

### 8.2 Web API（`api.py`）

新增 12 个 FastAPI 端点：

- `POST /memory/{encode, retrieve, consolidate}`
- `POST /planning/{plan, decompose, sequence}`
- `POST /multimodal/{align, fuse, retrieve}`
- `POST /rl/{train, evaluate, plan_action}`

每个端点：`Depends(verify_api_key)` + `@_limit("30/minute")`（train 限 `5/minute`）+ `_ensure_finite` + `Field(min_length=, max_length=)`。

### 8.3 MCP（`mcp_server.py`）

新增 12 个 `@_error_to_dict` 工具：

- `memory_encode` / `memory_retrieve` / `memory_consolidate`
- `planning_plan_trajectory` / `planning_decompose_goal` / `planning_sequence_actions`
- `multimodal_align` / `multimodal_fuse` / `multimodal_retrieve`
- `rl_train` / `rl_evaluate` / `rl_plan_action`

工具总数从 22 → 34。所有输入经 `_ensure_finite`，输出经 `_to_py()`。

### 8.4 入口字段一致性（沿用 v4.2 约束）

- CLI 输出 schema 与 Web API response schema 与 MCP `_to_py()` 输出字段保持一致。
- 字段缺失时（如 `obstacle_violations` 在无 obstacles 时）三层入口都应一致地省略。
- 三层入口都路由通过 `ZeroDataModel` facade，不绕过 `model._lock`。

---

## 9. 测试覆盖目标

### 9.1 单元测试（按 capability 组织）

| 文件 | 测试数 | 内容 |
|------|--------|------|
| `test_capabilities_memory.py` | ≥ 25 | Rules 3 + 每类 4 方法 × 5 测 = 23 + 边界 5 |
| `test_capabilities_memory_advanced.py` | ≥ 25 | 同上 |
| `test_capabilities_planning.py` | ≥ 25 | 同上 |
| `test_capabilities_planning_advanced.py` | ≥ 25 | 同上 |
| `test_capabilities_multimodal.py` | ≥ 25 | 同上 |
| `test_capabilities_multimodal_advanced.py` | ≥ 20 | 3 类 × 5 测 + 边界 |
| `test_capabilities_rl.py` | ≥ 25 | 同上 |
| `test_capabilities_rl_advanced.py` | ≥ 20 | 3 类 × 5 测 + 边界 |
| `test_cli_phase6.py` | ≥ 24 | 12 子命令 × (成功 + 边界) |
| `test_api_phase6.py` | ≥ 24 | 12 端点 × (成功 + 边界) + 鉴权 |
| `test_mcp_phase6.py` | ≥ 36 | 12 工具 × (成功 + 边界 + 错误 + 序列化) |
| `test_zero_data_model_phase6_integration.py` | ≥ 20 | 4 capability × 5 测，端到端 |

**目标测试总数 ≥ 290**，加现有 324 后总测试数 ≥ 614。

### 9.2 边界情况清单（每类至少覆盖）

- 空输入 / 单点输入
- NaN / Inf 输入
- shape 不匹配
- dim=1 / dim=128
- 确定性（同 seed 同结果）
- 数值健壮性（极端值不崩溃）
- 向后兼容（rules=None 时默认值）

---

## 10. 验收标准

1. 4 个 capability 共 ~30 类按本规范实现。
2. **≥ 614 个测试通过**（324 现有 + ≥ 290 新增）。
3. 现有 capability 测试仍通过（无回归）。
4. `ruff check` 在所有新文件上无告警。
5. 三层入口（CLI / Web API / MCP）字段一致；输入清洗一致；错误处理一致。
6. 与 `causal_emergence` 子包的集成点至少 5 处：`ChaoticAssociativeMemory` / `DifferentialGenerator` / `CausalInferenceEngine` / `HamiltonianSampler` / `emergence_cycle`。
7. 性能：单次 `memory.retrieve` < 100ms；单次 `planning.plan` < 1s（n_steps ≤ 32）；单次 `rl.train(100 episodes)` < 5s。
8. 审查报告：phase-6 实现审查 + round-2 修复审查 + v3 对账审查（沿用 v3 流程）。

---

## 11. 范围之外

- 神经网络训练 / 反向传播（仍零数据）。
- 外部 RL gym 环境接入（用合成 MDP）。
- 跨进程分布式记忆（单进程）。
- 多 agent 协同。
- GPU 加速。
- 生产级 MCTS（不引入 mctslib）。
- 真实多模态数据集（用合成 embedding）。
- 在线学习 / 增量学习的生产部署（接口提供，但不承诺 SLA）。

---

## 12. 实施顺序

按依赖关系实施：

1. **Memory**（无外部依赖，与 `chaotic_memory` 单向集成）→ 第 1 周
2. **Planning**（依赖 `differential_generator` + `causal_inference_engine`）→ 第 2 周
3. **Multimodal**（依赖 `emergence_cycle`）→ 第 3 周
4. **RL**（依赖 `hmc_sampler` + `differential_generator`）→ 第 4 周
5. **ZeroDataModel facade 集成 + 三层入口扩展**→ 第 5 周
6. **测试套件 + 审查 + 提交**→ 第 6 周

每个 capability 完成后立即跑测试 + ruff，避免堆积。
