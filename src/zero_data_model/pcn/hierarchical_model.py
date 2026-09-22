"""Hierarchical ZeroDataModel — reorganize 6 modules into 3 layers.

架构
====

原 ZeroDataModel 的 6 个核心模块是平级的（consciousness,
active_inference, category_engine, quantum_hybrid, biological, math_universe）。

HierarchicalZeroDataModel 把它们重组为 3 个深度层级：

L0（底层，感知层）：
  - 输入：原始观测（signal.observations）
  - 模块：consciousness（注意力、感知）+ biological（底层数据编码）
  - 输出：L0 隐状态（对观测的内部表示）

L1（中层，短期行为）：
  - 输入：L0 隐状态
  - 模块：active_inference（自由能、动作）+ quantum_hybrid（探索）
  - 输出：L1 隐状态（短期行为模式）

L2（高层，抽象规则）：
  - 输入：L1 隐状态
  - 模块：category_engine（范畴推理）+ math_universe（数学结构）
  - 输出：L2 隐状态（抽象规则）

跨层级元认知：causal_emergence（因果涌现）+ 6 个认知升级模块

数据流
=====
think() 中：
1. 自底向上（误差传播）：
   L0.process(observation) → L0_state → L1.process(L0_state) → L1_state
   → L2.process(L1_state) → L2_state
   每层计算 error = state - prediction_from_higher
   误差逐层上传

2. 自顶向下（预测传播）：
   L2.predict() → L1_prediction → L1.predict() → L0_prediction
   预测逐层下传

3. 更新：
   每层用 PCN 更新规则调整 state 和 W_gen/W_rec
"""
from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

# 惰性导入避免循环依赖
from .pcn_layer import PCNLayer

if TYPE_CHECKING:
    from ..experiment.bayesian_experiment_planner import BayesianExperimentPlannerV2
    from ..experiment.experiment_logger import ExperimentLogger
    from ..knowledge.logic_engine import LogicEngine
    from ..knowledge.reasoning_graph import ReasoningGraph
    from ..metacog.meta_trigger import MetaTrigger
    from ..metacog.second_order_belief import SecondOrderBelief
    from ..observability.audit import AuditLogger
    from ..observability.counterfactual_explainer import CounterfactualExplainer
    from ..observability.human_teaching import HumanFeedback
    from ..observability.thought_chain import ThoughtChain
    from ..tools.safety import SafetyChecker
    from ..tools.tool_policy import ToolPolicy
    from ..tools.tool_registry import ToolRegistry


class HierarchicalZeroDataModel:
    """3-layer hierarchical predictive coding wrapper around ZeroDataModel.

    不直接继承 ZeroDataModel（避免 __init__ 复杂度），而是组合：
    - 持有一个 ZeroDataModel 实例（实际计算）
    - 在外层包裹 PCN 层级

    第二阶段（Phase 2）新增可选的神经符号融合、元认知和主动实验
    模块。启用后，``think()`` 会在 PCN 循环之后运行 Phase 2 循环，
    将逻辑一致性检查、二阶信念更新、元认知触发和实验规划的结果
    写入 ``signal.metadata["cognitive_upgrades"]``，供前端面板消费。

    第三阶段（Phase 3）新增可选的多智能体社会、工具使用和全栈可观测
    性模块。启用后，``think()`` 会在 Phase 2 循环之后运行 Phase 3
    循环，将思维链记录、工具调用决策和审计日志写入
    ``signal.metadata["phase3"]``。

    Parameters
    ----------
    model : ZeroDataModel
        被包装的底层模型
    use_pcn : bool
        是否启用 PCN 层级（False 时退化为普通 ZeroDataModel）
    pcn_lr : float
        PCN 层学习率
    use_phase2 : bool
        是否启用第二阶段模块（逻辑引擎、元认知、实验规划器）。
        False 时退化为纯 PCN 模式（向后兼容）。
    enable_tools : bool
        是否启用工具使用（ToolRegistry + ToolPolicy + SafetyChecker）。
    enable_teaching : bool
        是否启用人类教学接口（HumanFeedback）。
    enable_observability : bool
        是否启用全栈可观测性（ThoughtChain + CounterfactualExplainer
        + AuditLogger）。
    enable_skills : bool
        是否启用技能层（``src/skills/``）。启用后，``think()`` 在
        PCN/Phase 2/Phase 3 循环之后调用所有已注册技能，将结果写入
        ``signal.metadata["skills"]``。``False`` 时零成本（向后兼容）。
    enabled_skills : list[str] | None
        显式指定启用的技能名称子集。``None`` 表示启用所有已注册
        技能。配合 ``enable_skills=True`` 使用。
    enable_jepa : bool
        是否启用 JEPA（联合嵌入预测架构）。启用后，``think()`` 在 PCN
        循环后计算目标隐表示（EMA）与在线预测，将 jepa_error 加权进
        自由能，结果写入 ``signal.metadata["jepa"]``。``False`` 时零成本。
    enable_gwt : bool
        是否启用 GWT（全局工作空间竞争）。启用后，各模块输出广播候选，
        softmax 竞争选择胜者广播，并计算整合信息 Φ，结果写入
        ``signal.metadata["gwt"]``。
    enable_discovery : bool
        是否启用自主科学发现引擎。启用后，每 N 步触发"假设→实验→
        分析→论文"闭环，论文存档到 ``discoveries/``，结果写入
        ``signal.metadata["discovery"]``。
    enable_embodiment : bool
        是否启用具身主动感知闭环（升级 1）。启用后，``think()`` 在
        输入编码阶段调用 EmbodiedBody 主动感知动作（视线移动/触觉
        探测/施力），多通道观测（视觉+触觉+本体感觉）扩展输入维度，
        SensorimotorPredictor 学习感知-运动偶联，结果写入
        ``signal.metadata["embodiment"]``。
    enable_iwsm : bool
        是否启用统一自我模型（升级 2，IWSM）。启用后，``think()`` 在
        GWT 广播后、动作选择前调用 SelfSchema（注意/动作/情感预测）、
        AutobiographicalMemory（自传体叙事）、PhiSelf（自我 Φ）、
        CounterfactualSelf（反事实自我解释），结果写入
        ``signal.metadata["iwsm"]``。
    jepa_lambda : float
        JEPA 误差在总自由能中的权重（0-1），默认 0.5。
    discovery_interval : int
        科学发现循环触发间隔（步数），默认 5000。
    """

    def __init__(
        self,
        model,
        *,
        use_pcn: bool = True,
        pcn_lr: float = 0.01,
        use_phase2: bool = False,
        enable_tools: bool = False,
        enable_teaching: bool = False,
        enable_observability: bool = False,
        enable_skills: bool = False,
        enabled_skills: list[str] | None = None,
        enable_jepa: bool = False,
        enable_gwt: bool = False,
        enable_discovery: bool = False,
        enable_embodiment: bool = False,
        enable_iwsm: bool = False,
        jepa_lambda: float = 0.5,
        discovery_interval: int = 5000,
    ):
        self._model = model
        self._use_pcn = use_pcn
        self._use_phase2 = use_phase2
        self._enable_tools = enable_tools
        self._enable_teaching = enable_teaching
        self._enable_observability = enable_observability
        self._enable_skills = enable_skills
        self._enabled_skill_names: set[str] | None = (
            set(enabled_skills) if enabled_skills is not None else None
        )
        self._enable_jepa = enable_jepa
        self._enable_gwt = enable_gwt
        self._enable_discovery = enable_discovery
        self._enable_embodiment = enable_embodiment
        self._enable_iwsm = enable_iwsm
        self._jepa_lambda = jepa_lambda
        self._discovery_interval = discovery_interval
        self._lock = threading.RLock()
        self._step_count: int = 0

        # Phase 2 组件（惰性初始化，避免循环导入）。
        self._logic_engine: LogicEngine | None = None
        self._reasoning_graph: ReasoningGraph | None = None
        self._second_order_belief: SecondOrderBelief | None = None
        self._meta_trigger: MetaTrigger | None = None
        self._experiment_planner: BayesianExperimentPlannerV2 | None = None
        self._experiment_logger: ExperimentLogger | None = None
        self._phase2_dim: int = 16  # 信念向量→谓词字典的截断维度。

        # Phase 3 组件（惰性初始化）。
        self._tool_registry: ToolRegistry | None = None
        self._tool_policy: ToolPolicy | None = None
        self._safety_checker: SafetyChecker | None = None
        self._human_feedback: HumanFeedback | None = None
        self._thought_chain: ThoughtChain | None = None
        self._counterfactual_explainer: CounterfactualExplainer | None = None
        self._audit_logger: AuditLogger | None = None

        # 技能层组件（SkillRegistry）。
        self._skill_registry: Any = None

        # JEPA / GWT / Discovery 组件（惰性初始化，避免循环导入）。
        self._jepa_module: Any = None
        self._gwt_selector: Any = None
        self._gwt_broadcaster: Any = None
        self._phi_calculator: Any = None
        self._science_loop: Any = None
        # GWT Φ 计算节流：每 10 步重算一次 Φ，其余步用缓存值。
        # compute_phi 涉及 n_modules^2 次高斯互信息估计，是 GWT 循环
        # 的主要开销（实测 ~7.5ms/step）。节流后均摊 <1ms/step。
        # module_states 仍每步 record（廉价），仅 Φ 重算被节流。
        self._gwt_step_count: int = 0
        self._phi_compute_interval: int = 10
        self._last_phi: float = 0.0

        # 具身/IWSM 组件（惰性初始化，避免循环导入）。
        self._embodied_body: Any = None
        self._sensorimotor_predictor: Any = None
        self._proprioceptive_encoder: Any = None
        self._self_schema: Any = None
        self._autobiographical_memory: Any = None
        self._phi_self: Any = None
        self._counterfactual_self: Any = None
        # 主动感知动作选择节流：每 K 步选择一次新感知动作，
        # 其余步复用上次动作（避免每步都执行完整的动作空间评估）。
        self._embodiment_step_count: int = 0
        self._embodiment_action_interval: int = 5
        self._last_embodied_action: int = 0

        if use_phase2:
            self._init_phase2(model.dim)

        if enable_tools or enable_teaching or enable_observability:
            self._init_phase3()

        if enable_skills:
            self._init_skills()

        if enable_jepa:
            self._init_jepa(model.dim)

        if enable_gwt:
            self._init_gwt()

        if enable_discovery:
            self._init_discovery()

        if enable_embodiment:
            self._init_embodiment(model.dim)

        if enable_iwsm:
            self._init_iwsm(model.dim)

        if use_pcn:
            dim = model.dim
            # 3 层 PCN
            # L0: 感知层（dim 维，接收观测）
            # L1: 短期行为层（dim 维，预测 L0）
            # L2: 抽象规则层（dim // 2 维，预测 L1，更小维度更抽象）
            # Floor L2 dim at 1 so dim=1 does not collapse to 0 (same
            # pattern as ZeroDataModel's enable_layered_predictor fix).
            _l2_dim = max(dim // 2, 1)
            self._l0 = PCNLayer(
                dim=dim, lower_dim=None, higher_dim=dim, lr=pcn_lr, seed=42
            )
            self._l1 = PCNLayer(
                dim=dim, lower_dim=dim, higher_dim=_l2_dim, lr=pcn_lr, seed=43
            )
            self._l2 = PCNLayer(
                dim=_l2_dim, lower_dim=dim, higher_dim=None, lr=pcn_lr, seed=44
            )
        else:
            self._l0 = None
            self._l1 = None
            self._l2 = None

    @property
    def model(self):
        return self._model

    @property
    def use_pcn(self) -> bool:
        return self._use_pcn

    def reset(self) -> None:
        """重置所有可变状态，避免 episode 之间的状态泄漏。

        H2 修复：原 ``HierarchicalZeroDataModel`` 没有 ``reset()`` 方法，
        导致以下状态在 episode 之间泄漏：
          - ``_step_count`` / ``_gwt_step_count`` / ``_embodiment_step_count``
          - ``_last_phi`` / ``_last_embodied_action`` / ``_last_visual_encoding``
          - ``_last_iwsm_state``（如果 IWSM 启用）
          - PCN 各层状态（L0/L1/L2）
          - JEPA / GWT / Discovery / Embodiment / IWSM 子模块的可变状态

        军事级健壮性要求：每次新 episode 必须从干净状态开始，否则
        上一个 episode 的隐状态会污染下一个 episode 的预测，使评估
        结果不可重现。
        """
        with self._lock:
            self._step_count = 0
            self._gwt_step_count = 0
            self._embodiment_step_count = 0
            self._last_phi = 0.0
            self._last_embodied_action = 0
            # 清除跨 episode 的视觉编码缓存
            if hasattr(self, "_last_visual_encoding"):
                del self._last_visual_encoding
            if hasattr(self, "_last_iwsm_state"):
                del self._last_iwsm_state

            # 重置 PCN 层
            if self._use_pcn:
                if self._l0 is not None and hasattr(self._l0, "reset"):
                    self._l0.reset()
                if self._l1 is not None and hasattr(self._l1, "reset"):
                    self._l1.reset()
                if self._l2 is not None and hasattr(self._l2, "reset"):
                    self._l2.reset()

            # 重置可选子模块（每个都安全包装，缺失属性跳过）
            for attr in (
                "_jepa_module",
                "_gwt_selector",
                "_gwt_broadcaster",
                "_phi_calculator",
                "_science_loop",
            ):
                mod = getattr(self, attr, None)
                if mod is not None and hasattr(mod, "reset"):
                    try:
                        mod.reset()
                    except Exception:
                        # 重置失败不应阻塞其他模块重置
                        pass

            # 具身模块重置
            if self._embodied_body is not None and hasattr(
                self._embodied_body, "reset"
            ):
                try:
                    self._embodied_body.reset()
                except Exception:
                    pass
            if self._sensorimotor_predictor is not None and hasattr(
                self._sensorimotor_predictor, "reset"
            ):
                try:
                    self._sensorimotor_predictor.reset()
                except Exception:
                    pass
            if self._proprioceptive_encoder is not None and hasattr(
                self._proprioceptive_encoder, "reset"
            ):
                try:
                    self._proprioceptive_encoder.reset()
                except Exception:
                    pass

            # IWSM 模块重置
            for attr in (
                "_self_schema",
                "_autobiographical_memory",
                "_phi_self",
                "_counterfactual_self",
            ):
                mod = getattr(self, attr, None)
                if mod is not None and hasattr(mod, "reset"):
                    try:
                        mod.reset()
                    except Exception:
                        pass

            # Phase 2 / Phase 3 模块重置（如有 reset 方法）
            for attr in (
                "_second_order_belief",
                "_meta_trigger",
                "_experiment_planner",
                "_experiment_logger",
                "_logic_engine",
                "_reasoning_graph",
                "_tool_registry",
                "_tool_policy",
                "_safety_checker",
                "_human_feedback",
                "_thought_chain",
                "_counterfactual_explainer",
                "_audit_logger",
            ):
                mod = getattr(self, attr, None)
                if mod is not None and hasattr(mod, "reset"):
                    try:
                        mod.reset()
                    except Exception:
                        pass

    @property
    def layers(self) -> dict:
        if not self._use_pcn:
            return {}
        with self._lock:
            return {"L0": self._l0, "L1": self._l1, "L2": self._l2}

    @property
    def belief_state(self):
        """委托到底层模型的 active_inference 隐状态。

        ZeroDataModel 本身没有 ``belief_state`` 属性，它位于
        ``active_inference.generative_model.belief_state``。暴露此属性使
        HierarchicalZeroDataModel 与 ZeroDataModel 的常用接口保持一致。
        """
        return self._model.active_inference.generative_model.belief_state

    def think(self, input_data=None, **kwargs):
        """层级化 think()。

        流程：
        1. 调用底层 ZeroDataModel.think() 获取 signal
        2. 自底向上传递观测到 PCN 层
        3. 自顶向下计算预测
        4. 用 PCN 更新规则调整各层状态
        5. 把 PCN 信息附加到 signal.metadata

        若 ``input_data`` 是多维帧（如 128×128 物理沙盒画面），
        会保留原始帧作为技能层的 ``raw_observation``，同时把展平后
        的 1D 向量喂给底层模型（底层模型期望 ``dim`` 长度的 1D 输入）。
        """
        # 捕获原始输入供技能层消费：多维帧（物理沙盒画面/触觉矩阵等）
        # 不能直接喂给期望 1D 向量的底层模型，需先展平。
        raw_input = input_data
        model_input = input_data
        if (
            isinstance(input_data, np.ndarray)
            and input_data.ndim > 1
        ):
            model_input = input_data.reshape(-1)

        # 1. 底层 think
        signal = self._model.think(model_input, **kwargs)

        if not self._use_pcn:
            return signal

        with self._lock:
            # C1 修复：所有后续循环（JEPA/GWT/Skills/Discovery）必须在
            # 锁内执行，确保读取 self._l0.state / self._l1.state 等
            # 共享状态时获得一致快照，避免多线程竞争导致的数据撕裂。
            # H1 修复：_step_count 在 think() 入口处递增一次，禁止在
            # 子循环中重复递增，避免 phase2+skills 同时启用时双递增。
            self._step_count += 1

            # 2. 提取观测（L0 输入）
            # signal.observations 是 numpy array
            obs = getattr(signal, "observations", None)
            if obs is None:
                # fallback: 用 belief_state
                obs = getattr(self._model, "belief_state", None)
            if obs is None:
                # final fallback: 用 signal.data（集成后的信念向量）
                obs = getattr(signal, "data", None)
            if obs is None:
                # 无法提取观测，跳过 PCN
                return signal

            obs = np.asarray(obs, dtype=np.float64).flatten()[: self._l0.dim]
            # NaN 防护（与 run_pcn_cycle 一致）
            if not np.all(np.isfinite(obs)):
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            # padding 如果 obs 比 dim 短
            if obs.shape[0] < self._l0.dim:
                padded = np.zeros(self._l0.dim)
                padded[: obs.shape[0]] = obs
                obs = padded

            # 2b. 具身主动感知循环（升级 1）。
            #     在输入编码阶段调用 EmbodiedBody 主动感知动作，
            #     多通道观测（视觉+触觉+本体感觉）扩展观测维度。
            #     扩展 belief 写回 obs（截断到 L0.dim，本体感觉编码
            #     作为额外通道拼接，超出部分被 L0.dim 截断）。
            if self._enable_embodiment and self._embodied_body is not None:
                extended_obs = self._run_embodiment_cycle(
                    signal=signal, belief=obs, raw_obs=raw_input
                )
                if isinstance(extended_obs, np.ndarray) and extended_obs.size >= obs.size:
                    # E1 修复：原代码 `obs = extended_obs[: self._l0.dim]`
                    # 直接截断到 L0.dim，但 extended_obs = [belief, prop_enc]，
                    # 截断后只剩 belief，本体感觉编码被完全丢弃——使具身
                    # 模块对 L0 输入毫无贡献，违背"具身主动感知扩展观测"
                    # 的设计目标。改为把本体感觉编码 element-wise 加到
                    # belief 前 prop_len 维，让两种信号都进入 L0。
                    obs = extended_obs[: self._l0.dim].astype(np.float64).copy()
                    prop_part = extended_obs[self._l0.dim:]
                    prop_len = min(prop_part.size, obs.size)
                    if prop_len > 0:
                        obs[:prop_len] += prop_part[:prop_len].astype(
                            np.float64
                        )
                    if not np.all(np.isfinite(obs)):
                        obs = np.nan_to_num(
                            obs, nan=0.0, posinf=0.0, neginf=0.0
                        )

            # 3. 自底向上 + 自顶向下
            # L0 接收观测
            self._l0.set_state(obs)

            # L2 先 predict（自顶向下）
            l2_pred_l1 = self._l2.predict()  # L2 对 L1 的预测

            # L1 predict（基于当前 state）
            l1_pred_l0 = self._l1.predict()  # L1 对 L0 的预测

            # L0 update（接收 L1 的预测，无 lower）
            error_l0 = self._l0.update(
                error_from_lower=None, prediction_from_higher=l1_pred_l0
            )

            # L1 update（接收 L0 的误差和 L2 的预测）
            error_l1 = self._l1.update(
                error_from_lower=error_l0, prediction_from_higher=l2_pred_l1
            )

            # L2 update（接收 L1 的误差，无 higher）
            error_l2 = self._l2.update(
                error_from_lower=error_l1, prediction_from_higher=None
            )

            # 4. 附加 PCN 信息到 metadata
            if not hasattr(signal, "metadata") or signal.metadata is None:
                try:
                    signal.metadata = {}
                except (AttributeError, TypeError):
                    return signal

            with contextlib.suppress(TypeError, KeyError):
                signal.metadata["pcn"] = {
                    "layer_errors": {
                        "L0": float(np.linalg.norm(error_l0)),
                        "L1": float(np.linalg.norm(error_l1)),
                        "L2": float(np.linalg.norm(error_l2)),
                    },
                    "l0_state_norm": float(np.linalg.norm(self._l0.state)),
                    "l1_state_norm": float(np.linalg.norm(self._l1.state)),
                    "l2_state_norm": float(np.linalg.norm(self._l2.state)),
                    "error_decreasing": (
                        float(np.linalg.norm(error_l0))
                        > float(np.linalg.norm(error_l1))
                    ),
                }

            # 5. Phase 2 循环（逻辑一致性 + 元认知 + 实验规划）。
            #    在 PCN 循环完成后运行，使用 PCN 层级误差作为预测误差
            #    和参数更新幅度的代理。memory_similarity 从已有的
            #    cognitive_upgrades 中提取（Phase G Hopfield 检索）。
            if self._use_phase2:
                _belief = self._l0.state.copy()
                _pe = float(np.linalg.norm(error_l0))
                _param_norm = float(np.linalg.norm(error_l1))
                _mem_sim = 1.0
                _existing_upgrades = signal.metadata.get(
                    "cognitive_upgrades", {}
                )
                if isinstance(_existing_upgrades, dict):
                    _mem = _existing_upgrades.get("memory_retrieved", {})
                    if isinstance(_mem, dict):
                        _sim = _mem.get("top1_similarity")
                        if isinstance(_sim, (int, float)):
                            _mem_sim = float(_sim)
                self._run_phase2_cycle(
                    signal=signal,
                    belief=_belief,
                    prediction_error=_pe,
                    param_update_norm=_param_norm,
                    memory_similarity=_mem_sim,
                )

            # 6. Phase 3 循环（工具使用 + 可观测性 + 人类教学）。
            #    在 Phase 2 循环之后运行，使用 PCN 层级误差和 Phase 2
            #    元认知置信度驱动工具调用决策。
            if (
                self._enable_tools
                or self._enable_teaching
                or self._enable_observability
            ):
                _belief_p3 = (
                    self._l0.state.copy() if self._l0 is not None else np.array([])
                )
                _pe_p3 = (
                    float(np.linalg.norm(error_l0))
                    if self._l0 is not None
                    else 0.0
                )
                # 从 Phase 2 元认知结果提取置信度（0-100 → 0-1）。
                _conf = 1.0
                _upgrades = signal.metadata.get("cognitive_upgrades", {})
                if isinstance(_upgrades, dict):
                    _meta = _upgrades.get("meta_cognition", {})
                    if isinstance(_meta, dict):
                        _raw_conf = _meta.get("confidence", 100.0)
                        _conf = float(_raw_conf) / 100.0
                _meta_triggered = (
                    _upgrades.get("meta_cognition", {}).get("triggered", False)
                    if isinstance(_upgrades, dict)
                    else False
                )
                self._run_phase3_cycle(
                    signal=signal,
                    belief=_belief_p3,
                    prediction_error=_pe_p3,
                    confidence=_conf,
                    meta_triggered=bool(_meta_triggered),
                    prediction_errors=(
                        {
                            "L0": float(np.linalg.norm(error_l0)),
                            "L1": float(np.linalg.norm(error_l1)),
                            "L2": float(np.linalg.norm(error_l2)),
                        }
                        if self._l0 is not None
                        else {}
                    ),
                )

            # 7. JEPA 循环（联合嵌入预测架构）。
            #    C1 修复：移入锁内，确保读取 self._l0.state 时获得
            #    与 PCN 循环一致的快照。
            if self._enable_jepa and self._jepa_module is not None:
                _jepa_obs = raw_input if raw_input is not None else (
                    self._l0.state.copy() if self._l0 is not None else np.zeros(self._model.dim)
                )
                _jepa_latent = (
                    self._l0.state.copy() if self._l0 is not None else np.zeros(self._model.dim)
                )
                # JEPA latent 维度对齐：截断/padding 到 jepa_module.latent_dim
                _jl = self._jepa_module.latent_dim
                if _jepa_latent.size > _jl:
                    _jepa_latent = _jepa_latent[:_jl]
                elif _jepa_latent.size < _jl:
                    _jepa_latent = np.pad(_jepa_latent, (0, _jl - _jepa_latent.size))
                self._run_jepa_cycle(signal, _jepa_obs, _jepa_latent)

            # 8. GWT 循环（全局工作空间竞争）。
            #    C1 修复：移入锁内，确保读取 PCN 各层状态时获得一致快照。
            if self._enable_gwt and self._gwt_selector is not None:
                _gwt_belief = (
                    self._l0.state.copy() if self._l0 is not None else np.zeros(self._model.dim)
                )
                _gwt_errors = signal.metadata.get("pcn", {}).get("layer_errors", {})
                if not isinstance(_gwt_errors, dict):
                    _gwt_errors = {}
                self._run_gwt_cycle(signal, _gwt_belief, _gwt_errors)

            # 8b. 统一自我模型 IWSM 循环（升级 2）。
            #     在 GWT 广播后、动作选择前调用，提供自我预测。
            #     SelfSchema 更新 + PhiSelf 计算 + AutobiographicalMemory
            #     存储 + CounterfactualSelf 转移记录。
            if self._enable_iwsm and self._self_schema is not None:
                _iwsm_belief = (
                    self._l0.state.copy() if self._l0 is not None else np.zeros(self._model.dim)
                )
                _iwsm_errors = signal.metadata.get("pcn", {}).get("layer_errors", {})
                if not isinstance(_iwsm_errors, dict):
                    _iwsm_errors = {}
                self._run_iwsm_cycle(signal, _iwsm_belief, _iwsm_errors)

            # 9. 技能层循环（感知/认知/交互/专业/元技能）。
            #    C1 修复：移入锁内，确保读取 PCN 状态与元认知结果一致。
            if self._enable_skills and self._skill_registry is not None:
                self._run_skills_cycle(signal, raw_input=raw_input)

            # 10. 科学发现循环（假设→实验→分析→论文）。
            #     C1 修复：移入锁内，确保读取 layer_errors 一致。
            if self._enable_discovery and self._science_loop is not None:
                _disc_errors = signal.metadata.get("pcn", {}).get("layer_errors", {})
                if not isinstance(_disc_errors, dict):
                    _disc_errors = {}
                self._run_discovery_cycle(signal, _disc_errors)

        return signal

    # ------------------------------------------------------------------ #
    # 技能层集成（src/skills/）
    # ------------------------------------------------------------------ #
    def _init_skills(self) -> None:
        """惰性初始化技能注册中心，注册全部 20 个技能。

        通过 ``enabled_skills`` 参数可在创建时白名单过滤。
        """
        from skills.base import SkillContext  # noqa: F401 (re-export hint)
        from skills.perception import (
            DepthEstimator,
            OlfactionEncoder,
            TactileEncoder,
            AudioSceneEncoder,
        )
        from skills.cognition import (
            AffectModule,
            AnalogyEngine,
            NarrativeModule,
            TheoryOfMindModule,
        )
        from skills.interaction import (
            DialogueAgent,
            DemonstrationLearner,
            MultiModalTranslator,
            TeachingModule,
        )
        from skills.expertise import (
            AnomalyDetector,
            GamePlayer,
            ProgramSynthesizer,
            TheoremProver,
        )
        from skills.meta import (
            CurriculumScheduler,
            EnergyMonitor,
            MemoryDecay,
            MetaLearner,
        )
        from skills.registry import SkillRegistry

        registry = SkillRegistry()

        # 按维度注册全部技能
        all_skills = [
            DepthEstimator(),
            TactileEncoder(),
            AudioSceneEncoder(),
            OlfactionEncoder(),
            TheoryOfMindModule(),
            NarrativeModule(),
            AffectModule(),
            AnalogyEngine(),
            DialogueAgent(),
            DemonstrationLearner(),
            TeachingModule(),
            MultiModalTranslator(),
            ProgramSynthesizer(),
            TheoremProver(),
            GamePlayer(),
            AnomalyDetector(),
            MetaLearner(),
            CurriculumScheduler(),
            MemoryDecay(),
            EnergyMonitor(),
        ]
        for skill in all_skills:
            # 应用白名单
            if (
                self._enabled_skill_names is not None
                and skill.name not in self._enabled_skill_names
            ):
                skill.enabled = False
            registry.register(skill)
        self._skill_registry = registry

    def _run_skills_cycle(self, signal: Any, raw_input: Any = None) -> None:
        """运行技能层循环，将结果写入 signal.metadata['skills']。

        Parameters
        ----------
        raw_input
            ``think()`` 的原始输入。若是多维帧（物理沙盒画面等），
            会作为 ``raw_observation`` 传给技能，使深度估计、触觉、
            嗅觉、多模态翻译等感知技能能消费原始帧。
        """
        from skills.base import SkillContext

        # 构造 SkillContext：从 PCN 状态 + signal 读取
        if self._l0 is not None:
            belief = self._l0.state.copy()
            prediction_errors = signal.metadata.get("pcn", {}).get(
                "layer_errors", {}
            )
            pe = float(prediction_errors.get("L0", 0.0))
        else:
            belief = np.zeros(32)
            prediction_errors = {}
            pe = 0.0

        # 置信度：从 Phase 2 元认知提取（0-100 → 0-1）
        conf = 1.0
        upgrades = signal.metadata.get("cognitive_upgrades", {})
        if isinstance(upgrades, dict):
            meta = upgrades.get("meta_cognition", {})
            if isinstance(meta, dict):
                raw = meta.get("confidence", 100.0)
                conf = float(raw) / 100.0

        # 原始观测优先用 think() 的原始输入（保留多维帧），
        # 否则回退到 signal.data（1D 信念向量）。
        raw_obs = raw_input if raw_input is not None else getattr(
            signal, "observations", None
        )

        ctx = SkillContext(
            belief=belief,
            prediction_error=pe,
            prediction_errors=dict(prediction_errors),
            confidence=conf,
            step=self._step_count,
            model=self._model,
            raw_observation=raw_obs,
        )

        try:
            results = self._skill_registry.process_all(ctx)
        except Exception:  # noqa: BLE001
            # 技能层失败不应影响 think() 返回
            results = {"_error": "skill registry failed"}

        if not hasattr(signal, "metadata") or signal.metadata is None:
            try:
                signal.metadata = {}
            except (AttributeError, TypeError):
                return
        signal.metadata["skills"] = results
        # H1 修复：_step_count 已在 think() 入口处递增，此处不再重复。

    def skills_snapshot(self) -> dict[str, Any]:
        """返回所有技能的快照（供前端技能面板）。

        若技能层未启用，返回空 dict。
        """
        if self._skill_registry is None:
            return {"enabled": False, "count": 0, "skills": []}
        snap = self._skill_registry.snapshot()
        snap["enabled"] = True
        return snap

    def enable_skill(self, name: str) -> bool:
        """启用指定技能。返回是否找到该技能。"""
        if self._skill_registry is None:
            return False
        return self._skill_registry.enable(name)

    def disable_skill(self, name: str) -> bool:
        """禁用指定技能。返回是否找到该技能。"""
        if self._skill_registry is None:
            return False
        return self._skill_registry.disable(name)

    # ------------------------------------------------------------------ #
    # JEPA / GWT / Discovery 初始化与运行
    # ------------------------------------------------------------------ #
    def _init_jepa(self, dim: int) -> None:
        """惰性初始化 JEPA 模块。"""
        from jepa.jepa_module import JEPAModule

        # JEPA 输入维度对齐模型 dim，隐空间维度取 dim//2（至少 8）
        latent_dim = max(dim // 2, 8)
        self._jepa_module = JEPAModule(
            input_dim=dim,
            latent_dim=latent_dim,
            lambda_jepa=self._jepa_lambda,
            seed=42,
        )

    def _init_gwt(self) -> None:
        """惰性初始化 GWT 组件。"""
        from gwt.attention_selector import AttentionSelector
        from gwt.workspace_broadcaster import WorkspaceBroadcaster
        from gwt.phi_calculator import PhiCalculator

        dim = self._model.dim
        self._gwt_selector = AttentionSelector(dim=dim, seed=42)
        self._gwt_broadcaster = WorkspaceBroadcaster(self._gwt_selector)
        # H3 修复：从 6 降到 4，匹配实际参与意识整合的模块数
        # （L0/L1/L2 + consciousness/belief）。原 n_modules=6 会用
        # belief 的相同副本填充，导致互信息退化为自信息，Φ 虚高。
        self._phi_calculator = PhiCalculator(
            n_modules=4, history_window=50
        )

    def _init_discovery(self) -> None:
        """惰性初始化科学发现引擎。"""
        from discovery.science_loop import ScienceLoop

        self._science_loop = ScienceLoop(
            trigger_interval=self._discovery_interval,
            output_dir="discoveries",
            require_approval=False,
            seed=42,
        )

    def _init_embodiment(self, dim: int) -> None:
        """惰性初始化具身主动感知模块（升级 1）。

        Parameters
        ----------
        dim : int
            模型隐空间维度，用于 SensorimotorPredictor 与 ProprioceptiveEncoder
            输出维度对齐。
        """
        from embodied.body_env import EmbodiedBody
        from embodied.sensorimotor_predictor import SensorimotorPredictor
        from embodied.proprioception import ProprioceptiveEncoder

        self._embodied_body = EmbodiedBody(
            view_size=max(dim, 16),
            tactile_grid=4,
            gaze_step=0.3,
            num_objects=2,
            seed=42,
        )
        self._sensorimotor_predictor = SensorimotorPredictor(
            dim=dim,
            n_probe=16,
            lr=0.01,
            seed=43,
        )
        self._proprioceptive_encoder = ProprioceptiveEncoder(
            input_dim=8,
            output_dim=dim,
            lr=0.01,
            seed=44,
        )

    def _init_iwsm(self, dim: int) -> None:
        """惰性初始化统一自我模型 IWSM（升级 2）。

        Parameters
        ----------
        dim : int
            模型隐空间维度，用于 SelfSchema / CounterfactualSelf 输出对齐。
        """
        from self.self_schema import SelfSchema
        from self.autobiographical_memory import AutobiographicalMemory
        from self.phi_self import PhiSelf
        from self.counterfactual_self import CounterfactualSelf

        # 推断动作空间大小：若启用具身，则用 EmbodiedBody.action_space_size；
        # 否则默认 8（与 active_inference 默认一致）。
        n_actions = 8
        if self._embodied_body is not None:
            try:
                n_actions = int(self._embodied_body.action_space_size)
            except (AttributeError, TypeError):
                n_actions = 8

        self._self_schema = SelfSchema(
            dim=dim,
            n_modules=4,
            n_actions=n_actions,
            affect_dim=4,
            lr=0.01,
            seed=42,
        )
        self._autobiographical_memory = AutobiographicalMemory(
            dim=dim,
            capacity=512,
            seed=43,
        )
        self._phi_self = PhiSelf(
            n_self_modules=4,
            history_window=50,
            sleep_threshold=0.1,
        )
        self._counterfactual_self = CounterfactualSelf(
            dim=dim,
            n_actions=n_actions,
            lr=0.01,
            seed=44,
        )

    def _run_jepa_cycle(self, signal: Any, raw_obs: Any, latent_state: np.ndarray) -> None:
        """运行 JEPA 循环：目标编码 → 在线预测 → 误差驱动 → 自由能合并。"""
        if self._jepa_module is None:
            return
        try:
            obs = np.asarray(raw_obs, dtype=np.float64).flatten()
            if obs.size == 0:
                obs = latent_state
            # M11 修复：NaN/inf 防护。原代码不清理 obs 中的 NaN，
            # 导致 NaN 通过 target_encoder.encode 传播到 jepa_error，
            # 进而污染 combine_free_energy。与 run_pcn_cycle 一致。
            if not np.all(np.isfinite(obs)):
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.all(np.isfinite(latent_state)):
                latent_state = np.nan_to_num(latent_state, nan=0.0, posinf=0.0, neginf=0.0)
            result = self._jepa_module.process(obs, latent_state)
            if not hasattr(signal, "metadata") or signal.metadata is None:
                signal.metadata = {}
            signal.metadata["jepa"] = result
            # 合并自由能：将 jepa_error 加权进 prediction_error
            jepa_err = result.get("jepa_error", 0.0)
            if isinstance(jepa_err, (int, float)):
                # 即使 jepa_error=0（完美预测）也计算合并自由能，
                # 此时 F_total = (1-λ)*F_pixel + λ*0 = (1-λ)*F_pixel
                combined = self._jepa_module.combine_free_energy(
                    base_prediction_error=float(
                        signal.metadata.get("pcn", {}).get("layer_errors", {}).get("L0", 0.0)
                    ),
                    jepa_error=float(jepa_err),
                )
                signal.metadata["jepa"]["combined_free_energy"] = round(
                    combined, 6
                )
        except Exception as exc:
            # 军事级健壮性：JEPA 失败不应中断 think()
            if not hasattr(signal, "metadata") or signal.metadata is None:
                signal.metadata = {}
            signal.metadata["jepa"] = {"error": f"{type(exc).__name__}: {exc}"}

    def _run_gwt_cycle(self, signal: Any, belief: np.ndarray, prediction_errors: dict) -> None:
        """运行 GWT 循环：构建候选 → 竞争广播 → Φ 计算。"""
        if self._gwt_selector is None:
            return
        try:
            from gwt.attention_selector import BroadcastCandidate

            # H2 修复：候选向量必须使用各 PCN 层的真实隐状态，
            # 而非 belief 的同一副本。原代码所有候选共享同一向量，
            # 导致无论哪个模块胜出，广播内容都相同——竞争失去意义。
            # 现在按层名映射到对应的 PCN 层状态，使竞争真正反映
            # 不同模块的信息内容。
            layer_states: dict[str, np.ndarray] = {}
            if self._l0 is not None:
                layer_states["L0"] = self._l0.state.copy()
            if self._l1 is not None:
                layer_states["L1"] = self._l1.state.copy()
            if self._l2 is not None:
                layer_states["L2"] = self._l2.state.copy()

            # 构建广播候选：每个 PCN 层 + 意识核心
            candidates: list[BroadcastCandidate] = []
            for layer_name, err in prediction_errors.items():
                err_f = float(err) if isinstance(err, (int, float)) else 0.0
                # F2 修复：err_f 可能为 NaN/负值/极大值，导致 confidence
                # 超出 [0,1] 或为 NaN，使 softmax 竞争失真。
                # 裁剪到 [0,1] 并防御非有限值。
                if not np.isfinite(err_f):
                    err_f = 0.0
                # 使用对应层的真实状态，而非 belief 副本
                vec = layer_states.get(layer_name, belief.copy())
                confidence = max(0.0, min(1.0, 1.0 - err_f))
                candidates.append(
                    BroadcastCandidate(
                        module_name=f"pcn_{layer_name}",
                        vector=vec,
                        attention_request=err_f,
                        confidence=confidence,
                    )
                )
            # 意识核心候选（belief 本身）
            candidates.append(
                BroadcastCandidate(
                    module_name="consciousness",
                    vector=belief,
                    attention_request=float(np.linalg.norm(belief)),
                    confidence=1.0,
                )
            )

            result = self._gwt_broadcaster.broadcast(candidates)

            # Φ 计算：每步记录模块状态（廉价），但 Φ 重算节流。
            # compute_phi 涉及 n_modules^2 高斯互信息估计（~7.5ms），
            # 节流到每 10 步重算一次，其余步复用缓存值，均摊 <1ms/step。
            # H3 修复：使用各 PCN 层的不同隐状态（L0/L1/L2 + belief），
            # 共 4 个真实模块。原代码用 belief.copy() 填充到 6 个，
            # 相同副本会让互信息退化为自信息，导致 Φ 虚高。
            # 现在直接使用 4 个不同状态，PhiCalculator(n_modules=4)。
            module_states: list[np.ndarray] = []
            if self._l0 is not None:
                module_states.append(self._l0.state.copy())
            if self._l1 is not None:
                module_states.append(self._l1.state.copy())
            if self._l2 is not None:
                module_states.append(self._l2.state.copy())
            module_states.append(belief.copy())
            # H3 修复：不再用 belief.copy() 填充到 6 个。
            # PhiCalculator 在 _init_gwt 中已初始化为 n_modules=4，
            # record() 会自动处理长度匹配。
            self._phi_calculator.record(module_states)
            self._gwt_step_count += 1
            if self._gwt_step_count % self._phi_compute_interval == 0:
                self._last_phi = self._phi_calculator.compute_phi()
            phi = self._last_phi

            if not hasattr(signal, "metadata") or signal.metadata is None:
                signal.metadata = {}
            signal.metadata["gwt"] = {
                "winner": result.winner.module_name if result.winner else None,
                "probabilities": result.probabilities,
                "broadcast_confidence": round(result.broadcast_confidence, 6),
                "global_timestamp": result.timestamp,
                "phi": round(phi, 6),
                "is_conscious": self._phi_calculator.is_conscious(),
                "phi_history": [round(p, 6) for p in self._phi_calculator.phi_history[-20:]],
                "selector": self._gwt_selector.snapshot(),
                "broadcaster": self._gwt_broadcaster.snapshot(),
            }
        except Exception as exc:
            # 军事级健壮性：GWT 失败不应中断 think()，记录错误供前端诊断
            if not hasattr(signal, "metadata") or signal.metadata is None:
                signal.metadata = {}
            signal.metadata["gwt"] = {"error": f"{type(exc).__name__}: {exc}"}

    def _run_discovery_cycle(self, signal: Any, prediction_errors: dict) -> None:
        """运行科学发现循环（按间隔触发）。"""
        if self._science_loop is None:
            return
        try:
            # 提取因果图与知识图谱（从 signal.metadata）
            causal_graph = signal.metadata.get("causal_graph", {})
            if not isinstance(causal_graph, dict):
                causal_graph = {}
            knowledge_graph = signal.metadata.get("kg_update", {})
            if not isinstance(knowledge_graph, dict):
                knowledge_graph = {}

            # 逻辑矛盾（从 Phase 2）
            upgrades = signal.metadata.get("cognitive_upgrades", {})
            contradictions: list[dict] = []
            if isinstance(upgrades, dict):
                lv = upgrades.get("logic_violations", {})
                if isinstance(lv, dict):
                    contradictions = lv.get("violations", []) or []

            cycle = self._science_loop.step(
                causal_graph=causal_graph or None,
                knowledge_graph=knowledge_graph or None,
                prediction_errors=prediction_errors,
                logic_contradictions=contradictions or None,
            )

            if not hasattr(signal, "metadata") or signal.metadata is None:
                signal.metadata = {}
            signal.metadata["discovery"] = self._science_loop.snapshot()
            if cycle is not None:
                signal.metadata["discovery"]["last_cycle"] = {
                    "cycle_id": cycle.cycle_id,
                    "n_hypotheses": cycle.n_hypotheses,
                    "n_experiments": cycle.n_experiments,
                    "n_accepted": cycle.n_accepted,
                    "n_rejected": cycle.n_rejected,
                    "papers_written": cycle.papers_written,
                    "findings": cycle.findings,
                    "error": cycle.error,
                }
        except Exception as exc:
            # 军事级健壮性：科学发现失败不应中断 think()
            if not hasattr(signal, "metadata") or signal.metadata is None:
                signal.metadata = {}
            signal.metadata["discovery"] = {"error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------ #
    # 具身主动感知循环（升级 1）
    # ------------------------------------------------------------------ #
    def _run_embodiment_cycle(
        self,
        signal: Any,
        belief: np.ndarray,
        raw_obs: Any,
    ) -> np.ndarray:
        """运行具身主动感知循环。

        流程：
        1. 选择主动感知动作（节流：每 K 步重新选择）
        2. EmbodiedBody.step(action) 获取多通道观测
        3. ProprioceptiveEncoder 编码本体感觉
        4. SensorimotorPredictor.update 预测感知-运动偶联
        5. 将本体感觉编码拼接到 belief（扩展观测维度）
        6. 写入 signal.metadata["embodiment"]

        Parameters
        ----------
        signal : Any
            当前 signal（用于写入 metadata）
        belief : ndarray
            当前 L0 信念状态（用于动作选择）
        raw_obs : Any
            原始观测（用于计算视觉变化）

        Returns
        -------
        ndarray
            扩展后的 belief（含本体感觉编码拼接），形状 (dim + prop_dim,)
            若 EmbodiedBody 未初始化则原样返回 belief。
        """
        if (
            self._embodied_body is None
            or self._sensorimotor_predictor is None
            or self._proprioceptive_encoder is None
        ):
            return belief

        try:
            # 1. 主动感知动作选择（节流）
            self._embodiment_step_count += 1
            belief_for_select = (
                belief.copy()
                if isinstance(belief, np.ndarray)
                else np.zeros(self._model.dim)
            )
            if (
                self._embodiment_step_count % self._embodiment_action_interval
                == 1
                or self._last_embodied_action == 0
            ):
                # 在主动感知动作空间中选（索引 4..N_TOTAL_ACTIONS-1）
                # 用 sensorimotor_predictor 的 select_perceptual_action
                n_perc = self._embodied_body.action_space_size
                self._last_embodied_action = (
                    self._sensorimotor_predictor.select_perceptual_action(
                        belief_for_select,
                        n_actions=n_perc,
                    )
                )
            action = int(self._last_embodied_action)

            # 2. 身体执行动作
            obs = self._embodied_body.step(action)

            # 3. 本体感觉编码
            prop_state = self._proprioceptive_encoder.step(obs.proprioception)

            # 4. 感知-运动预测器更新
            #    视觉变化 = 当前帧与上一帧之差（展平到 dim）
            vis_frame_flat = np.asarray(obs.visual_frame, dtype=np.float64).flatten()
            vis_dim = self._sensorimotor_predictor.dim
            if vis_frame_flat.size >= vis_dim:
                cur_vis = vis_frame_flat[:vis_dim]
            else:
                padded = np.zeros(vis_dim)
                padded[: vis_frame_flat.size] = vis_frame_flat
                cur_vis = padded
            # 归一化到 [-1, 1]
            if cur_vis.max() > 1.0:
                cur_vis = cur_vis / 255.0
            # 上一时刻视觉（缓存于 metadata）
            prev_vis = getattr(self, "_last_visual_encoding", None)
            if prev_vis is None:
                visual_change = np.zeros(vis_dim)
            else:
                visual_change = cur_vis - prev_vis
            self._last_visual_encoding = cur_vis.copy()

            # 触觉读数
            tactile = np.asarray(obs.tactile, dtype=np.float64).flatten()

            # 更新预测器
            pred = self._sensorimotor_predictor.update(
                belief=belief_for_select
                if isinstance(belief, np.ndarray)
                else np.zeros(self._model.dim),
                action_id=action,
                actual_visual_change=visual_change,
                actual_tactile=tactile,
            )

            # 5. 扩展 belief：拼接本体感觉编码
            extended_belief = np.concatenate(
                [np.asarray(belief).flatten(), prop_state.encoded]
            )

            # 6. 写入 metadata
            if not hasattr(signal, "metadata") or signal.metadata is None:
                signal.metadata = {}
            signal.metadata["embodiment"] = {
                "action": action,
                "action_name": self._embodied_body.parse_action(action).action_type,
                "gaze_yaw": float(obs.gaze_yaw),
                "gaze_pitch": float(obs.gaze_pitch),
                "touched_object": int(obs.touched_object_id),
                "proprioception_error": float(prop_state.prediction_error),
                "body_schema_confidence": float(prop_state.body_schema_confidence),
                "visual_prediction_error": float(pred.visual_error),
                "tactile_prediction_error": float(pred.tactile_error),
                "sensorimotor": self._sensorimotor_predictor.get_snapshot(),
                "body_schema": self._proprioceptive_encoder.get_body_schema_snapshot(),
                "body_snapshot": self._embodied_body.get_snapshot()
                if hasattr(self._embodied_body, "get_snapshot")
                else {},
            }
            return extended_belief
        except Exception as exc:
            # 军事级健壮性：具身模块失败不应中断 think()
            if not hasattr(signal, "metadata") or signal.metadata is None:
                signal.metadata = {}
            signal.metadata["embodiment"] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
            return belief

    # ------------------------------------------------------------------ #
    # 统一自我模型 IWSM 循环（升级 2）
    # ------------------------------------------------------------------ #
    def _run_iwsm_cycle(
        self,
        signal: Any,
        belief: np.ndarray,
        prediction_errors: dict,
    ) -> None:
        """运行统一自我模型循环。

        在 GWT 广播后、动作选择前调用，提供自我预测。

        流程：
        1. 构建整合状态（GWT winner 或 L0 state）
        2. 推断实际注意力（GWT winner one-hot）
        3. 推断实际动作（上一时刻动作，从 embodiment/active_inference 获取）
        4. 推断实际情感（自由能分布）
        5. SelfSchema.update(integrated, attention, action, affect)
        6. PhiSelf.record(self_module_states) + compute_phi
        7. AutobiographicalMemory.store(state_snapshot, prediction, outcome, fe, step)
        8. CounterfactualSelf.record_transition(state_before, action, state_after)
        9. 偶尔（每 K 步）触发反事实叙述
        10. 写入 signal.metadata["iwsm"]
        """
        if (
            self._self_schema is None
            or self._autobiographical_memory is None
            or self._phi_self is None
            or self._counterfactual_self is None
        ):
            return

        try:
            # 1. 整合状态 = GWT 广播后的状态（如有），否则 L0 state
            gwt_md = signal.metadata.get("gwt", {}) if hasattr(signal, "metadata") else {}
            if not isinstance(gwt_md, dict):
                gwt_md = {}
            winner = gwt_md.get("winner")

            integrated = (
                np.asarray(belief, dtype=np.float64).flatten()
                if isinstance(belief, np.ndarray)
                else np.zeros(self._model.dim)
            )
            # padding/truncate 到 self_schema.dim
            iwsm_dim = self._self_schema.dim
            if integrated.size > iwsm_dim:
                integrated = integrated[:iwsm_dim]
            elif integrated.size < iwsm_dim:
                integrated = np.pad(integrated, (0, iwsm_dim - integrated.size))

            # 2. 实际注意力：GWT winner → one-hot(n_modules)
            n_modules = self._self_schema.n_modules
            module_names = ["L0", "L1", "L2", "consciousness"]
            actual_attn = np.zeros(n_modules)
            if winner and isinstance(winner, str):
                for i, name in enumerate(module_names[:n_modules]):
                    if name in winner or winner.endswith(name):
                        actual_attn[i] = 1.0
                        break
            if actual_attn.sum() == 0:
                actual_attn = np.ones(n_modules) / n_modules

            # 3. 实际动作：从 embodiment metadata 获取，否则 0
            emb_md = signal.metadata.get("embodiment", {})
            if not isinstance(emb_md, dict):
                emb_md = {}
            actual_action = int(emb_md.get("action", 0)) if emb_md else 0
            # 限制到 self_schema.n_actions 范围内
            if not (0 <= actual_action < self._self_schema.n_actions):
                actual_action = 0

            # 4. 实际情感：从 PCN 误差分布推断
            pe = prediction_errors if isinstance(prediction_errors, dict) else {}
            affect = np.zeros(self._self_schema.affect_dim)
            affect_dim = self._self_schema.affect_dim
            layer_keys = ["L0", "L1", "L2", "global"]
            for i, k in enumerate(layer_keys[:affect_dim]):
                v = pe.get(k, 0.0)
                try:
                    affect[i] = float(v)
                except (TypeError, ValueError):
                    affect[i] = 0.0
            # 归一化（防止 NaN/inf）
            if not np.all(np.isfinite(affect)):
                affect = np.zeros(affect_dim)
            max_aff = float(affect.max()) if affect.size > 0 else 0.0
            if max_aff > 1.0:
                affect = affect / max_aff

            # 5. SelfSchema 更新
            self_diag = self._self_schema.update(
                integrated_state=integrated,
                actual_attention=actual_attn,
                actual_action=actual_action,
                actual_affect=affect,
            )

            # 6. PhiSelf：记录自我模块状态 + 计算 Φ_self
            self_module_states: list[np.ndarray] = []
            if self._l0 is not None:
                self_module_states.append(self._l0.state.copy())
            if self._l1 is not None:
                self_module_states.append(self._l1.state.copy())
            if self._proprioceptive_encoder is not None:
                # 用本体感觉编码作为第 3 个自我模块
                self_module_states.append(
                    np.asarray(integrated[: self._proprioceptive_encoder.output_dim])
                )
            self_module_states.append(integrated.copy())
            # PhiSelf 节流：每 10 步重算一次
            self._phi_self.record(self_module_states)
            if self._step_count % 10 == 0:
                phi_self_value = self._phi_self.compute_phi()
            else:
                phi_self_value = self._phi_self.last_phi

            # 7. AutobiographicalMemory：存储自我片段
            free_energy = float(sum(pe.values())) if pe else 0.0
            if not np.isfinite(free_energy):
                free_energy = 0.0
            self_pred_dict = {
                "attention": actual_attn.tolist(),
                "action": actual_action,
                "affect": affect.tolist(),
                "winner": winner,
            }
            actual_outcome_dict = {
                "prediction_errors": {k: float(v) for k, v in pe.items()},
                "step": self._step_count,
                "gwt_winner": winner,
            }
            self._autobiographical_memory.store(
                state_snapshot=integrated,
                self_prediction=self_pred_dict,
                actual_outcome=actual_outcome_dict,
                free_energy=free_energy,
                step=self._step_count,
                modality="iwsm",
            )

            # 8. CounterfactualSelf 记录转移
            prev_state = getattr(self, "_last_iwsm_state", integrated.copy())
            self._counterfactual_self.record_transition(
                state_before=prev_state,
                action=actual_action,
                state_after=integrated,
                outcome=actual_outcome_dict,
                free_energy=free_energy,
            )
            self._last_iwsm_state = integrated.copy()

            # 9. 偶尔生成反事实叙述（每 20 步）
            cf_result = None
            if self._step_count % 20 == 0:
                cf_result = self._counterfactual_self.generate_counterfactual()

            # 10. 写入 metadata
            if not hasattr(signal, "metadata") or signal.metadata is None:
                signal.metadata = {}
            iwsm_md = {
                "self_schema": self_diag,
                "self_schema_stats": self._self_schema.stats.__dict__
                if hasattr(self._self_schema.stats, "__dict__")
                else {},
                "phi_self": float(phi_self_value),
                "phi_self_history": (
                    list(self._phi_self.phi_history[-20:])
                    if hasattr(self._phi_self, "phi_history")
                    else []
                ),
                "is_sleeping": bool(phi_self_value < self._phi_self.sleep_threshold)
                if hasattr(self._phi_self, "sleep_threshold")
                else False,
                "autobiographical": {
                    "n_episodes": len(
                        self._autobiographical_memory._episodes
                    )
                    if hasattr(self._autobiographical_memory, "_episodes")
                    else 0,
                },
                "counterfactual": {
                    "narratives_generated": getattr(
                        self._counterfactual_self, "_narratives_generated", 0
                    ),
                    "last_regret": (
                        float(self._counterfactual_self._regret_history[-1])
                        if getattr(self._counterfactual_self, "_regret_history", None)
                        else 0.0
                    ),
                    "last_narrative": (
                        cf_result.narrative if cf_result is not None else ""
                    ),
                },
            }
            signal.metadata["iwsm"] = iwsm_md
        except Exception as exc:
            # 军事级健壮性：IWSM 失败不应中断 think()
            if not hasattr(signal, "metadata") or signal.metadata is None:
                signal.metadata = {}
            signal.metadata["iwsm"] = {"error": f"{type(exc).__name__}: {exc}"}

    def jepa_snapshot(self) -> dict:
        """返回 JEPA 模块快照。"""
        if self._jepa_module is None:
            return {"enabled": False}
        with self._lock:
            return self._jepa_module.snapshot()

    def gwt_snapshot(self) -> dict:
        """返回 GWT 模块快照。"""
        if self._gwt_selector is None:
            return {"enabled": False}
        with self._lock:
            return {
                "enabled": True,
                "selector": self._gwt_selector.snapshot(),
                "broadcaster": self._gwt_broadcaster.snapshot(),
                "phi": self._phi_calculator.snapshot(),
            }

    def discovery_snapshot(self) -> dict:
        """返回科学发现引擎快照。"""
        if self._science_loop is None:
            return {"enabled": False}
        with self._lock:
            return self._science_loop.snapshot()

    def embodiment_snapshot(self) -> dict:
        """返回具身主动感知模块快照。"""
        if self._embodied_body is None:
            return {"enabled": False}
        with self._lock:
            snap: dict[str, Any] = {"enabled": True}
            try:
                snap["body"] = (
                    self._embodied_body.get_snapshot()
                    if hasattr(self._embodied_body, "get_snapshot")
                    else {}
                )
            except Exception:
                snap["body"] = {}
            try:
                snap["sensorimotor"] = (
                    self._sensorimotor_predictor.get_snapshot()
                    if self._sensorimotor_predictor is not None
                    else {}
                )
            except Exception:
                snap["sensorimotor"] = {}
            try:
                snap["proprioception"] = (
                    self._proprioceptive_encoder.get_body_schema_snapshot()
                    if self._proprioceptive_encoder is not None
                    else {}
                )
            except Exception:
                snap["proprioception"] = {}
            snap["last_action"] = int(self._last_embodied_action)
            snap["step_count"] = int(self._embodiment_step_count)
            return snap

    def iwsm_snapshot(self) -> dict:
        """返回统一自我模型 IWSM 快照。"""
        if self._self_schema is None:
            return {"enabled": False}
        with self._lock:
            snap: dict[str, Any] = {"enabled": True}
            try:
                snap["self_schema"] = self._self_schema.stats.__dict__
            except Exception:
                snap["self_schema"] = {}
            try:
                snap["phi_self"] = float(self._phi_self.last_phi)
                snap["phi_self_history"] = list(
                    self._phi_self.phi_history[-20:]
                )
            except Exception:
                snap["phi_self"] = 0.0
                snap["phi_self_history"] = []
            try:
                snap["autobiographical"] = {
                    "n_episodes": len(self._autobiographical_memory._episodes)
                }
            except Exception:
                snap["autobiographical"] = {"n_episodes": 0}
            try:
                snap["counterfactual"] = {
                    "narratives_generated": getattr(
                        self._counterfactual_self, "_narratives_generated", 0
                    ),
                    "n_regrets": len(
                        getattr(self._counterfactual_self, "_regret_history", [])
                    ),
                }
            except Exception:
                snap["counterfactual"] = {}
            return snap

    def run_pcn_cycle(self, obs: np.ndarray) -> dict | None:
        """执行一次 PCN 前向/反向传播，返回层级误差信息。

        Phase G (四.3) 集成钩子：``ZeroDataModel.think()`` 在主处理完成后
        调用本方法，将观测喂入 L0/L1/L2 层级并运行预测编码更新。

        与 ``think()`` 的区别：本方法**不**调用 ``self._model.think()``
        （避免循环递归），只执行 PCN 层级的 forward/backward pass。

        流程（与 ``think()`` 内 PCN 部分一致）：
        1. L0 接收观测 ``set_state(obs)``
        2. 自顶向下：L2.predict() → L1.predict()
        3. 自底向上更新：L0.update → L1.update → L2.update
        4. 返回 ``layer_errors`` / ``state_norms`` / ``error_decreasing``

        Parameters
        ----------
        obs : ndarray
            观测向量（任意长度，自动截断/padding 到 ``L0.dim``）。

        Returns
        -------
        dict | None
            PCN 信息字典，含 ``layer_errors``/``state_norms``/
            ``error_decreasing``；``use_pcn=False`` 或 obs 为 None 时
            返回 ``None``。
        """
        if not self._use_pcn:
            return None
        if obs is None:
            return None

        with self._lock:
            obs = np.asarray(obs, dtype=np.float64).flatten()[: self._l0.dim]
            # padding 如果 obs 比 dim 短
            if obs.shape[0] < self._l0.dim:
                padded = np.zeros(self._l0.dim)
                padded[: obs.shape[0]] = obs
                obs = padded

            # NaN 防护
            if not np.all(np.isfinite(obs)):
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

            # L0 接收观测
            self._l0.set_state(obs)

            # 自顶向下预测
            l2_pred_l1 = self._l2.predict()  # L2 对 L1 的预测
            l1_pred_l0 = self._l1.predict()   # L1 对 L0 的预测

            # 自底向上更新
            error_l0 = self._l0.update(
                error_from_lower=None, prediction_from_higher=l1_pred_l0
            )
            error_l1 = self._l1.update(
                error_from_lower=error_l0, prediction_from_higher=l2_pred_l1
            )
            error_l2 = self._l2.update(
                error_from_lower=error_l1, prediction_from_higher=None
            )

            err_l0_norm = float(np.linalg.norm(error_l0)) if np.all(np.isfinite(error_l0)) else 0.0
            err_l1_norm = float(np.linalg.norm(error_l1)) if np.all(np.isfinite(error_l1)) else 0.0
            err_l2_norm = float(np.linalg.norm(error_l2)) if np.all(np.isfinite(error_l2)) else 0.0

            return {
                "layer_errors": {
                    "L0": err_l0_norm,
                    "L1": err_l1_norm,
                    "L2": err_l2_norm,
                },
                "state_norms": {
                    "L0": float(np.linalg.norm(self._l0.state)),
                    "L1": float(np.linalg.norm(self._l1.state)),
                    "L2": float(np.linalg.norm(self._l2.state)),
                },
                "error_decreasing": err_l0_norm > err_l1_norm,
            }

    def get_layer_snapshot(self) -> dict:
        """获取各层快照（用于可视化）。"""
        if not self._use_pcn:
            return {}
        with self._lock:
            return {
                "L0": self._l0.get_snapshot(),
                "L1": self._l1.get_snapshot(),
                "L2": self._l2.get_snapshot(),
            }

    def __getattr__(self, name):
        """委托给底层 model（让 HierarchicalZeroDataModel 可像 ZeroDataModel 一样使用）。

        F1 修复：防止 _model 未初始化时（pickle/copy/deepcopy/__init__
        异常中途）触发 __getattr__("_model") → getattr(self._model, "_model")
        → 再次 __getattr__ → 无限递归直到 RecursionError。对 _model
        及其内部属性直接抛 AttributeError 以打破循环。
        """
        if name == "_model":
            raise AttributeError(name)
        model = self.__dict__.get("_model")
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)

    # ------------------------------------------------------------------ #
    # Phase 2 集成（神经符号融合 + 元认知 + 主动实验）
    # ------------------------------------------------------------------ #
    def _init_phase2(self, dim: int) -> None:
        """惰性初始化 Phase 2 模块。"""
        from ..experiment.bayesian_experiment_planner import (
            BayesianExperimentPlannerV2,
        )
        from ..experiment.experiment_logger import ExperimentLogger
        from ..knowledge.logic_engine import LogicEngine
        from ..knowledge.reasoning_graph import ReasoningGraph
        from ..metacog.meta_trigger import MetaTrigger
        from ..metacog.second_order_belief import SecondOrderBelief

        self._logic_engine = LogicEngine()
        self._reasoning_graph = ReasoningGraph()
        # 二阶信念维度取 min(dim, 64) 以控制计算量。
        self._second_order_belief = SecondOrderBelief(dim=min(dim, 64))
        self._meta_trigger = MetaTrigger()
        self._experiment_planner = BayesianExperimentPlannerV2(
            eval_interval=500, n_samples=5, seed=42
        )
        self._experiment_planner.default_candidates(dim=min(dim, 64))
        self._experiment_logger = ExperimentLogger()

    def _belief_to_predicates(self, belief: np.ndarray) -> dict[str, float]:
        """将信念向量转换为谓词字典（供 LogicEngine 使用）。

        每个维度映射为谓词 ``d{i}``，真值为归一化后的激活值。
        截断到 ``_phase2_dim`` 维以控制字典大小。
        """
        bs = np.asarray(belief, dtype=np.float64).flatten()
        n = min(len(bs), self._phase2_dim)
        # 归一化到 [0, 1]（sigmoid）。
        # M5 修复：先 clip 到 [-50, 50] 防止 np.exp(-val) 溢出。
        # 原代码对大负值（如 -1000）会触发 RuntimeWarning: overflow
        # encountered in exp，虽结果正确（0.0）但污染日志。
        preds: dict[str, float] = {}
        for i in range(n):
            val = float(bs[i])
            if not np.isfinite(val):
                preds[f"d{i}"] = 0.5
                continue
            val = max(-50.0, min(50.0, val))
            preds[f"d{i}"] = 1.0 / (1.0 + np.exp(-val))
        return preds

    def _run_phase2_cycle(
        self,
        signal: Any,
        belief: np.ndarray,
        prediction_error: float,
        param_update_norm: float,
        memory_similarity: float,
    ) -> None:
        """运行 Phase 2 循环，将结果写入 signal.metadata。

        流程：
        1. 逻辑一致性检查（LogicEngine.check_consistency）
        2. 二阶信念更新（SecondOrderBelief.update）
        3. 元认知触发（MetaTrigger.update）
        4. 周期性实验规划（ExperimentPlanner.evaluate）
        5. 推理图查询（ReasoningGraph 传递性/类比推理）
        6. 将所有结果写入 metadata["cognitive_upgrades"]
        """
        if not self._use_phase2:
            return

        # 确保 metadata 和 cognitive_upgrades 存在。
        if not hasattr(signal, "metadata") or signal.metadata is None:
            try:
                signal.metadata = {}
            except (AttributeError, TypeError):
                return
        upgrades = signal.metadata.get("cognitive_upgrades")
        if not isinstance(upgrades, dict):
            upgrades = {}
            signal.metadata["cognitive_upgrades"] = upgrades

        # --- 1. 逻辑一致性检查 --- #
        belief_preds = self._belief_to_predicates(belief)
        contradiction = 0.0
        violations: list[dict] = []
        if self._logic_engine is not None:
            try:
                contradiction = self._logic_engine.check_consistency(
                    belief_preds
                )
            except Exception:
                contradiction = 0.0
            # 收集违反详情（简化：只报告矛盾度）。
            n_rules = len(self._logic_engine.rules) if self._logic_engine else 0
            if contradiction > 0.0 and n_rules > 0:
                violations.append({
                    "rule": "consistency_check",
                    "inferred": round(float(contradiction), 6),
                    "actual": 0.0,
                    "penalty": round(float(contradiction), 6),
                })

        upgrades["logic_violations"] = {
            "n_violations": len(violations),
            "total_penalty": round(float(contradiction), 6),
            "contradiction": round(float(contradiction), 6),
            "violations": violations,
        }

        # --- 2. 二阶信念更新 --- #
        meta_result: dict = {}
        if self._second_order_belief is not None:
            try:
                meta_result = self._second_order_belief.update(
                    belief_state=belief,
                    prediction_error=prediction_error,
                    param_update_norm=param_update_norm,
                    memory_similarity=memory_similarity,
                )
            except Exception:
                meta_result = {}

        # --- 3. 元认知触发 --- #
        trigger_result: dict = {"triggered": False, "mode": "normal"}
        if self._meta_trigger is not None and meta_result:
            try:
                trigger_result = self._meta_trigger.update(
                    uncertainty=meta_result.get("predicted_uncertainty", 0.5),
                    topic="belief_state",
                )
            except Exception:
                trigger_result = {"triggered": False, "mode": "normal"}

        confidence = meta_result.get("confidence", 0.5)
        mean_unc = meta_result.get("mean_uncertainty", 0.5)
        mode = "explore" if trigger_result.get("triggered") else "exploit"
        upgrades["meta_cognition"] = {
            "confidence": round(float(confidence) * 100.0, 2),
            "mode": mode,
            "mean_uncertainty": round(float(mean_unc), 6),
            "predicted_uncertainty": round(
                float(meta_result.get("predicted_uncertainty", 0.5)), 6
            ),
            "triggered": bool(trigger_result.get("triggered", False)),
            "threshold": round(
                float(trigger_result.get("threshold", 0.5)), 6
            ),
            "decomposition": meta_result.get("decomposition", {}),
        }

        # --- 4. 周期性实验规划 --- #
        exp_data: dict = {}
        if self._experiment_planner is not None:
            try:
                eval_result = self._experiment_planner.evaluate(
                    current_state=belief,
                    step=self._step_count,
                )
                if eval_result is not None:
                    # 执行选中的实验。
                    best = self._experiment_planner.select_best(belief)
                    if best is not None:
                        exec_result = self._experiment_planner.execute(
                            best, belief
                        )
                        # 记录到实验日志器。
                        if self._experiment_logger is not None:
                            self._experiment_logger.log_from_dict({
                                "step": self._step_count,
                                "hypothesis": best.name,
                                "intervention": best.intervention_type,
                                "info_gain": exec_result.get("actual_gain", 0.0),
                            })
                    exp_data = {
                        "name": eval_result.get("experiment", ""),
                        "intervention": eval_result.get(
                            "intervention_type", ""
                        ),
                        "predicted_gain": round(
                            float(eval_result.get("predicted_gain", 0.0)), 6
                        ),
                        "param_uncertainty": round(float(mean_unc), 6),
                        "history": (
                            list(self._experiment_planner._history)[-10:]
                            if self._experiment_planner._history
                            else []
                        ),
                    }
            except Exception:
                exp_data = {}

            # 即使未到评估间隔，也提供实验状态。
            if not exp_data:
                stats = self._experiment_planner.stats
                exp_data = {
                    "name": "",
                    "intervention": "",
                    "predicted_gain": 0.0,
                    "param_uncertainty": round(float(mean_unc), 6),
                    "history": [],
                    "n_candidates": stats.get("n_candidates", 0),
                    "n_executed": stats.get("n_executed", 0),
                }

        # 合并实验日志里程碑。
        if self._experiment_logger is not None:
            exp_data["milestones"] = self._experiment_logger.get_milestones()
        upgrades["experiment"] = exp_data

        # --- 5. 推理图查询 --- #
        reasoning: dict = {}
        if self._reasoning_graph is not None:
            try:
                virtual_obs = (
                    self._reasoning_graph.get_virtual_observations()
                )
                _rg_stats = self._reasoning_graph.stats
                reasoning = {
                    "pending_virtual_observations": len(virtual_obs),
                    "n_facts": int(_rg_stats.get("n_triples", 0)),
                    "n_rules": 0,  # ReasoningGraph 只有三元组，无规则
                    "recent_inferences": virtual_obs[:5],
                }
            except Exception:
                reasoning = {}
        upgrades["reasoning_chain"] = reasoning

        # H1 修复：_step_count 已在 think() 入口处递增，此处不再重复。
        # 原 self._step_count += 1 已移除，避免 phase2+skills 同时启用时双递增。

    @property
    def logic_engine(self) -> LogicEngine | None:
        """暴露 LogicEngine 供外部添加规则。"""
        return self._logic_engine

    @property
    def reasoning_graph(self) -> ReasoningGraph | None:
        """暴露 ReasoningGraph 供外部添加事实。"""
        return self._reasoning_graph

    @property
    def experiment_planner(self) -> BayesianExperimentPlannerV2 | None:
        """暴露 ExperimentPlanner 供外部附加沙盒。"""
        return self._experiment_planner

    @property
    def experiment_logger(self) -> ExperimentLogger | None:
        """暴露 ExperimentLogger 供外部查询里程碑。"""
        return self._experiment_logger

    # ------------------------------------------------------------------ #
    # Phase 3 集成（工具使用 + 全栈可观测性 + 人类教学）
    # ------------------------------------------------------------------ #
    def _init_phase3(self) -> None:
        """惰性初始化 Phase 3 模块。"""
        if self._enable_tools:
            from ..tools.safety import SafetyChecker
            from ..tools.tool_policy import ToolPolicy
            from ..tools.tool_registry import ToolRegistry

            self._tool_registry = ToolRegistry()
            self._safety_checker = SafetyChecker()
            self._tool_policy = ToolPolicy(
                registry=self._tool_registry,
                safety_checker=self._safety_checker,
            )

        if self._enable_teaching:
            from ..observability.human_teaching import HumanFeedback

            self._human_feedback = HumanFeedback()

        if self._enable_observability:
            from ..observability.audit import AuditLogger
            from ..observability.counterfactual_explainer import (
                CounterfactualExplainer,
            )
            from ..observability.thought_chain import ThoughtChain

            self._thought_chain = ThoughtChain()
            self._counterfactual_explainer = CounterfactualExplainer()
            self._audit_logger = AuditLogger()

    def _run_phase3_cycle(
        self,
        signal: Any,
        belief: np.ndarray,
        prediction_error: float,
        confidence: float,
        meta_triggered: bool,
        prediction_errors: dict[str, float],
    ) -> None:
        """运行 Phase 3 循环，将结果写入 signal.metadata['phase3']。

        流程：
        1. 记录思维链（ThoughtChain）
        2. 工具调用决策（ToolPolicy）+ 执行
        3. 人类教学反馈检查
        4. 审计日志记录
        5. 将所有结果写入 metadata['phase3']
        """
        phase3_data: dict[str, Any] = {}
        tool_called = ""

        # --- 1. 思维链记录 --- #
        if self._thought_chain is not None:
            belief_list = (
                belief.flatten().tolist()[:20] if belief.size > 0 else []
            )
            self._thought_chain.record(
                step=self._step_count,
                belief_before=belief_list,
                belief_after=belief_list,
                prediction_errors=prediction_errors,
                confidence=confidence,
                meta_triggered=meta_triggered,
                free_energy=prediction_error,
            )
            phase3_data["thought_chain"] = {
                "n_records": self._thought_chain.length,
                "n_anomalies": self._thought_chain.n_anomalies,
            }

        # --- 2. 工具调用决策 --- #
        if self._tool_policy is not None and belief.size > 0:
            try:
                context = belief.flatten()[: min(belief.size, 32)]
                decision = self._tool_policy.decide(
                    context=context,
                    prediction_error=prediction_error,
                    confidence=confidence,
                )
                if decision is not None:
                    result = self._tool_policy.execute(decision)
                    tool_called = decision.tool_name
                    phase3_data["tool_call"] = {
                        "tool": decision.tool_name,
                        "success": result.success,
                        "output": str(result.output)[:200],
                        "expected_fe": round(decision.expected_free_energy, 6),
                        "reason": decision.reason,
                    }
                    # 审计日志
                    if self._audit_logger is not None:
                        from ..observability.audit import AuditEventType

                        self._audit_logger.log(
                            event_type=AuditEventType.TOOL_CALL,
                            module="hierarchical_model",
                            context={"step": self._step_count},
                            result={
                                "tool": decision.tool_name,
                                "success": result.success,
                            },
                        )
            except Exception:
                phase3_data["tool_call"] = {"error": "tool_decision_failed"}

        # 重新记录思维链的工具调用（如果有）。
        # 已在上面 record 中 tool_called 默认为空，这里若调用了工具
        # 则额外记一条标注工具的异常记录，便于前端高亮。
        if tool_called and self._thought_chain is not None:
            self._thought_chain.record(
                step=self._step_count,
                confidence=confidence,
                tool_called=tool_called,
                prediction_errors=prediction_errors,
            )

        # --- 3. 人类教学反馈 --- #
        if self._human_feedback is not None:
            corrections = self._human_feedback.get_corrections(n=1)
            if corrections:
                phase3_data["human_correction"] = {
                    "step": corrections[0].get("step", -1),
                    "error_signal": corrections[0].get("error_signal", 0.0),
                }
                # 审计日志
                if self._audit_logger is not None:
                    from ..observability.audit import AuditEventType

                    self._audit_logger.log(
                        event_type=AuditEventType.HUMAN_INTERVENTION,
                        module="hierarchical_model",
                        context={"step": self._step_count},
                        result={"correction": corrections[0]},
                    )
            phase3_data["teaching"] = self._human_feedback.stats

        # --- 4. 审计日志：动作选择 --- #
        if self._audit_logger is not None:
            from ..observability.audit import AuditEventType

            self._audit_logger.log(
                event_type=AuditEventType.ACTION_SELECTION,
                module="hierarchical_model",
                context={"step": self._step_count},
                result={
                    "confidence": round(confidence, 4),
                    "prediction_error": round(prediction_error, 6),
                    "meta_triggered": meta_triggered,
                    "tool_called": tool_called,
                },
            )
            phase3_data["audit"] = {
                "n_entries": self._audit_logger.n_entries,
            }

        # --- 5. 写入 metadata --- #
        if not hasattr(signal, "metadata") or signal.metadata is None:
            try:
                signal.metadata = {}
            except (AttributeError, TypeError):
                return
        signal.metadata["phase3"] = phase3_data

    # ------------------------------------------------------------------ #
    # Phase 3 属性
    # ------------------------------------------------------------------ #
    @property
    def tool_registry(self) -> ToolRegistry | None:
        """暴露 ToolRegistry 供外部注册工具。"""
        return self._tool_registry

    @property
    def tool_policy(self) -> ToolPolicy | None:
        """暴露 ToolPolicy 供外部调整策略参数。"""
        return self._tool_policy

    @property
    def safety_checker(self) -> SafetyChecker | None:
        """暴露 SafetyChecker 供外部查询违规日志。"""
        return self._safety_checker

    @property
    def human_feedback(self) -> HumanFeedback | None:
        """暴露 HumanFeedback 供外部给予反馈。"""
        return self._human_feedback

    @property
    def thought_chain(self) -> ThoughtChain | None:
        """暴露 ThoughtChain 供外部查询思维记录。"""
        return self._thought_chain

    @property
    def counterfactual_explainer(self) -> CounterfactualExplainer | None:
        """暴露 CounterfactualExplainer 供外部生成反事实解释。"""
        return self._counterfactual_explainer

    @property
    def audit_logger(self) -> AuditLogger | None:
        """暴露 AuditLogger 供外部查询/导出审计日志。"""
        return self._audit_logger

    def get_observability_snapshot(self) -> dict[str, Any]:
        """获取全栈可观测性快照（供前端面板消费）。

        汇总思维链统计、审计日志统计、工具调用历史和人类教学统计。
        """
        snapshot: dict[str, Any] = {}
        if self._thought_chain is not None:
            snapshot["thought_chain"] = self._thought_chain.stats
            snapshot["thought_chain_recent"] = self._thought_chain.get_recent(10)
            snapshot["thought_chain_anomalies"] = (
                self._thought_chain.get_anomalies(10)
            )
        if self._audit_logger is not None:
            snapshot["audit"] = self._audit_logger.stats
            snapshot["audit_recent"] = self._audit_logger.get_recent(10)
        if self._tool_policy is not None:
            snapshot["tool_policy"] = {
                "confidence_threshold": self._tool_policy.confidence_threshold,
            }
        if self._human_feedback is not None:
            snapshot["human_feedback"] = self._human_feedback.stats
        if self._counterfactual_explainer is not None:
            snapshot["counterfactual"] = self._counterfactual_explainer.stats
        return snapshot
