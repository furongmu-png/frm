# src/zero_data_model/plasticity/architect.py
"""架构可塑性优化器：动态模块生长与剪枝。

本模块为 ZeroDataModel 提供可选的"结构可塑性"钩子。通过持续记录各子模块的
预测误差，定期评估每个模块的误差占比（占所有模块误差总和的比例），据此决定：

- 分裂（split）：某模块长期承担过高误差（占比 > split_error_ratio），说明其
  表征能力不足，需要复制并施加微小扰动以扩充容量；
- 剪枝（prune）：某模块长期误差占比极低（< prune_error_ratio），说明其处于
  休眠状态，可移入"休眠池"以节省计算，并在需要时重新激活。

设计要点：
- 完全独立、自包含、可插拔；不修改 model.py，仅通过外部传入的 modules 列表
  进行增删。
- 所有 numpy 操作均做防御性处理（窄异常捕获），避免形状不匹配导致崩溃。
- 被剪枝的模块保存在 self._dormant_modules 中，可经 reactivate() 恢复。
- ``evaluate()`` **不修改**调用方传入的 ``modules`` 列表，只返回"计划动作"
  （list of dict），由调用方（model.py）在自己的锁下执行实际增删。
- 通过 ``self._lock``（``RLock``）保护所有可变状态（``_error_history``/
  ``_high_streak``/``_low_streak``/``_dormant_modules``/``rng`` 等）。
"""

from __future__ import annotations

import contextlib
import copy
import logging
import threading
from collections import deque
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


class ArchitectureOptimizer:
    """动态模块生长/剪枝优化器。

    通过误差占比的连续计数（streak）触发模块的分裂或剪枝，实现 ZeroDataModel
    的结构可塑性。``evaluate()`` 只**规划**动作并返回描述，不直接修改调用方
    传入的 ``modules`` 列表；实际增删由调用方（model.py）执行。

    Parameters
    ----------
    dim:
        表征维度。``dim`` 是公共 API 契约的一部分（``model.py`` 通过
        ``ArchitectureOptimizer(dim=dim, seed=...)`` 构造），本优化器内部
        不强制使用，保留以便未来扩展且不破坏既有调用方。
    seed:
        随机数种子，用于分裂时的权重扰动。
    eval_interval:
        每隔多少次 ``evaluate()`` 调用执行一次误差占比评估。
    split_threshold_steps:
        连续高误差达到该**评估周期数**即触发分裂。

        .. note::
            此处的 "steps" 计的是 ``evaluate()`` 调用次数（评估周期），
            **而非** ``think()`` 的原始步数。由于 streak 仅在评估实际发生时
            （即 ``_step_count % eval_interval == 0``）递增，故阈值单位是
            "评估周期"。保留 ``*_steps`` 命名以兼容 model.py 既有调用。
    split_error_ratio:
        高误差占比阈值。
    prune_threshold_steps:
        连续低误差达到该**评估周期数**即触发剪枝（单位同上）。
    prune_error_ratio:
        低误差占比阈值。
    """

    def __init__(
        self,
        dim: int = 64,  # pylint: disable=unused-argument
        seed: int = 42,
        eval_interval: int = 100,
        split_threshold_steps: int = 200,
        split_error_ratio: float = 0.3,
        prune_threshold_steps: int = 500,
        prune_error_ratio: float = 0.01,
    ) -> None:
        # 存储配置参数。dim 保留为公共 API 契约（model.py 传入），内部不用，
        # 故标记 unused-argument。
        self.dim = dim
        self.seed = seed
        self.eval_interval = max(1, eval_interval)
        # 注意：*_threshold_steps 的单位是 evaluate() 调用次数（评估周期），
        # 不是 think() 原始步数（见类 docstring）。
        self.split_threshold_steps = split_threshold_steps
        self.split_error_ratio = split_error_ratio
        self.prune_threshold_steps = prune_threshold_steps
        self.prune_error_ratio = prune_error_ratio

        # 随机数生成器（用于分裂时的权重扰动）
        self.rng = np.random.default_rng(seed)

        # 每个模块的误差历史（deque maxlen=500，按模块名索引）
        self._error_history: dict[str, deque] = {}
        # 被剪枝但保留的休眠模块（可重新激活）
        self._dormant_modules: dict[str, Any] = {}
        # 连续高误差计数 / 连续低误差计数（按模块名索引，单位：评估周期）
        self._high_streak: dict[str, int] = {}
        self._low_streak: dict[str, int] = {}
        # streak 起始的 think() cycle（用于诊断 streak 持续了多少原始周期）
        self._streak_start_cycle: dict[str, int] = {}

        # 步数计数与统计计数器
        self._step_count = 0
        self._total_splits = 0
        self._total_prunes = 0
        self._active_count = 0

        # 可重入锁：保护 _error_history/_high_streak/_low_streak/
        # _dormant_modules/_streak_start_cycle/rng/_step_count 等所有可变状态。
        # think() 会并发调用各模块，record_errors/evaluate/_split_module/
        # _prune_module/reactivate 均需在 self._lock 下访问共享状态。
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def record_errors(
        self, module_names: list[str], errors: list[float]
    ) -> None:
        """记录每个模块的本步误差。

        将 ``errors`` 中每个误差追加到对应模块的 deque 中；若该模块尚无
        误差历史，则按需创建一个 maxlen=500 的 deque。
        """
        with self._lock:
            for name, err in zip(module_names, errors, strict=False):
                if name not in self._error_history:
                    self._error_history[name] = deque(maxlen=500)
                try:
                    self._error_history[name].append(float(err))
                except (ValueError, TypeError, FloatingPointError, OverflowError) as exc:
                    # 单个误差记录失败不应影响其它模块；记录后继续
                    _logger.warning(
                        "record_errors: failed to record error for %r: %r",
                        name, exc,
                    )
                    continue

    def evaluate(self, modules: list, cycle: int) -> dict[str, Any]:
        """周期性评估模块误差占比，规划分裂/剪枝动作。

        每次调用内部步数计数 +1；每隔 ``eval_interval`` 次调用执行一次评估：
        计算每个模块的平均误差占总误差的比例，并维护连续高/低误差计数
        （单位：评估周期）。当连续高误差超过 ``split_threshold_steps`` 时
        规划分裂该模块；连续低误差超过 ``prune_threshold_steps`` 时规划剪枝。

        ``cycle`` 为外部传入的 ``think()`` 周期序号，用于记录 streak 起始
        周期并在动作描述中报告 streak 持续的原始周期数（诊断用）。

        .. important::
            本方法**不修改**调用方传入的 ``modules`` 列表。开头先做
            ``modules = list(modules)`` 浅拷贝，任何后续操作只影响该拷贝；
            实际的分裂/剪枝增删由调用方（model.py）根据返回的 ``actions``
            在自己的锁下执行。返回值中的 ``actions`` 为计划动作描述列表。

        Returns
        -------
        dict
            ``{"actions", "splits", "prunes", "dormant_count", "module_count"}``。
            其中 ``actions`` 为 ``{"action", "module", "reason", ...}`` dict
            列表；``splits``/``prunes`` 为模块名列表（向后兼容）。
        """
        # 关键：浅拷贝输入列表，确保后续任何 append/pop 只影响拷贝，
        # 不破坏调用方（model.py 的 self.modules）。
        modules = list(modules)

        with self._lock:
            # 顶层捕获以保护 think()：记录异常类型与消息后返回安全默认
            try:
                self._step_count += 1
                self._active_count = len(modules)

                actions: list[dict[str, Any]] = []
                splits: list[str] = []
                prunes: list[str] = []

                # 仅在评估间隔触发时进行占比计算
                if self._step_count % self.eval_interval != 0:
                    return {
                        "actions": actions,
                        "splits": splits,
                        "prunes": prunes,
                        "dormant_count": len(self._dormant_modules),
                        "module_count": len(modules),
                    }

                # 先对模块列表做快照，收集名称与平均误差，避免评估期间列表变动干扰
                names: list[str] = []
                means: list[float] = []
                for mod in modules:
                    name = self._module_name(mod)
                    names.append(name)
                    hist = self._error_history.get(name)
                    if hist is None or len(hist) == 0:
                        means.append(0.0)
                    else:
                        try:
                            m = float(np.mean(list(hist)))
                            # 防御 NaN/inf
                            if not np.isfinite(m):
                                m = 0.0
                            means.append(m)
                        except (ValueError, TypeError, FloatingPointError, OverflowError):
                            means.append(0.0)

                total = float(np.sum(means))
                if not np.isfinite(total):
                    total = 0.0
                if total < 1e-12:
                    # 所有模块误差都接近零：重置 streak 并直接返回
                    for name in names:
                        self._high_streak[name] = 0
                        self._low_streak[name] = 0
                        self._streak_start_cycle[name] = cycle
                    return {
                        "actions": actions,
                        "splits": splits,
                        "prunes": prunes,
                        "dormant_count": len(self._dormant_modules),
                        "module_count": len(modules),
                    }

                # 计算每个模块的误差占比并维护 streak（单位：评估周期）
                for name, mean_err in zip(names, means, strict=False):
                    # NaN 防御：total==0 已在上面处理；此处再兜底
                    ratio = 0.0 if total == 0.0 else mean_err / total
                    if not np.isfinite(ratio):
                        ratio = 0.0

                    if ratio > self.split_error_ratio:
                        prev = self._high_streak.get(name, 0)
                        self._high_streak[name] = prev + 1
                        if prev == 0:
                            # streak 刚开始：记录起始 cycle
                            self._streak_start_cycle[name] = cycle
                        self._low_streak[name] = 0
                    elif ratio < self.prune_error_ratio:
                        prev = self._low_streak.get(name, 0)
                        self._low_streak[name] = prev + 1
                        if prev == 0:
                            self._streak_start_cycle[name] = cycle
                        self._high_streak[name] = 0
                    else:
                        self._high_streak[name] = 0
                        self._low_streak[name] = 0
                        self._streak_start_cycle[name] = cycle

                    # 分裂触发：连续高误差超过阈值
                    if self._high_streak.get(name, 0) > self.split_threshold_steps:
                        plan = self._split_module(modules, name, cycle)
                        if plan is not None:
                            actions.append(plan)
                            new_name = plan.get("new_name") or name
                            splits.append(new_name)
                        self._high_streak[name] = 0
                        self._streak_start_cycle[name] = cycle

                    # 剪枝触发：连续低误差超过阈值
                    if self._low_streak.get(name, 0) > self.prune_threshold_steps:
                        plan = self._prune_module(modules, name, cycle)
                        if plan is not None:
                            actions.append(plan)
                            prunes.append(name)
                        self._low_streak[name] = 0
                        self._streak_start_cycle[name] = cycle

                self._active_count = len(modules)
                return {
                    "actions": actions,
                    "splits": splits,
                    "prunes": prunes,
                    "dormant_count": len(self._dormant_modules),
                    "module_count": len(modules),
                }
            except Exception as exc:  # noqa: BLE001 - 保护 think()，需宽捕获
                # 宽捕获以保护 think() 不被架构评估异常中断；记录类型与消息
                _logger.warning(
                    "ArchitectureOptimizer.evaluate failed (%s): %s",
                    type(exc).__name__, exc,
                )
                return {
                    "actions": [],
                    "splits": [],
                    "prunes": [],
                    "dormant_count": len(self._dormant_modules),
                    "module_count": len(modules),
                }

    def reactivate(self, modules: list, name: str) -> bool:
        """将一个休眠模块重新激活并放回活跃列表。

        Returns
        -------
        bool
            成功恢复返回 True；休眠池中不存在该名称则返回 False。
        """
        with self._lock:
            try:
                if name in self._dormant_modules:
                    mod = self._dormant_modules.pop(name)
                    modules.append(mod)
                    return True
                return False
            except (KeyError, TypeError, AttributeError) as exc:
                _logger.warning(
                    "ArchitectureOptimizer.reactivate failed: %r", exc
                )
                return False

    def dormant_names(self) -> list[str]:
        """返回所有休眠模块的名称列表。"""
        with self._lock:
            return list(self._dormant_modules.keys())

    @property
    def stats(self) -> dict[str, Any]:
        """返回优化器的统计信息。"""
        with self._lock:
            return {
                "active_count": self._active_count,
                "dormant_count": len(self._dormant_modules),
                "total_splits": self._total_splits,
                "total_prunes": self._total_prunes,
            }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _module_name(mod: Any) -> str:
        """获取模块名称：优先 name 属性，其次类名。"""
        try:
            name = getattr(mod, "name", None)
            if isinstance(name, str) and name:
                return name
        except (AttributeError, TypeError):
            pass
        return mod.__class__.__name__

    def _split_module(
        self, modules: list, name: str, cycle: int
    ) -> dict[str, Any] | None:
        """**规划**分裂指定模块：深拷贝并对权重施加微小扰动，返回动作描述。

        .. important::
            本方法**不修改** ``modules`` 列表（不 append）。它只准备新模块
            并返回一个描述 dict，由调用方决定是否实际追加。在描述中附带准备
            好的 ``new_module`` 以便调用方直接使用。

        在拷贝上对其 ``transition``/``emission`` 等 numpy 数组属性叠加
        ``rng.normal(0, 0.01, size=arr.shape)`` 的小扰动，并以
        ``{name}_split_{N}`` 作为新模块名。失败时返回 None。
        """
        with self._lock:
            try:
                target = None
                for mod in modules:
                    if self._module_name(mod) == name:
                        target = mod
                        break
                if target is None:
                    return None

                new_mod = copy.deepcopy(target)

                # 对权重类 numpy 数组属性施加微小随机扰动
                for attr in ("transition", "emission"):
                    arr = getattr(new_mod, attr, None)
                    if isinstance(arr, np.ndarray):
                        try:
                            perturbed = arr + self.rng.normal(
                                0.0, 0.01, size=arr.shape
                            )
                            # 防御 NaN/inf 扰动结果
                            if not np.isfinite(perturbed).all():
                                continue
                            setattr(new_mod, attr, perturbed)
                        except (ValueError, TypeError, FloatingPointError, OverflowError) as exc:
                            # 形状不匹配等异常：保留原值，仅作记录
                            _logger.warning(
                                "_split_module: perturb %s failed: %r", attr, exc
                            )
                            continue

                # 计数并设置新名称
                self._total_splits += 1
                new_name = f"{name}_split_{self._total_splits}"
                # 模块可能使用 __slots__ 或只读 name，忽略设置失败
                with contextlib.suppress(AttributeError, TypeError):
                    new_mod.name = new_name

                streak_cycles = cycle - self._streak_start_cycle.get(name, cycle)
                return {
                    "action": "split",
                    "module": name,
                    "new_name": new_name,
                    "new_module": new_mod,
                    "reason": (
                        f"high error ratio streak > "
                        f"{self.split_threshold_steps} eval-cycles"
                    ),
                    "eval_cycle": cycle,
                    "streak_cycles": streak_cycles,
                }
            except (
                ValueError, TypeError, AttributeError,
                FloatingPointError, OverflowError,
            ) as exc:
                _logger.warning("_split_module failed: %r", exc)
                return None

    def _prune_module(
        self, modules: list, name: str, cycle: int
    ) -> dict[str, Any] | None:
        """**规划**剪枝指定模块：返回动作描述，**不修改** ``modules`` 列表。

        .. important::
            本方法**不**从 ``modules`` 弹出、也**不**将模块移入
            ``_dormant_modules``。它只返回一个描述 dict，由调用方决定是否
            实际执行剪枝（并负责维护休眠池）。

        Returns
        -------
        dict | None
            成功定位模块返回描述 dict；未找到返回 None。
        """
        with self._lock:
            try:
                # 仅查找是否存在目标模块（不弹出）
                found = any(
                    self._module_name(mod) == name for mod in modules
                )
                if not found:
                    return None

                self._total_prunes += 1
                streak_cycles = cycle - self._streak_start_cycle.get(name, cycle)
                return {
                    "action": "prune",
                    "module": name,
                    "reason": (
                        f"low error ratio streak > "
                        f"{self.prune_threshold_steps} eval-cycles"
                    ),
                    "eval_cycle": cycle,
                    "streak_cycles": streak_cycles,
                }
            except (
                ValueError, TypeError, AttributeError,
                FloatingPointError, OverflowError,
            ) as exc:
                _logger.warning("_prune_module failed: %r", exc)
                return None
