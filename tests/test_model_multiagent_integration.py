"""multiagent 模块集成进 ``think()`` 认知循环的单元测试。

覆盖任务第四步要求的 5 个契约：
1. ``enable_multiagent=False`` 时 metadata 不出现 ``phase7_multiagent``。
2. ``enable_multiagent=True`` 时 metadata 出现 ``phase7_multiagent`` 且字段正确。
3. multiagent hook 异常时不 crash ``think()``。
4. 三个特化指标被正确更新（用 prometheus_client 的 REGISTRY 检查）。
5. 多周期 ``think()`` 后 ``collaboration_events`` 累积。
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.base import Signal
from zero_data_model.metrics import ZDM_METRICS
from zero_data_model.model import ZeroDataModel

# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #


def _input(dim: int = 8) -> np.ndarray:
    """确定性的 8 维输入向量。"""
    return np.random.default_rng(0).normal(0, 1, dim)


def _read_gauge(name: str) -> float | None:
    """从 prometheus 默认 REGISTRY 读取一个无标签 Gauge 的当前值。

    找不到时返回 None。仅在 prometheus_client 已安装时可用。
    """
    from prometheus_client import REGISTRY

    for mfamily in REGISTRY.collect():
        if mfamily.name == name:
            for sample in mfamily.samples:
                if sample.name == name:
                    return sample.value
    return None


# --------------------------------------------------------------------------- #
# 1. enable_multiagent=False：不出现 phase7_multiagent
# --------------------------------------------------------------------------- #


def test_disabled_no_phase7_multiagent():
    """默认构造（multiagent 关闭）：metadata 不含 cognitive_upgrades。"""
    model = ZeroDataModel(dim=8, seed=42)
    assert model.multiagent_world is None
    assert model.multiagent_communication is None
    assert model.multiagent_culture is None

    sig = model.think(_input())
    md = sig.metadata
    # 默认所有 flag 关闭 → 不出现 cognitive_upgrades（零回归契约）
    assert "cognitive_upgrades" not in md
    assert "phase7_multiagent" not in md


def test_disabled_no_phase7_multiagent_with_other_flags():
    """其他升级 flag 开启但 multiagent 关闭：cognitive_upgrades 中无 phase7_multiagent。"""
    model = ZeroDataModel(dim=8, seed=42, enable_architect=True)
    assert model.multiagent_world is None

    sig = model.think(_input())
    upgrades = sig.metadata.get("cognitive_upgrades", {})
    assert "phase7_multiagent" not in upgrades


# --------------------------------------------------------------------------- #
# 2. enable_multiagent=True：出现 phase7_multiagent 且字段正确
# --------------------------------------------------------------------------- #


def test_enabled_surfaces_phase7_multiagent():
    """启用 multiagent：metadata.cognitive_upgrades.phase7_multiagent 字段齐全。"""
    model = ZeroDataModel(dim=8, seed=42, enable_multiagent=True)
    assert model.multiagent_world is not None
    assert model.multiagent_communication is not None
    assert model.multiagent_culture is not None

    sig = model.think(_input())
    ma = sig.metadata["cognitive_upgrades"]["phase7_multiagent"]

    assert set(ma.keys()) == {
        "collaboration_events",
        "communication_usage",
        "culture_generations",
        "agents",
        "step",
    }
    assert ma["agents"] == 2
    assert ma["step"] >= 1
    # 通信与文化每周期各驱动一次 → 至少为 1
    assert ma["communication_usage"] >= 1
    assert ma["culture_generations"] >= 1
    # 协作事件取决于动作对齐，非负即可
    assert ma["collaboration_events"] >= 0


# --------------------------------------------------------------------------- #
# 3. multiagent hook 异常时不 crash think()
# --------------------------------------------------------------------------- #


def test_hook_exception_does_not_crash_think():
    """world.step 抛异常时被 try/except 收窄，think() 仍正常返回。"""
    model = ZeroDataModel(dim=8, seed=42, enable_multiagent=True)

    def boom(*_args, **_kwargs):
        raise RuntimeError("multiagent boom")

    model.multiagent_world.step = boom  # type: ignore[method-assign]

    # 不应抛出
    sig = model.think(_input())
    # hook 失败 → phase7_multiagent 未写入 upgrade_meta
    upgrades = sig.metadata.get("cognitive_upgrades", {})
    assert "phase7_multiagent" not in upgrades
    # think() 核心结果仍然有效
    assert sig.data is not None


# --------------------------------------------------------------------------- #
# 4. 特化指标被正确更新（用 prometheus_client 的 REGISTRY 检查）
# --------------------------------------------------------------------------- #


def test_specialized_metrics_updated():
    """三个特化指标在 think() 后被设置为 metadata 中对应的值。"""
    if not ZDM_METRICS.prometheus_available:
        pytest.skip("prometheus_client 未安装，指标为 no-op，跳过 REGISTRY 检查")

    model = ZeroDataModel(dim=8, seed=42, enable_multiagent=True)
    sig = model.think(_input())
    ma = sig.metadata["cognitive_upgrades"]["phase7_multiagent"]

    assert _read_gauge("zdm_multiagent_collaboration_events") == float(
        ma["collaboration_events"]
    )
    assert _read_gauge("zdm_communication_usage") == float(
        ma["communication_usage"]
    )
    assert _read_gauge("zdm_culture_generations") == float(
        ma["culture_generations"]
    )


# --------------------------------------------------------------------------- #
# 5. 多周期 think() 后 collaboration_events 累积
# --------------------------------------------------------------------------- #


def test_collaboration_events_accumulate_over_cycles():
    """多周期 think() 后 collaboration_events 单调累积。

    为使协作可确定性地发生，将内部各智能体的 think 替换为返回固定对齐
    信号（余弦相似度 = 1.0 >= 0.8），于是每步 world.step 都会记录 1 对
    智能体的协作事件。
    """
    model = ZeroDataModel(dim=8, seed=42, enable_multiagent=True)

    fixed = Signal(data=np.ones(8))

    def _aligned_think(_obs):  # noqa: ARG001
        return fixed

    for agent in model.multiagent_world.agents:
        agent.model.think = _aligned_think  # type: ignore[method-assign]

    events: list[int] = []
    for _ in range(4):
        sig = model.think(_input())
        events.append(
            sig.metadata["cognitive_upgrades"]["phase7_multiagent"][
                "collaboration_events"
            ]
        )

    # 单调非减
    assert all(events[i + 1] >= events[i] for i in range(len(events) - 1))
    # 至少增长一次（每步 1 对智能体 → 1 事件/步）
    assert events[-1] > events[0]
    assert events[-1] >= 4
