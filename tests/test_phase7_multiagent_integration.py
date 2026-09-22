"""Phase 7 — Multi-agent think() integration tests.

集成层面验证 multiagent 模块被正确 hook 进 ``ZeroDataModel.think()`` 认知
循环：与其它升级 flag 共存、多周期统计单调推进、内部智能体不递归、metadata
JSON 可序列化，以及特化指标与 metadata 一致。

与 ``test_model_multiagent_integration.py``（单元契约）互补，本文件聚焦端到端
集成行为。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from zero_data_model.metrics import ZDM_METRICS
from zero_data_model.model import ZeroDataModel


def _input(dim: int = 8) -> np.ndarray:
    return np.random.default_rng(7).normal(0, 1, dim)


def _read_gauge(name: str) -> float | None:
    try:
        from prometheus_client import REGISTRY
    except ImportError:
        return None
    for mfamily in REGISTRY.collect():
        if mfamily.name == name:
            for sample in mfamily.samples:
                if sample.name == name:
                    return sample.value
    return None


# --------------------------------------------------------------------------- #
# 1. 默认构造：multiagent 属性为 None（零回归）
# --------------------------------------------------------------------------- #


def test_default_construct_multiagent_unset():
    model = ZeroDataModel(dim=8, seed=42)
    assert model.multiagent_world is None
    assert model.multiagent_communication is None
    assert model.multiagent_culture is None
    sig = model.think(_input())
    # 默认无任何升级 flag → 不出现 cognitive_upgrades
    assert "cognitive_upgrades" not in sig.metadata


# --------------------------------------------------------------------------- #
# 2. 与其它升级 flag 共存：metadata JSON 可序列化 + phase7_multiagent 在场
# --------------------------------------------------------------------------- #


def test_multiagent_with_other_flags_json_serializable():
    model = ZeroDataModel(
        dim=8,
        seed=42,
        enable_architect=True,
        enable_multiagent=True,
    )
    sig = model.think(_input())
    md = sig.metadata
    # cognitive_upgrades 存在且含 phase7_multiagent
    assert "cognitive_upgrades" in md
    assert "phase7_multiagent" in md["cognitive_upgrades"]
    # 整个 metadata 可被 json 序列化（_sanitize_for_json 已剥离 numpy 类型）
    serialized = json.dumps(md)
    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert "phase7_multiagent" in parsed["cognitive_upgrades"]


# --------------------------------------------------------------------------- #
# 3. 多周期：step / communication_usage / culture_generations 单调推进
# --------------------------------------------------------------------------- #


def test_multi_cycle_stats_progress_monotonically():
    """自然多周期：step、communication_usage、culture_generations 随周期单调递增。

    这三者由 hook 每周期确定性驱动（world.step + 一次 record_usage + 一次
    record_generation），故可严格断言递增；collaboration_events 仅断言非减。
    """
    model = ZeroDataModel(dim=8, seed=42, enable_multiagent=True)
    steps: list[int] = []
    comm: list[int] = []
    cult: list[int] = []
    collab: list[int] = []
    for _ in range(3):
        sig = model.think(_input())
        ma = sig.metadata["cognitive_upgrades"]["phase7_multiagent"]
        steps.append(ma["step"])
        comm.append(ma["communication_usage"])
        cult.append(ma["culture_generations"])
        collab.append(ma["collaboration_events"])

    # step 每周期 +1
    assert steps == [1, 2, 3]
    # 通信/文化每周期 +1
    assert comm == [1, 2, 3]
    assert cult == [1, 2, 3]
    # 协作事件非减
    assert all(collab[i + 1] >= collab[i] for i in range(len(collab) - 1))
    # world 内部步数与 metadata 一致
    assert model.multiagent_world.step_count == 3


# --------------------------------------------------------------------------- #
# 4. 内部智能体不递归：inner models 未启用 multiagent（无无限递归）
# --------------------------------------------------------------------------- #


def test_inner_agents_do_not_recurse_multiagent():
    """MultiAgentWorld 内部构造的 ZeroDataModel 使用默认 flag，
    其 multiagent_world 为 None，因此 think() 不会无限递归。"""
    model = ZeroDataModel(dim=8, seed=42, enable_multiagent=True)
    for agent in model.multiagent_world.agents:
        assert agent.model.multiagent_world is None
    # think() 能正常返回即证明无无限递归
    sig = model.think(_input())
    assert sig.data is not None
    assert sig.metadata["cognitive_upgrades"]["phase7_multiagent"]["agents"] == 2


# --------------------------------------------------------------------------- #
# 5. 特化指标与 metadata 一致（prometheus 可用时）
# --------------------------------------------------------------------------- #


def test_metrics_consistent_with_metadata():
    if not ZDM_METRICS.prometheus_available:
        pytest.skip("prometheus_client 未安装，跳过 REGISTRY 一致性检查")
    model = ZeroDataModel(dim=8, seed=42, enable_multiagent=True)
    # 跑两周期，使指标值非平凡
    model.think(_input())
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
    # 两周期后通信/文化计数 >= 2
    assert ma["communication_usage"] >= 2
    assert ma["culture_generations"] >= 2


# --------------------------------------------------------------------------- #
# 6. hook 部分组件异常仍不拖垮 think()（通信通道异常场景）
# --------------------------------------------------------------------------- #


def test_communication_failure_does_not_crash_think():
    model = ZeroDataModel(dim=8, seed=42, enable_multiagent=True)

    def boom(*_a, **_k):
        raise ValueError("comm boom")

    model.multiagent_communication.record_usage = boom  # type: ignore[method-assign]
    sig = model.think(_input())
    # hook 失败 → phase7_multiagent 未写入，但 think() 仍返回有效信号
    upgrades = sig.metadata.get("cognitive_upgrades", {})
    assert "phase7_multiagent" not in upgrades
    assert sig.data is not None
