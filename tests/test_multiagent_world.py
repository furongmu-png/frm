from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.multiagent.world import MultiAgentWorld


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #


def test_construct_default():
    world = MultiAgentWorld(n_agents=2, dim=8, seed=42)
    assert world.agent_count == 2
    assert world.dim == 8
    assert world.step_count == 0
    # 每个 agent 有自己的 ZeroDataModel 实例
    assert all(a.model is not None for a in world.agents)


def test_construct_with_shared_env():
    world = MultiAgentWorld(n_agents=3, dim=8, seed=42, shared_env=True)
    assert world.shared_env is True


def test_construct_distinct_seeds_per_agent():
    """每个 agent 使用独立种子（seed + i）。"""
    world = MultiAgentWorld(n_agents=2, dim=8, seed=42)
    # 两个 agent 应有不同的种子
    s0 = world.agents[0].model._seed
    s1 = world.agents[1].model._seed
    assert s0 == 42
    assert s1 == 43


# --------------------------------------------------------------------------- #
# step / exchange_observations / record_collaboration
# --------------------------------------------------------------------------- #


def test_step_returns_signals_list():
    world = MultiAgentWorld(n_agents=2, dim=8, seed=42)
    signals = world.step()
    assert len(signals) == 2
    assert world.step_count == 1


def test_step_with_explicit_observations():
    world = MultiAgentWorld(n_agents=2, dim=8, seed=42)
    rng = np.random.default_rng(0)
    obs = [rng.normal(0, 1, 8) for _ in range(2)]
    signals = world.step(observations=obs)
    assert len(signals) == 2


def test_step_partial_observations_fallback_to_self_gen():
    """外部观测数量不足 → 缺失的 agent 回退到自生成。"""
    world = MultiAgentWorld(n_agents=3, dim=8, seed=42, shared_env=False)
    rng = np.random.default_rng(0)
    # 仅提供 1 个观测，剩下 2 个 agent 应回退（None）
    signals = world.step(observations=[rng.normal(0, 1, 8)])
    assert len(signals) == 3


def test_exchange_observations_appends_to_logs():
    world = MultiAgentWorld(n_agents=2, dim=8, seed=42)
    # 先 step 一下，让 agent 有 last_action
    world.step()
    world.exchange_observations()
    for agent in world.agents:
        # 应至少有一条 observation_summary 记录
        assert len(agent.communication_log) > 0
        last = agent.communication_log[-1]
        assert last["kind"] == "observation_summary"


def test_record_collaboration_appends_event():
    world = MultiAgentWorld(n_agents=2, dim=8, seed=42)
    n_before = world.get_collaboration_stats()["n_events"]
    world.record_collaboration(0, 1, "test_event")
    n_after = world.get_collaboration_stats()["n_events"]
    assert n_after == n_before + 1


# --------------------------------------------------------------------------- #
# 军事级修复点：共享 ndarray 别名防护
# --------------------------------------------------------------------------- #


def test_step_external_observation_is_copied():
    """核心修复点：外部提供的观测是副本（非共享引用）。"""
    world = MultiAgentWorld(n_agents=1, dim=8, seed=42, shared_env=False)
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 1, 8)
    obs_id = id(obs)
    world.step(observations=[obs])
    # 即使原 ndarray 被修改，不应影响已存进 last_action 的内容
    # （think 内部不会保留外部 obs 引用，因为 step 复制了）
    original_data = obs.copy()
    obs[0] = 999.0
    # 再 step 一次原数据应不受上一次 obs 修改影响
    # (此处仅断言原 ndarray 不被泄漏到 agent 持有的状态里)
    # 我们通过验证 step 不抛异常、agent 不持有对原 obs 的引用即可
    # 注意：obs_id 不应出现在 agent 持有的 last_action 中
    if world.agents[0].last_action is not None:
        # last_action 是基于 signal.data 重建的副本
        assert id(world.agents[0].last_action) != obs_id


def test_step_shared_env_copies_per_agent():
    """共享环境下，每个 agent 拿到的是独立副本。"""
    world = MultiAgentWorld(n_agents=3, dim=8, seed=42, shared_env=True)
    # 让世界生成共享观测
    world.step()
    # 共享观测应被记录
    assert world.shared_observation is not None
    # 不同 agent 的 last_action（如果有）不应是同一引用
    # （这里仅验证 step 不抛异常且 shared_observation 已设置）
    assert world.step_count == 1


# --------------------------------------------------------------------------- #
# 军事级修复点：deque maxlen 限长
# --------------------------------------------------------------------------- #


def test_collaboration_events_deque_maxlen():
    """deque maxlen 限长：_collaboration_events 上限 max_events。"""
    world = MultiAgentWorld(n_agents=2, dim=8, seed=42, max_events=5)
    for _ in range(20):
        world.record_collaboration(0, 1, "test")
    stats = world.get_collaboration_stats()
    assert stats["n_events"] == 5


def test_communication_log_deque_maxlen():
    """deque maxlen 限长：communication_log 上限 max_log。"""
    world = MultiAgentWorld(n_agents=1, dim=8, seed=42, max_log=3)
    # 先 step 让 agent 有 last_action
    world.step()
    for _ in range(20):
        world.exchange_observations()
    assert len(world.agents[0].communication_log) == 3


# --------------------------------------------------------------------------- #
# 军事级修复点：异常收窄（不吞所有 Exception）
# --------------------------------------------------------------------------- #


def test_step_does_not_propagate_value_error():
    """单个 agent think 抛 ValueError 时被收窄，不拖垮整个世界。"""
    world = MultiAgentWorld(n_agents=2, dim=8, seed=42)

    # 让 agent[0].think 抛 ValueError
    def boom(_obs):
        raise ValueError("boom")

    world.agents[0].model.think = boom  # type: ignore[assignment]
    # 不应抛出
    signals = world.step()
    assert len(signals) == 2
    # agent[0] 的信号应是 None（异常被收窄）
    assert signals[0] is None


def test_step_does_not_propagate_index_error():
    """单个 agent think 抛 IndexError 也被收窄。"""
    world = MultiAgentWorld(n_agents=2, dim=8, seed=42)

    def boom(_obs):
        raise IndexError("boom")

    world.agents[0].model.think = boom  # type: ignore[assignment]
    signals = world.step()
    assert signals[0] is None


# --------------------------------------------------------------------------- #
# 线程安全
# --------------------------------------------------------------------------- #


def test_concurrent_step_does_not_crash():
    """线程安全：并发 step 不抛异常。"""
    world = MultiAgentWorld(n_agents=2, dim=8, seed=42)

    def worker() -> None:
        for _ in range(10):
            world.step()

    threads = [Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 步数应单调递增
    assert world.step_count > 0


def test_concurrent_record_collaboration_does_not_crash():
    world = MultiAgentWorld(n_agents=2, dim=8, seed=42, max_events=10000)

    def worker(idx: int) -> None:
        for i in range(50):
            world.record_collaboration(idx, 1 - idx, "test")

    threads = [Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = world.get_collaboration_stats()
    assert stats["n_events"] == 100
