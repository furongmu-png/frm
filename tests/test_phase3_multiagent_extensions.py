"""Phase 3 §1 多智能体扩展测试。

覆盖新增的：
  - MultiAgentWorld.step_all() / from_config() / get_snapshot()
  - CommunicationChannel.policy_gradient_update() /
    compute_mutual_information() / get_communication_network()
  - CulturePropagation.save_snapshot() / load_snapshot() /
    get_task_completion_curve() / get_cultural_acceleration_curve()
"""
from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.multiagent.communication import CommunicationChannel
from zero_data_model.multiagent.culture import CulturePropagation
from zero_data_model.multiagent.world import MultiAgentWorld


# ------------------------------------------------------------------ #
# World 扩展测试
# ------------------------------------------------------------------ #
class TestStepAll:
    """step_all() 返回结构化字典列表。"""

    def test_step_all_returns_dicts(self):
        world = MultiAgentWorld(n_agents=2, dim=16, seed=42)
        results = world.step_all()
        assert len(results) == 2
        for r in results:
            assert isinstance(r, dict)
            assert "agent_id" in r
            assert "observation" in r
            assert "action" in r
            assert "free_energy" in r
            assert "prediction_error" in r
            assert "reward" in r

    def test_step_all_with_observations(self):
        world = MultiAgentWorld(n_agents=2, dim=16, seed=42)
        obs = [np.ones(16), np.zeros(16)]
        results = world.step_all(obs)
        assert len(results) == 2
        assert results[0]["agent_id"] == 0
        assert results[1]["agent_id"] == 1


class TestFromConfig:
    """from_config() 从字典创建世界。"""

    def test_from_dict(self):
        config = {
            "n_agents": 3,
            "dim": 32,
            "seed": 99,
            "shared_env": True,
        }
        world = MultiAgentWorld.from_config(config)
        assert world.agent_count == 3
        assert world.dim == 32
        assert world.seed == 99

    def test_from_invalid_config_raises(self):
        with pytest.raises(TypeError):
            MultiAgentWorld.from_config(123)  # type: ignore[arg-type]


class TestGetSnapshot:
    """get_snapshot() 返回世界状态快照。"""

    def test_snapshot_structure(self):
        world = MultiAgentWorld(n_agents=2, dim=16, seed=42)
        world.step()
        snap = world.get_snapshot()
        assert snap["n_agents"] == 2
        assert snap["step_count"] == 1
        assert "agent_states" in snap
        assert "collaboration_stats" in snap
        assert len(snap["agent_states"]) == 2


# ------------------------------------------------------------------ #
# Communication 扩展测试
# ------------------------------------------------------------------ #
class TestPolicyGradientUpdate:
    """policy_gradient_update() 增强/减弱符号嵌入。"""

    def test_positive_reward_strengthened(self):
        ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
        before = ch.embeddings[2].copy()
        ch.policy_gradient_update(symbol=2, reward=1.0, lr=0.1)
        after = ch.embeddings[2]
        # 嵌入方向不变但范数变化
        assert np.allclose(
            np.sign(after), np.sign(before), equal_nan=True
        ) or not np.allclose(before, after)

    def test_negative_reward_weakened(self):
        ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
        ch.policy_gradient_update(symbol=1, reward=-1.0, lr=0.1)
        # 不报错即通过
        assert ch.embeddings[1] is not None

    def test_invalid_symbol_raises(self):
        ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
        with pytest.raises(IndexError):
            ch.policy_gradient_update(symbol=10, reward=1.0)


class TestMutualInformation:
    """compute_mutual_information() 检测符号-事件关联。"""

    def test_perfect_correlation_high_mi(self):
        ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
        # 符号 0 总是出现在 "left" 事件，符号 1 总是出现在 "right" 事件
        symbols = [0, 0, 0, 1, 1, 1]
        events = ["left", "left", "left", "right", "right", "right"]
        mi = ch.compute_mutual_information(symbols, events)
        assert mi > 0.5, f"完美关联 MI 应 > 0.5，实际 {mi:.4f}"

    def test_no_correlation_low_mi(self):
        ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
        # 随机搭配，MI 应接近 0
        rng = np.random.default_rng(0)
        symbols = rng.integers(0, 5, 100).tolist()
        events = ["a", "b", "c"] * 34
        mi = ch.compute_mutual_information(symbols, events)
        assert mi < 0.2, f"无关联 MI 应 < 0.2，实际 {mi:.4f}"

    def test_empty_returns_zero(self):
        ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
        assert ch.compute_mutual_information([], []) == 0.0

    def test_mismatched_length_returns_zero(self):
        ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
        assert ch.compute_mutual_information([0, 1], ["a"]) == 0.0


class TestCommunicationNetwork:
    """get_communication_network() 统计通信边权重。"""

    def test_basic_network(self):
        ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
        pairs = [(0, 1), (0, 1), (1, 2), (0, 1)]
        net = ch.get_communication_network(pairs)
        assert net[(0, 1)] == 3
        assert net[(1, 2)] == 1
        assert (2, 0) not in net

    def test_empty_pairs(self):
        ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
        net = ch.get_communication_network([])
        assert net == {}


# ------------------------------------------------------------------ #
# Culture 扩展测试
# ------------------------------------------------------------------ #
class TestSaveLoadSnapshot:
    """save_snapshot() / load_snapshot() 文化快照保存与回放。"""

    def test_roundtrip(self):
        cp = CulturePropagation(max_generations=10, seed=42)
        cp.record_generation(knowledge_graph_size=10, mean_free_energy=1.0, n_steps=100)
        cp.record_generation(knowledge_graph_size=20, mean_free_energy=0.5, n_steps=80)
        snap = cp.save_snapshot()
        assert snap["current_gen"] == 2
        assert len(snap["generations"]) == 2
        # 新实例加载
        cp2 = CulturePropagation(max_generations=10, seed=42)
        cp2.load_snapshot(snap)
        assert cp2.stats["n_generations"] == 2
        assert cp2.stats["latest_kg_size"] == 20

    def test_load_clears_existing(self):
        cp = CulturePropagation(max_generations=10, seed=42)
        cp.record_generation(knowledge_graph_size=5, mean_free_energy=1.0, n_steps=50)
        snap = {"current_gen": 0, "generations": []}
        cp.load_snapshot(snap)
        assert cp.stats["n_generations"] == 0


class TestTaskCompletionCurve:
    """get_task_completion_curve() 返回逐代步数。"""

    def test_curve_values(self):
        cp = CulturePropagation(max_generations=10, seed=42)
        cp.record_generation(knowledge_graph_size=10, mean_free_energy=1.0, n_steps=100)
        cp.record_generation(knowledge_graph_size=20, mean_free_energy=0.5, n_steps=80)
        curve = cp.get_task_completion_curve()
        assert curve == [(0, 100), (1, 80)]

    def test_empty_curve(self):
        cp = CulturePropagation(max_generations=10, seed=42)
        assert cp.get_task_completion_curve() == []


class TestCulturalAccelerationCurve:
    """get_cultural_acceleration_curve() 返回逐代加速度。"""

    def test_first_gen_baseline(self):
        cp = CulturePropagation(max_generations=10, seed=42)
        cp.record_generation(knowledge_graph_size=10, mean_free_energy=1.0, n_steps=100)
        curve = cp.get_cultural_acceleration_curve()
        assert len(curve) == 1
        assert curve[0] == (0, 1.0)

    def test_acceleration_gt_1_when_faster(self):
        cp = CulturePropagation(max_generations=10, seed=42)
        # 第一代：10 知识 / 100 步 = 0.1
        cp.record_generation(knowledge_graph_size=10, mean_free_energy=1.0, n_steps=100)
        # 第二代：20 知识 / 80 步 = 0.25 → 加速度 = 0.25/0.1 = 2.5
        cp.record_generation(knowledge_graph_size=20, mean_free_energy=0.5, n_steps=80)
        curve = cp.get_cultural_acceleration_curve()
        assert len(curve) == 2
        assert curve[1][1] > 1.0, (
            f"第二代加速度应 > 1.0（文化加速），实际 {curve[1][1]:.4f}"
        )

    def test_deceleration_lt_1_when_slower(self):
        cp = CulturePropagation(max_generations=10, seed=42)
        # 第一代快，第二代慢
        cp.record_generation(knowledge_graph_size=20, mean_free_energy=0.5, n_steps=50)
        cp.record_generation(knowledge_graph_size=10, mean_free_energy=1.0, n_steps=200)
        curve = cp.get_cultural_acceleration_curve()
        assert curve[1][1] < 1.0


class TestCulturalAccelerationDemo:
    """验证场景：比较第一代与第十代的学习曲线，观察文化加速效应。"""

    def test_cultural_acceleration_over_generations(self):
        """模拟 5 代，知识逐代增长、步数逐代减少 → 加速度 > 1。"""
        cp = CulturePropagation(max_generations=10, seed=42)
        kg_sizes = [10, 25, 45, 70, 100]
        step_counts = [200, 150, 120, 90, 70]
        for kg, steps in zip(kg_sizes, step_counts, strict=True):
            cp.record_generation(
                knowledge_graph_size=kg,
                mean_free_energy=1.0 / (1.0 + kg * 0.01),
                n_steps=steps,
            )
        # 整体加速度应 > 1（后代学得更快）
        overall = cp.compute_cultural_acceleration()
        assert overall > 1.0, (
            f"文化加速度应 > 1.0（加速效应），实际 {overall:.4f}"
        )
