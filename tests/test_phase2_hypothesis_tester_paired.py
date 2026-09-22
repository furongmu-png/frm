# tests/test_phase2_hypothesis_tester_paired.py
"""第二阶段 §3.2 配对假设与因果图更新单元测试。

验证 HypothesisTester 的扩展方法：
- generate_paired_hypotheses(): 为不确定边生成 H0 vs H1
- update_causal_graph(): 用贝叶斯因子更新转移矩阵

关键修复点：
- 不确定边（|w| < threshold）才生成配对
- 配对假设共享 (cause_var, effect_var)
- BF > 1 增强 / BF < 1 减弱 / BF = 1 不变
- 越界索引抛 ValueError
- 输入矩阵不被修改（返回副本）
"""
from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.experiment.hypothesis_tester import (
    Hypothesis,
    HypothesisTester,
)


# --------------------------------------------------------------------------- #
# 配对假设生成
# --------------------------------------------------------------------------- #
class TestGeneratePairedHypotheses:
    """测试 generate_paired_hypotheses()。"""

    def test_returns_pairs_for_uncertain_edges(self) -> None:
        """不确定边（|w| < threshold）生成 H0+H1 配对。"""
        ht = HypothesisTester(edge_threshold=0.05, seed=42)
        # 只有 m[0,1]=0.01 < 0.05 是不确定边；其余非对角 > 0.2（强边）。
        m = np.array([
            [0.0, 0.01, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ])
        pairs = ht.generate_paired_hypotheses(m)
        assert isinstance(pairs, list)
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair["cause_var"] == 0
        assert pair["effect_var"] == 1
        # H0: 不导致 / H1: 导致
        assert pair["h0"].direction == "no_effect"
        assert pair["h1"].direction == "causes"
        # 共享 cause/effect
        assert pair["h0"].cause_var == pair["h1"].cause_var == 0
        assert pair["h0"].effect_var == pair["h1"].effect_var == 1

    def test_skips_strong_edges(self) -> None:
        """强边（|w| > 0.2）不生成配对。"""
        ht = HypothesisTester(edge_threshold=0.05, seed=42)
        # 所有非对角元素都 > 0.2（强边）→ 无不确定边 → 无配对。
        m = np.array([
            [0.0, 0.5, 0.6],
            [0.5, 0.0, 0.7],
            [0.5, 0.5, 0.0],
        ])
        pairs = ht.generate_paired_hypotheses(m)
        assert pairs == []

    def test_skips_diagonal(self) -> None:
        """对角元素被跳过。"""
        ht = HypothesisTester(edge_threshold=0.05, seed=42)
        # 对角线小（但应跳过），非对角大。
        m = np.array([
            [0.01, 0.5],
            [0.5, 0.01],
        ])
        pairs = ht.generate_paired_hypotheses(m)
        assert pairs == []

    def test_max_pairs_respected(self) -> None:
        """max_pairs 限制返回数量。"""
        ht = HypothesisTester(edge_threshold=0.05, seed=42)
        # 4x4 全零非对角 → 12 个不确定边
        m = np.zeros((4, 4))
        pairs = ht.generate_paired_hypotheses(m, max_pairs=3)
        assert len(pairs) == 3

    def test_appends_to_hypotheses_deque(self) -> None:
        """生成的假设被加入 hypotheses 历史。"""
        ht = HypothesisTester(edge_threshold=0.05, seed=42)
        # 只有 m[0,1]=0.01 是不确定边；m[1,0]=0.5 是强边（不生成配对）。
        m = np.array([[0.0, 0.01], [0.5, 0.0]])
        ht.generate_paired_hypotheses(m)
        # H0 + H1 = 2 个假设
        assert len(ht.hypotheses) == 2

    def test_none_transition_raises_type_error(self) -> None:
        """核心修复点：None 抛 TypeError。"""
        ht = HypothesisTester(seed=42)
        with pytest.raises(TypeError):
            ht.generate_paired_hypotheses(None)  # type: ignore[arg-type]

    def test_non_square_raises_value_error(self) -> None:
        ht = HypothesisTester(seed=42)
        with pytest.raises(ValueError):
            ht.generate_paired_hypotheses(np.zeros((2, 3)))

    def test_pair_structure(self) -> None:
        """每个 pair 字典结构完整。"""
        ht = HypothesisTester(edge_threshold=0.05, seed=42)
        # 只有 m[0,1]=0.02 是不确定边；m[1,0]=0.5 是强边。
        m = np.array([[0.0, 0.02], [0.5, 0.0]])
        pairs = ht.generate_paired_hypotheses(m)
        p = pairs[0]
        assert set(p.keys()) == {"h0", "h1", "cause_var", "effect_var"}
        assert isinstance(p["h0"], Hypothesis)
        assert isinstance(p["h1"], Hypothesis)


# --------------------------------------------------------------------------- #
# 因果图更新
# --------------------------------------------------------------------------- #
class TestUpdateCausalGraph:
    """测试 update_causal_graph()。"""

    def test_bf_greater_than_one_strengthens_edge(self) -> None:
        """BF > 1 → 增强边。"""
        ht = HypothesisTester(seed=42)
        m = np.array([[0.0, 0.0], [0.0, 0.0]])
        new_m = ht.update_causal_graph(m, cause_var=0, effect_var=1,
                                       bayes_factor=10.0)
        # delta = log(10) * 0.1 ≈ 0.23
        assert new_m[0, 1] > 0.0
        assert abs(new_m[0, 1] - np.log(10.0) * 0.1) < 1e-6

    def test_bf_less_than_one_weakens_edge(self) -> None:
        """BF < 1 → 减弱边。"""
        ht = HypothesisTester(seed=42)
        m = np.array([[0.0, 0.5], [0.0, 0.0]])
        new_m = ht.update_causal_graph(m, cause_var=0, effect_var=1,
                                       bayes_factor=0.1)
        # delta = log(0.1) * 0.1 ≈ -0.23
        assert new_m[0, 1] < 0.5

    def test_bf_equal_one_no_change(self) -> None:
        """BF = 1 → 无变化。"""
        ht = HypothesisTester(seed=42)
        m = np.array([[0.0, 0.5], [0.0, 0.0]])
        new_m = ht.update_causal_graph(m, cause_var=0, effect_var=1,
                                       bayes_factor=1.0)
        assert abs(new_m[0, 1] - 0.5) < 1e-9

    def test_does_not_modify_input(self) -> None:
        """返回新数组，不修改输入。"""
        ht = HypothesisTester(seed=42)
        m = np.array([[0.0, 0.5], [0.0, 0.0]])
        m_copy = m.copy()
        ht.update_causal_graph(m, cause_var=0, effect_var=1, bayes_factor=10.0)
        # 原矩阵未被修改。
        np.testing.assert_array_equal(m, m_copy)

    def test_clips_to_range(self) -> None:
        """更新值被 clip 到 [-1, 1]。"""
        ht = HypothesisTester(seed=42)
        m = np.array([[0.0, 0.9], [0.0, 0.0]])
        # BF 极大 → delta 大，但应被 clip 到 1.0
        new_m = ht.update_causal_graph(m, cause_var=0, effect_var=1,
                                       bayes_factor=1e10)
        assert new_m[0, 1] <= 1.0
        assert new_m[0, 1] == 1.0  # 0.9 + large delta → clip 到 1.0

    def test_out_of_range_cause_var_raises(self) -> None:
        ht = HypothesisTester(seed=42)
        m = np.zeros((2, 2))
        with pytest.raises(ValueError):
            ht.update_causal_graph(m, cause_var=5, effect_var=1,
                                   bayes_factor=2.0)

    def test_out_of_range_effect_var_raises(self) -> None:
        ht = HypothesisTester(seed=42)
        m = np.zeros((2, 2))
        with pytest.raises(ValueError):
            ht.update_causal_graph(m, cause_var=0, effect_var=-1,
                                   bayes_factor=2.0)

    def test_non_square_raises_value_error(self) -> None:
        ht = HypothesisTester(seed=42)
        with pytest.raises(ValueError):
            ht.update_causal_graph(np.zeros((2, 3)), cause_var=0,
                                   effect_var=1, bayes_factor=2.0)

    def test_returns_ndarray(self) -> None:
        ht = HypothesisTester(seed=42)
        m = np.zeros((2, 2))
        new_m = ht.update_causal_graph(m, cause_var=0, effect_var=1,
                                       bayes_factor=2.0)
        assert isinstance(new_m, np.ndarray)

    def test_other_edges_unchanged(self) -> None:
        """更新只影响指定边，其他边不变。"""
        ht = HypothesisTester(seed=42)
        m = np.array([[0.0, 0.3, 0.5],
                      [0.0, 0.0, 0.7],
                      [0.0, 0.0, 0.0]])
        new_m = ht.update_causal_graph(m, cause_var=0, effect_var=1,
                                       bayes_factor=10.0)
        # m[0,2] 和 m[1,2] 应保持不变。
        assert abs(new_m[0, 2] - 0.5) < 1e-9
        assert abs(new_m[1, 2] - 0.7) < 1e-9
        # 只有 m[0,1] 被更新。
        assert new_m[0, 1] != 0.3


# --------------------------------------------------------------------------- #
# 集成：配对假设 → 实验设计 → 更新因果图
# --------------------------------------------------------------------------- #
class TestIntegrationFlow:
    """测试完整的"假设→实验→更新"流程。"""

    def test_full_flow_updates_causal_graph(self) -> None:
        """完整流程：生成配对 → 设计实验 → 计算BF → 更新因果图。"""
        ht = HypothesisTester(edge_threshold=0.05, seed=42)
        m = np.array([[0.0, 0.01, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])
        # 1. 生成配对假设
        pairs = ht.generate_paired_hypotheses(m)
        assert len(pairs) >= 1
        pair = pairs[0]
        # 2. 设计实验（固定 cause_var）
        design = ht.design_experiment(pair["h1"])
        assert design["fix_var"] == pair["cause_var"]
        assert design["observe_var"] == pair["effect_var"]
        # 3. 用低方差数据更新 H1（支持 causes）
        ht.update_with_data(pair["h1"], np.array([0.5, 0.5, 0.5, 0.5]))
        assert pair["h1"].supported is True
        assert pair["h1"].bayes_factor == 5.0
        # 4. 用 BF 更新因果图
        new_m = ht.update_causal_graph(
            m, pair["cause_var"], pair["effect_var"],
            pair["h1"].bayes_factor,
        )
        # BF=5 → 增强 m[0,1]
        assert new_m[0, 1] > m[0, 1]
