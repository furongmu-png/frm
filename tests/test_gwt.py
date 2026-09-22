"""GWT 模块单元测试：AttentionSelector / WorkspaceBroadcaster / PhiCalculator。"""

from __future__ import annotations

import numpy as np
import pytest

from src.gwt.attention_selector import AttentionSelector, BroadcastCandidate
from src.gwt.workspace_broadcaster import WorkspaceBroadcaster
from src.gwt.phi_calculator import PhiCalculator


# ------------------------------------------------------------------ #
# AttentionSelector
# ------------------------------------------------------------------ #
class TestAttentionSelector:
    def _candidates(self, n: int = 3) -> list[BroadcastCandidate]:
        rng = np.random.default_rng(0)
        return [
            BroadcastCandidate(
                module_name=f"mod_{i}",
                vector=rng.standard_normal(8),
                attention_request=float(i + 1),  # mod_2 请求最大
                confidence=0.5 + 0.1 * i,
            )
            for i in range(n)
        ]

    def test_argmax_winner_is_highest_request(self):
        """argmax 模式：最高请求的模块胜出。"""
        sel = AttentionSelector(dim=8, temperature=1.0, sample=False)
        candidates = self._candidates(3)
        result = sel.compete(candidates)
        assert result.winner is not None
        assert result.winner.module_name == "mod_2"

    def test_probabilities_sum_to_one(self):
        """softmax 概率和为 1。"""
        sel = AttentionSelector(dim=8, sample=False)
        result = sel.compete(self._candidates(4))
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6

    def test_low_temperature_sharpens_winner(self):
        """低温使胜者概率更突出。"""
        sel_low = AttentionSelector(dim=8, temperature=0.1, sample=False)
        sel_high = AttentionSelector(dim=8, temperature=10.0, sample=False)
        candidates = self._candidates(3)
        r_low = sel_low.compete(candidates)
        r_high = sel_high.compete(candidates)
        # 低温下胜者概率应更高
        winner_prob_low = r_low.probabilities["mod_2"]
        winner_prob_high = r_high.probabilities["mod_2"]
        assert winner_prob_low > winner_prob_high

    def test_broadcast_vector_normalized_and_aligned(self):
        """广播向量 L2 归一化并对齐到 dim。"""
        sel = AttentionSelector(dim=16, sample=False)
        result = sel.compete(self._candidates(2))
        assert result.broadcast is not None
        assert result.broadcast.shape == (16,)
        assert abs(np.linalg.norm(result.broadcast) - 1.0) < 1e-6 or np.linalg.norm(
            result.broadcast
        ) < 1e-8

    def test_empty_candidates_returns_none(self):
        """边界：无候选返回 winner=None。"""
        sel = AttentionSelector(dim=8)
        result = sel.compete([])
        assert result.winner is None

    def test_inhibition_applied_to_losers(self):
        """败者被短暂抑制。"""
        sel = AttentionSelector(dim=8, inhibition_steps=3, sample=False)
        sel.compete(self._candidates(3))
        # mod_0, mod_1 应被抑制，mod_2 不应被抑制
        assert sel.get_inhibition("mod_0") > 0
        assert sel.get_inhibition("mod_1") > 0
        assert sel.get_inhibition("mod_2") == 0

    def test_winner_history_recorded(self):
        """胜者历史被记录。"""
        sel = AttentionSelector(dim=8, sample=False)
        for _ in range(5):
            sel.compete(self._candidates(3))
        assert len(sel.winner_history) == 5
        assert all(w == "mod_2" for w in sel.winner_history)

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            AttentionSelector(dim=0)
        with pytest.raises(ValueError):
            AttentionSelector(dim=8, temperature=0)
        with pytest.raises(ValueError):
            AttentionSelector(dim=8, inhibition_steps=-1)

    def test_snapshot_fields(self):
        sel = AttentionSelector(dim=8, sample=False)
        sel.compete(self._candidates(3))
        snap = sel.snapshot()
        assert "recent_winners" in snap
        assert "winner_distribution" in snap
        assert "inhibited_modules" in snap

    def test_sample_mode_uses_rng(self):
        """采样模式：固定 seed 产生确定性结果。"""
        sel1 = AttentionSelector(dim=8, sample=True, seed=42)
        sel2 = AttentionSelector(dim=8, sample=True, seed=42)
        r1 = sel1.compete(self._candidates(3))
        r2 = sel2.compete(self._candidates(3))
        assert r1.winner.module_name == r2.winner.module_name


# ------------------------------------------------------------------ #
# WorkspaceBroadcaster
# ------------------------------------------------------------------ #
class TestWorkspaceBroadcaster:
    def test_register_and_broadcast(self):
        """注册模块收到广播。"""
        sel = AttentionSelector(dim=8, sample=False)
        wb = WorkspaceBroadcaster(sel)
        received: list[tuple[np.ndarray, float]] = []

        def cb(vec, lr):
            received.append((vec.copy(), lr))

        wb.register("mod_a", cb)
        candidates = [
            BroadcastCandidate("mod_a", np.ones(8), attention_request=1.0, confidence=0.8)
        ]
        wb.broadcast(candidates)
        assert len(received) == 1
        # 学习率受置信度调制
        assert received[0][1] > 0

    def test_high_confidence_strong_alignment(self):
        """高置信→强对齐（高学习率）。"""
        sel = AttentionSelector(dim=8, sample=False)
        wb = WorkspaceBroadcaster(sel, alignment_lr=0.5)
        lrs = []

        wb.register("m", lambda v, lr: lrs.append(lr))
        wb.broadcast(
            [BroadcastCandidate("m", np.ones(8), attention_request=1.0, confidence=1.0)]
        )
        wb.broadcast(
            [BroadcastCandidate("m", np.ones(8), attention_request=1.0, confidence=0.2)]
        )
        assert lrs[0] > lrs[1]

    def test_inhibited_module_reduced_lr(self):
        """被抑制模块学习率降低。"""
        sel = AttentionSelector(dim=8, sample=False, inhibition_steps=5)
        wb = WorkspaceBroadcaster(sel, alignment_lr=0.5)
        lrs = []
        wb.register("loser", lambda v, lr: lrs.append(lr))
        # winner 是 winner_mod，loser 被抑制
        wb.broadcast(
            [
                BroadcastCandidate(
                    "loser", np.ones(8), attention_request=0.1, confidence=1.0
                ),
                BroadcastCandidate(
                    "winner_mod", np.ones(8), attention_request=2.0, confidence=1.0
                ),
            ]
        )
        assert lrs[-1] < 0.5 * 1.0  # 受抑制

    def test_unregister(self):
        sel = AttentionSelector(dim=8, sample=False)
        wb = WorkspaceBroadcaster(sel)
        wb.register("x", lambda v, lr: None)
        assert "x" in wb.registered_modules
        wb.unregister("x")
        assert "x" not in wb.registered_modules

    def test_global_timestamp_advances(self):
        """全局时间步每次广播递增。"""
        sel = AttentionSelector(dim=8, sample=False)
        wb = WorkspaceBroadcaster(sel)
        wb.broadcast([BroadcastCandidate("m", np.ones(8), 1.0)])
        wb.broadcast([BroadcastCandidate("m", np.ones(8), 1.0)])
        assert wb.global_timestamp == 2

    def test_callback_failure_isolated(self):
        """单模块 callback 失败不影响其他模块。"""
        sel = AttentionSelector(dim=8, sample=False)
        wb = WorkspaceBroadcaster(sel)
        received = []
        wb.register("bad", lambda v, lr: (_ for _ in ()).throw(RuntimeError("boom")))
        wb.register("good", lambda v, lr: received.append(v))
        wb.broadcast([BroadcastCandidate("bad", np.ones(8), 1.0)])
        assert len(received) == 1

    def test_snapshot_fields(self):
        sel = AttentionSelector(dim=8, sample=False)
        wb = WorkspaceBroadcaster(sel)
        wb.register("m", lambda v, lr: None)
        wb.broadcast([BroadcastCandidate("m", np.ones(8), 1.0, confidence=0.7)])
        snap = wb.snapshot()
        assert "last_winner" in snap
        assert "last_confidence" in snap
        assert "last_broadcast_3d" in snap
        assert "selector" in snap

    def test_invalid_lr_raises(self):
        with pytest.raises(ValueError):
            WorkspaceBroadcaster(AttentionSelector(dim=8), alignment_lr=0)
        with pytest.raises(ValueError):
            WorkspaceBroadcaster(AttentionSelector(dim=8), alignment_lr=1.5)


# ------------------------------------------------------------------ #
# PhiCalculator
# ------------------------------------------------------------------ #
class TestPhiCalculator:
    def test_phi_zero_with_insufficient_history(self):
        """历史不足时 Φ=0。"""
        calc = PhiCalculator(n_modules=3, history_window=10)
        assert calc.compute_phi() == 0.0

    def test_phi_increases_with_correlated_modules(self):
        """相关模块产生更高 Φ（整合信息）。"""
        rng = np.random.default_rng(0)
        calc_corr = PhiCalculator(n_modules=3, history_window=30)
        calc_indep = PhiCalculator(n_modules=3, history_window=30)
        # 相关：所有模块共享同一信号
        for _ in range(30):
            shared = rng.standard_normal(8)
            calc_corr.record([shared, shared + 0.1 * rng.standard_normal(8), shared * 0.9])
        # 独立：各模块独立噪声
        for _ in range(30):
            calc_indep.record(
                [rng.standard_normal(8), rng.standard_normal(8), rng.standard_normal(8)]
            )
        phi_corr = calc_corr.compute_phi()
        phi_indep = calc_indep.compute_phi()
        assert phi_corr >= phi_indep

    def test_is_conscious_threshold(self):
        """清醒/睡眠判断。"""
        calc = PhiCalculator(n_modules=2, history_window=10, sleep_threshold=0.5)
        rng = np.random.default_rng(0)
        for _ in range(10):
            s = rng.standard_normal(4)
            calc.record([s, s])
        calc.compute_phi()
        # 相关模块应产生较高 Φ
        if calc.last_phi > 0.5:
            assert calc.is_conscious()
        else:
            assert not calc.is_conscious()

    def test_partition_recorded(self):
        """划分被记录。"""
        calc = PhiCalculator(n_modules=3, history_window=10)
        rng = np.random.default_rng(0)
        for _ in range(10):
            calc.record([rng.standard_normal(4) for _ in range(3)])
        calc.compute_phi()
        assert calc.last_partition is not None

    def test_phi_history_capped(self):
        """Φ 历史上限。"""
        calc = PhiCalculator(n_modules=2, history_window=5)
        for _ in range(600):
            calc.record([np.array([1.0]), np.array([1.0])])
            calc.compute_phi()
        assert len(calc.phi_history) <= 500

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            PhiCalculator(n_modules=0)
        with pytest.raises(ValueError):
            PhiCalculator(n_modules=2, history_window=1)

    def test_snapshot_fields(self):
        calc = PhiCalculator(n_modules=2, history_window=10)
        calc.compute_phi()
        snap = calc.snapshot()
        assert "phi" in snap
        assert "is_conscious" in snap
        assert "phi_history" in snap
        assert "last_partition" in snap

    def test_module_count_mismatch_adapts(self):
        """边界：模块数不匹配时自动适配。"""
        calc = PhiCalculator(n_modules=3, history_window=10)
        # 只提供 2 个模块状态 → 填充
        calc.record([np.array([1.0]), np.array([2.0])])
        # 提供 4 个 → 截断
        calc.record([np.array([1.0])] * 4)
        # 不应崩溃
        assert True

    def test_configure_sleep_threshold(self):
        calc = PhiCalculator(n_modules=2, history_window=10)
        calc.configure(sleep_threshold=0.3)
        assert calc.sleep_threshold == 0.3

    def test_mismatched_history_lengths_no_crash(self):
        """B23: 不同模块历史长度不一致时不崩溃（截断到最短）。"""
        calc = PhiCalculator(n_modules=2, history_window=50)
        # mod_0 记录 10 步，mod_1 仅记录 3 步（模拟 record 调用不一致）
        for i in range(10):
            calc._module_histories[0].append(np.array([float(i)]))
        for i in range(3):
            calc._module_histories[1].append(np.array([float(i) * 2]))
        # 不应崩溃（之前会因 xi * xj 形状不匹配报 ValueError）
        phi = calc.compute_phi()
        assert isinstance(phi, float)
        assert np.isfinite(phi)


# ------------------------------------------------------------------ #
# 军事级审查回归测试
# ------------------------------------------------------------------ #
class TestMilitaryReviewRegressions:
    """第一轮军事级审查发现的 bug 回归测试。"""

    def _candidates(self, n: int) -> list[BroadcastCandidate]:
        return [
            BroadcastCandidate(
                module_name=f"mod_{i}",
                vector=np.random.default_rng(i).standard_normal(8),
                attention_request=float(i),
            )
            for i in range(n)
        ]

    def test_nan_attention_request_handled(self):
        """B24: NaN attention_request 不污染 softmax。"""
        sel = AttentionSelector(dim=8, sample=False)
        candidates = [
            BroadcastCandidate(
                module_name="nan_mod",
                vector=np.ones(8),
                attention_request=float("nan"),
            ),
            BroadcastCandidate(
                module_name="good_mod",
                vector=np.ones(8),
                attention_request=1.0,
            ),
        ]
        result = sel.compete(candidates)
        # nan 被替换为 0，good_mod 应胜出
        assert result.winner is not None
        assert result.winner.module_name == "good_mod"
        # 概率应有限
        for p in result.probabilities.values():
            assert np.isfinite(p)
            assert 0.0 <= p <= 1.0

    def test_inf_attention_request_handled(self):
        """B24: inf attention_request 不污染 softmax。"""
        sel = AttentionSelector(dim=8, sample=False)
        candidates = [
            BroadcastCandidate(
                module_name="inf_mod",
                vector=np.ones(8),
                attention_request=float("inf"),
            ),
            BroadcastCandidate(
                module_name="normal_mod",
                vector=np.ones(8),
                attention_request=1.0,
            ),
        ]
        result = sel.compete(candidates)
        assert result.winner is not None
        for p in result.probabilities.values():
            assert np.isfinite(p)

    def test_inhibition_no_off_by_one(self):
        """B4: inhibition_steps=3 应持续 3 步而非 2 步。

        修复前: 新设抑制在同一步被衰减，实际持续仅 2 步。
        修复后: 先衰减旧抑制再设新抑制，确保完整 3 步。

        场景: mod_2 胜出，mod_0/mod_1 被抑制。
        后续步不再让 mod_0/mod_1 参与（仅 mod_2 竞争），
        验证抑制逐步衰减 3→2→1→清除。
        """
        sel = AttentionSelector(dim=8, inhibition_steps=3, sample=False)
        # step 0: 三模块竞争，mod_2 胜
        sel.compete(self._candidates(3))
        assert sel.get_inhibition("mod_0") == 3, "抑制应从完整值 3 开始"
        assert sel.get_inhibition("mod_1") == 3

        # 后续步仅 mod_2 竞争（mod_0/mod_1 不参与）
        only_winner = [BroadcastCandidate(
            module_name="mod_2", vector=np.ones(8), attention_request=2.0
        )]
        # step 1: mod_0/mod_1 衰减 3→2
        sel.compete(only_winner)
        assert sel.get_inhibition("mod_0") == 2, f"step1 应为 2，实际 {sel.get_inhibition('mod_0')}"
        assert sel.get_inhibition("mod_1") == 2
        # step 2: 2→1
        sel.compete(only_winner)
        assert sel.get_inhibition("mod_0") == 1, f"step2 应为 1，实际 {sel.get_inhibition('mod_0')}"
        assert sel.get_inhibition("mod_1") == 1
        # step 3: 1→0→清除
        sel.compete(only_winner)
        assert sel.get_inhibition("mod_0") == 0, f"step3 应为 0（清除），实际 {sel.get_inhibition('mod_0')}"
        assert sel.get_inhibition("mod_1") == 0

    def test_sample_mode_probability_normalization(self):
        """B25: sample=True 时概率归一化防止 choice 报错。"""
        sel = AttentionSelector(dim=8, sample=True, seed=42)
        # 重复调用不应因浮点误差崩溃
        for _ in range(50):
            result = sel.compete(self._candidates(5))
            assert result.winner is not None

    def test_type_ignore_comment_removed(self):
        """B3: 确认 # type: ignore[unreachable] 已移除。"""
        import inspect

        from src.gwt.attention_selector import AttentionSelector as AS

        source = inspect.getsource(AS.compete)
        assert "type: ignore[unreachable]" not in source, (
            "误置的 type:ignore 注释应已移除（该代码块非不可达）"
        )
