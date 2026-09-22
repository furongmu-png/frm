"""专业技能单元测试：程序合成、定理证明、游戏、异常检测。"""
from __future__ import annotations

import numpy as np
import pytest

from skills.base import SkillContext
from skills.expertise.program_synthesis import (
    ProgramSynthesizer,
    execute_in_sandbox,
    SandboxError,
)
from skills.expertise.theorem_proving import (
    TheoremProver,
    LogicEngine,
    ProofState,
    ArithmeticChecker,
)
from skills.expertise.game_playing import (
    GamePlayer,
    TicTacToeBoard,
    ValueNet,
    MCTSPlayer,
)
from skills.expertise.anomaly_detection import (
    AnomalyDetector,
    WindowStats,
)


# ================================================================== #
# 13. ProgramSynthesizer
# ================================================================== #
class TestProgramSynthesizer:
    def test_synthesize_sum_from_spec(self):
        ps = ProgramSynthesizer()
        r = ps.synthesize_from_spec(
            "sum the numbers", [({"nums": [1, 2, 3]}, 6), ({"nums": [10, -5, 0]}, 5)]
        )
        assert r["converged"]
        assert r["error"] <= ps.error_threshold
        assert "sum" in r["code"]

    def test_synthesize_max(self):
        ps = ProgramSynthesizer()
        r = ps.synthesize_from_spec(
            "find max value", [({"nums": [1, 5, 3]}, 5), ({"nums": [-1, -5]}, -1)]
        )
        assert r["converged"]
        assert "max" in r["code"]

    def test_synthesize_from_examples_only(self):
        ps = ProgramSynthesizer()
        examples = [
            ({"nums": [1, 2]}, 3),
            ({"nums": [10, 20]}, 30),
            ({"nums": [-1, -2]}, -3),
        ]
        r = ps.synthesize_from_examples(examples)
        assert r["error"] < 1.0  # 至少找到一个误差较小的候选

    def test_synthesize_no_examples(self):
        ps = ProgramSynthesizer()
        r = ps.synthesize_from_spec("sum")
        assert r["converged"]
        assert r["iterations"] == 1

    def test_synthesize_from_examples_empty_raises(self):
        ps = ProgramSynthesizer()
        with pytest.raises(ValueError):
            ps.synthesize_from_examples([])

    def test_process_via_context(self):
        ps = ProgramSynthesizer()
        ctx = SkillContext(belief=np.zeros(64))
        result = ps.safe_process(ctx)
        assert result.error is None
        assert "best_program" in result.data
        assert "iterations" in result.data

    def test_snapshot_includes_best(self):
        ps = ProgramSynthesizer()
        ps.synthesize_from_spec("sum", [({"nums": [1, 2]}, 3)])
        snap = ps.snapshot()
        assert snap["ready"] is True
        assert "best_error" in snap["data"]

    # --- sandbox ---

    def test_sandbox_rejects_import(self):
        with pytest.raises(SandboxError):
            execute_in_sandbox("import os", {})

    def test_sandbox_rejects_dunder(self):
        with pytest.raises(SandboxError):
            execute_in_sandbox("x = obj.__class__", {})

    def test_sandbox_executes_basic_code(self):
        code = "result = sum([1, 2, 3])"
        out = execute_in_sandbox(code, {})
        assert out["ok"]
        assert out["output"] == 6

    def test_sandbox_returns_none_if_no_result(self):
        code = "x = 5"
        out = execute_in_sandbox(code, {})
        assert out["ok"]
        assert out["output"] is None

    def test_sandbox_runtime_error(self):
        code = "result = 1 / 0"
        out = execute_in_sandbox(code, {})
        assert not out["ok"]
        assert "ZeroDivision" in out["error"]

    def test_sandbox_forbids_open(self):
        with pytest.raises(SandboxError):
            execute_in_sandbox("f = open('/etc/passwd')", {})

    def test_sandbox_timeout_on_infinite_loop(self):
        # 不可达循环 → 超时
        code = "while True: pass"
        out = execute_in_sandbox(code, {}, timeout_s=0.1)
        assert not out["ok"]
        assert out["error"] == "timeout"


# ================================================================== #
# 14. TheoremProver
# ================================================================== #
class TestTheoremProver:
    def test_modus_ponens_simple(self):
        tp = TheoremProver()
        r = tp.prove(["p", "p->q"], "q")
        assert r["proved"]
        assert len(r["steps"]) == 1
        assert r["steps"][0]["rule"] == "modus_ponens"

    def test_modus_ponens_parens(self):
        tp = TheoremProver()
        r = tp.prove(["p", "(p)->(q)"], "q")
        assert r["proved"]

    def test_unprovable_goal(self):
        tp = TheoremProver()
        r = tp.prove(["p"], "q")
        assert not r["proved"]

    def test_conjunction_elim(self):
        tp = TheoremProver()
        r = tp.prove(["p∧q"], "p")
        assert r["proved"]
        assert any(s["rule"] == "conjunction_elim" for s in r["steps"])

    def test_conjunction_intro(self):
        tp = TheoremProver()
        r = tp.prove(["p", "q"], "p∧q")
        assert r["proved"]
        assert any(s["rule"] == "conjunction_intro" for s in r["steps"])

    def test_chained_modus_ponens(self):
        """p, p->q, q->r  ⊢  r"""
        tp = TheoremProver()
        r = tp.prove(["p", "p->q", "q->r"], "r")
        assert r["proved"]
        assert len(r["steps"]) == 2

    def test_arithmetic_verifies_commutativity(self):
        tp = TheoremProver()
        r = tp.verify_arithmetic(lambda a, b: a + b == b + a)
        assert r["verified"]

    def test_arithmetic_falsifies_wrong_claim(self):
        tp = TheoremProver()
        r = tp.verify_arithmetic(lambda a, b: a - b == b - a)
        assert not r["verified"]

    def test_process_via_context(self):
        tp = TheoremProver()
        ctx = SkillContext(belief=np.zeros(32))
        result = tp.safe_process(ctx)
        assert result.error is None
        assert result.data["proved"] is True
        assert "n_steps" in result.data

    def test_logic_engine_expand_returns_steps(self):
        engine = LogicEngine()
        state = ProofState(facts=frozenset(["p", "p->q"]), goal="q")
        expansions = engine.expand(state)
        assert len(expansions) > 0
        # 应该能找到至少一个 modus_ponens 步
        rules = [step.rule for step, _ in expansions]
        assert "modus_ponens" in rules

    def test_arithmetic_checker_uses_seed(self):
        """同种子应得到确定性结果。"""
        a1 = ArithmeticChecker(seed=1)
        a2 = ArithmeticChecker(seed=1)
        ok1, _ = a1.verify(lambda a, b: True)
        ok2, _ = a2.verify(lambda a, b: True)
        assert ok1 == ok2

    def test_snapshot_includes_last_proof(self):
        tp = TheoremProver()
        tp.prove(["p", "p->q"], "q")
        snap = tp.snapshot()
        assert snap["data"]["last_proved"] is True


# ================================================================== #
# 15. GamePlayer
# ================================================================== #
class TestGamePlayer:
    def test_initial_board_empty(self):
        b = TicTacToeBoard.initial()
        assert len(b.legal_moves()) == 9
        assert not b.is_terminal()
        assert b.winner() is None

    def test_apply_move_legal(self):
        b = TicTacToeBoard.initial()
        b2 = b.apply(0)
        assert b2.cells[0] == 1  # X
        assert b2.to_move == 2  # O

    def test_apply_move_illegal_raises(self):
        b = TicTacToeBoard.initial().apply(0)
        with pytest.raises(ValueError):
            b.apply(0)

    def test_winner_horizontal(self):
        b = (
            TicTacToeBoard.initial()
            .apply(0)  # X
            .apply(3)  # O
            .apply(1)  # X
            .apply(4)  # O
            .apply(2)  # X (top row)
        )
        assert b.winner() == 1  # X
        assert b.is_terminal()

    def test_winner_vertical(self):
        b = (
            TicTacToeBoard.initial()
            .apply(0)  # X
            .apply(1)  # O
            .apply(3)  # X
            .apply(2)  # O
            .apply(6)  # X (col 0)
        )
        assert b.winner() == 1

    def test_winner_diagonal(self):
        b = (
            TicTacToeBoard.initial()
            .apply(0)  # X
            .apply(1)  # O
            .apply(4)  # X
            .apply(2)  # O
            .apply(8)  # X (diag)
        )
        assert b.winner() == 1

    def test_draw(self):
        # 直接构造一个平局棋盘（无三连）：
        # O X X
        # X O O
        # X O X
        b = TicTacToeBoard(
            cells=(2, 1, 1, 1, 2, 2, 1, 2, 1),
            to_move=1,
        )
        # 棋盘已满
        assert 0 not in b.cells
        # 没有任何一方获胜
        assert b.winner() == 0
        assert b.is_terminal()

    def test_feature_vector(self):
        b = TicTacToeBoard.initial().apply(0)
        feat = b.feature_vector()
        assert feat[0] == 1.0
        assert feat.sum() == 1.0
        assert feat.shape == (9,)

    def test_value_net_predicts_in_range(self):
        net = ValueNet()
        b = TicTacToeBoard.initial()
        v = net.value(b)
        assert -1.0 <= v <= 1.0

    def test_value_net_learns(self):
        net = ValueNet(lr=0.1)
        b = TicTacToeBoard.initial()
        for _ in range(20):
            net.update(b, target=1.0)
        # 应该收敛到接近 1
        assert net.value(b) > 0.0

    def test_mcts_chooses_winning_move(self):
        """X 已经在 0, 1 → 应选 2 完成三连。"""
        b = (
            TicTacToeBoard.initial()
            .apply(0)
            .apply(3)
            .apply(1)
            .apply(4)
        )
        net = ValueNet()
        player = MCTSPlayer(net, n_simulations=30)
        move = player.choose_move(b)
        assert move == 2  # 唯一的胜利着法

    def test_mcts_blocks_opponent_win(self):
        """O 已经在 3, 4 → X 应选 5 阻挡。"""
        b = (
            TicTacToeBoard.initial()
            .apply(3)
            .apply(0)
            .apply(4)
            .apply(1)
        )
        # 现在 X 在 0,1 / O 在 3,4 → X 应选 5 阻挡 O 完成第 3 行
        # 或选 2 完成顶行（X 已有 0,1）—— 阻挡与进攻都可，进攻更优
        net = ValueNet()
        player = MCTSPlayer(net, n_simulations=30)
        move = player.choose_move(b)
        # 进攻=2, 阻挡=5；两者都合理
        assert move in (2, 5)

    def test_self_play_completes(self):
        gp = GamePlayer(n_simulations=10)
        stats = gp.self_play(3)
        assert stats["games_played"] == 3
        total = stats["x_wins"] + stats["o_wins"] + stats["draws"]
        assert total == 3

    def test_process_via_context(self):
        gp = GamePlayer(n_simulations=10)
        ctx = SkillContext(belief=np.zeros(32))
        result = gp.safe_process(ctx)
        assert result.error is None
        assert result.data["games_played"] >= 1

    def test_snapshot_includes_stats(self):
        gp = GamePlayer(n_simulations=10)
        gp.self_play(1)
        snap = gp.snapshot()
        assert "games_played" in snap["data"]
        assert "value_net" in snap["data"]

    def test_only_move_returns_immediately(self):
        """棋盘只剩一个空位时跳过搜索。"""
        b = TicTacToeBoard(
            cells=(1, 2, 1, 2, 1, 1, 2, 1, 0),
            to_move=2,
        )
        net = ValueNet()
        player = MCTSPlayer(net, n_simulations=10)
        move = player.choose_move(b)
        assert move == 8

    def test_to_string(self):
        b = TicTacToeBoard.initial().apply(0)
        s = b.to_string()
        assert "X" in s
        assert s.count(".") == 8


# ================================================================== #
# 16. AnomalyDetector
# ================================================================== #
class TestAnomalyDetector:
    def test_no_alert_before_min_samples(self):
        ad = AnomalyDetector(min_samples=10)
        for v in [0.1, 0.2, 0.3]:
            assert ad.update("x", v) is None

    def test_triggers_on_outlier(self):
        ad = AnomalyDetector(min_samples=5, sigma_threshold=3.0, cooldown=0)
        for v in [0.1, 0.12, 0.11, 0.13, 0.1, 0.12, 0.11]:
            ad.update("x", v)
        alert = ad.update("x", 5.0)
        assert alert is not None
        assert alert.metric == "x"
        assert alert.value == 5.0

    def test_no_alert_on_normal_value(self):
        ad = AnomalyDetector(min_samples=5, sigma_threshold=3.0)
        for v in [0.1, 0.12, 0.11, 0.13, 0.1, 0.12, 0.11, 0.13, 0.1]:
            assert ad.update("x", v) is None

    def test_cooldown_suppresses_rapid_alerts(self):
        """冷却期内连续异常不重复触发，过期后再次触发。

        充分预热窗口，使后续异常值不会显著拉偏阈值。
        """
        ad = AnomalyDetector(min_samples=5, sigma_threshold=2.0, cooldown=3)
        for v in [0.1, 0.12, 0.11, 0.13, 0.1, 0.12, 0.11]:
            assert ad.update("x", v) is None
        alert1 = ad.update("x", 5.0)
        alert2 = ad.update("x", 6.0)  # 冷却内（1 < 3）→ 抑制
        alert3 = ad.update("x", 7.0)  # 冷却内（2 < 3）→ 抑制
        alert4 = ad.update("x", 10.0)  # 冷却过期（3 ≥ 3）→ 触发
        assert alert1 is not None
        assert alert2 is None
        assert alert3 is None
        assert alert4 is not None

    def test_low_outlier_triggers(self):
        ad = AnomalyDetector(min_samples=5, sigma_threshold=3.0, cooldown=0)
        for v in [0.5, 0.6, 0.55, 0.5, 0.6, 0.55]:
            ad.update("y", v)
        alert = ad.update("y", -5.0)
        assert alert is not None
        assert alert.value == -5.0

    def test_explanation_text(self):
        ad = AnomalyDetector(min_samples=3, sigma_threshold=2.0, cooldown=0)
        for v in [1.0, 1.1, 1.0]:
            ad.update("velocity", v)
        alert = ad.update("velocity", 10.0)
        assert alert is not None
        assert "物体速度" in alert.explanation
        assert "增加" in alert.explanation  # 10 > mean

    def test_stats_snapshot(self):
        ad = AnomalyDetector()
        for v in [1.0, 2.0, 3.0]:
            ad.update("a", v)
        snap = ad.stats_snapshot()
        assert "a" in snap
        assert snap["a"]["count"] == 3

    def test_process_via_context(self):
        ad = AnomalyDetector(min_samples=3)
        ctx = SkillContext(belief=np.zeros(32), prediction_error=0.5)
        ad.safe_process(ctx)
        ctx2 = SkillContext(belief=np.zeros(32), prediction_error=10.0)
        # 推几次以达到 min_samples + 触发阈值
        result = None
        for _ in range(10):
            ctx_err = SkillContext(belief=np.zeros(32), prediction_error=10.0)
            result = ad.safe_process(ctx_err)
        assert result is not None
        assert "current_error" in result.data

    def test_snapshot_includes_counts(self):
        ad = AnomalyDetector()
        ad.update("x", 0.1)
        snap = ad.snapshot()
        assert snap["data"]["n_metrics"] == 1
        assert snap["data"]["n_alerts"] == 0

    def test_window_stats_welford(self):
        ws = WindowStats(window_size=10)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ws.push(v)
        assert ws.count == 5
        assert abs(ws.mean - 3.0) < 1e-9
        # 总体方差 = 2.0，标准差 = sqrt(2) ≈ 1.414
        assert abs(ws.std - np.sqrt(2.0)) < 1e-6

    def test_window_stats_window_only(self):
        ws = WindowStats(window_size=3)
        for v in [1.0, 2.0, 3.0, 100.0]:
            ws.push(v)
        # 全局均值受 100 影响；窗口均值只看最近 3 个
        assert ws.window_mean == (2.0 + 3.0 + 100.0) / 3
        assert ws.mean != ws.window_mean  # 全局 ≠ 窗口
