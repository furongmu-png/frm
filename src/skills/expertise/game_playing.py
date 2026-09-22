"""游戏策略学习技能。

提供轻量 Tic-Tac-Toe（井字棋）接口，支持自我对弈与 MCTS 搜索。
不依赖外部棋盘库，所有逻辑自包含。

- ``TicTacToeBoard``：状态机 + 合法动作生成 + 胜负判定
- ``GamePlayer``：自我对弈 + MCTS + 简易价值网络（线性 + 位置特征）
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# 棋盘
# ------------------------------------------------------------------ #


#: 3x3 位置编码（行优先）
_EMPTY = 0
_X = 1
_O = 2

#: 获胜线
_WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # 横
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # 竖
    (0, 4, 8), (2, 4, 6),              # 对角
)


@dataclass
class TicTacToeBoard:
    """井字棋状态。

    Attributes
    ----------
    cells : tuple of 9 ints (0/1/2)
    to_move : int (1=X 或 2=O)
    """
    cells: tuple[int, ...] = (_EMPTY,) * 9
    to_move: int = _X

    @classmethod
    def initial(cls) -> "TicTacToeBoard":
        return cls()

    def legal_moves(self) -> list[int]:
        return [i for i, c in enumerate(self.cells) if c == _EMPTY]

    def apply(self, move: int) -> "TicTacToeBoard":
        if move not in self.legal_moves():
            raise ValueError(f"illegal move {move}")
        cells = list(self.cells)
        cells[move] = self.to_move
        return TicTacToeBoard(
            cells=tuple(cells),
            to_move=_O if self.to_move == _X else _X,
        )

    def winner(self) -> int | None:
        for a, b, c in _WIN_LINES:
            if self.cells[a] != _EMPTY and self.cells[a] == self.cells[b] == self.cells[c]:
                return self.cells[a]
        if _EMPTY not in self.cells:
            return 0  # 平局
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None

    def feature_vector(self) -> np.ndarray:
        """9 维特征：X 位置为 +1，O 位置为 -1，空为 0。"""
        arr = np.zeros(9, dtype=np.float64)
        for i, c in enumerate(self.cells):
            if c == _X:
                arr[i] = 1.0
            elif c == _O:
                arr[i] = -1.0
        return arr

    def to_string(self) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = []
        for r in range(3):
            rows.append(" ".join(symbols[self.cells[r * 3 + c]] for c in range(3)))
        return "\n".join(rows)


# ------------------------------------------------------------------ #
# 价值网络（线性 + bias）
# ------------------------------------------------------------------ #


class ValueNet:
    """线性价值网络：从 9 维特征预测当前玩家赢率。

    在线 SGD 更新，目标 = 实际胜负（+1 当前玩家赢，-1 输，0 平）。
    """

    def __init__(self, lr: float = 0.05, seed: int = 42) -> None:
        self.lr = lr
        self._rng = np.random.default_rng(seed)
        self._w = self._rng.standard_normal(9) * 0.05
        self._b = 0.0

    def value(self, board: TicTacToeBoard) -> float:
        """返回当前玩家视角的赢率 ∈ [-1, 1]。"""
        feat = board.feature_vector()
        # 当前玩家视角：把自己标记视为正
        if board.to_move == _O:
            feat = -feat
        v = float(feat @ self._w + self._b)
        return float(np.tanh(v))

    def update(self, board: TicTacToeBoard, target: float) -> float:
        """SGD 更新，返回预测误差。"""
        feat = board.feature_vector()
        if board.to_move == _O:
            feat = -feat
        v = float(feat @ self._w + self._b)
        err = target - float(np.tanh(v))
        grad = (1.0 - float(np.tanh(v)) ** 2) * err
        self._w += self.lr * grad * feat
        self._b += self.lr * grad
        return abs(err)

    def params(self) -> dict[str, Any]:
        return {"w_norm": float(np.linalg.norm(self._w)), "bias": float(self._b)}


# ------------------------------------------------------------------ #
# MCTS
# ------------------------------------------------------------------ #


@dataclass
class _GameNode:
    board: TicTacToeBoard
    parent: Any = None
    move: int | None = None
    children: list[Any] = field(default_factory=list)
    visits: int = 0
    wins: float = 0.0
    untried: list[int] = field(default_factory=list)

    def ucb(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent is not None else 1
        return self.wins / self.visits + c * math.sqrt(
            math.log(max(parent_visits, 1)) / max(self.visits, 1)
        )


class MCTSPlayer:
    """MCTS 搜索器（含价值网络先验）。"""

    def __init__(
        self,
        value_net: ValueNet,
        *,
        n_simulations: int = 50,
        exploration_c: float = 1.414,
        rollout_depth: int = 9,
        seed: int = 7,
    ) -> None:
        self.value_net = value_net
        self.n_simulations = n_simulations
        self.exploration_c = exploration_c
        self.rollout_depth = rollout_depth
        self._rng = np.random.default_rng(seed)

    def choose_move(self, board: TicTacToeBoard) -> int:
        # 启发式先手：若有直接获胜着法，立即下。
        win_move = self._find_winning_move(board, board.to_move)
        if win_move is not None:
            return win_move
        # 启发式阻挡：若对手有直接获胜着法，立即阻挡。
        opp = _O if board.to_move == _X else _X
        block_move = self._find_winning_move(board, opp)
        if block_move is not None:
            return block_move

        root = _GameNode(board=board)
        root.untried = board.legal_moves()

        if len(root.untried) == 1:
            return root.untried[0]

        for _ in range(self.n_simulations):
            node = self._select(root)
            if node.untried:
                node = self._expand(node)
            reward = self._simulate(node.board)
            self._backprop(node, reward)

        # 选访问次数最多（更稳健）
        best = max(root.children, key=lambda n: n.visits)
        return best.move  # type: ignore[arg-type]

    def _find_winning_move(
        self, board: TicTacToeBoard, player: int
    ) -> int | None:
        """查找让 player 立即获胜的着法。"""
        for move in board.legal_moves():
            after = board.apply(move)
            if after.winner() == player:
                return move
        return None

    # ------------------------------------------------------------------ #
    def _select(self, node: _GameNode) -> _GameNode:
        while not node.untried and node.children:
            node = max(node.children, key=lambda n: n.ucb(self.exploration_c))
            if node.board.is_terminal():
                return node
        return node

    def _expand(self, node: _GameNode) -> _GameNode:
        if not node.untried:
            return node
        move = node.untried.pop()
        child_board = node.board.apply(move)
        child = _GameNode(board=child_board, parent=node, move=move)
        child.untried = child_board.legal_moves()
        node.children.append(child)
        return child

    def _simulate(self, board: TicTacToeBoard) -> float:
        """随机 rollout，结合价值网络作为叶子估值。"""
        if board.is_terminal():
            w = board.winner()
            # reward 是 root 玩家视角
            return self._reward_from_winner(w, board)
        # 价值网络先验
        v = self.value_net.value(board)
        # 简短 rollout（最多 rollout_depth 步）
        cur = board
        steps = 0
        while not cur.is_terminal() and steps < self.rollout_depth:
            moves = cur.legal_moves()
            move = int(self._rng.integers(0, len(moves)))
            cur = cur.apply(moves[move])
            steps += 1
        if cur.is_terminal():
            w = cur.winner()
            return self._reward_from_winner(w, board)
        # 没分出胜负 → 用价值网络的预测作为奖励
        return v * 0.5  # 软奖励

    def _reward_from_winner(
        self, winner: int | None, root_board: TicTacToeBoard
    ) -> float:
        if winner is None:
            return 0.0
        if winner == 0:
            return 0.0
        # 当前轮到的玩家
        return 1.0 if winner == root_board.to_move else -1.0

    def _backprop(self, node: _GameNode, reward: float) -> None:
        # reward 是 root 视角；逐层翻转
        sign = 1.0
        while node is not None:
            node.visits += 1
            node.wins += reward * sign
            sign = -sign
            node = node.parent


# ------------------------------------------------------------------ #
# 主技能类
# ------------------------------------------------------------------ #


class GamePlayer(SkillBase):
    """游戏策略技能：自我对弈 + MCTS。

    - ``self_play(n)``：进行 n 局自我对弈，积累训练样本更新价值网络。
    - ``choose_move(board)``：对给定棋局选最优着。
    - ``process``：默认演示一局自我对弈。
    """

    name = "game_playing"
    dimension = "expertise"

    #: 完整自我对弈（~9 手 × MCTS）开销很大，节流到每 10 步一局。
    #: ``self_play()`` / ``choose_move()`` 直接调用不受节流影响。
    process_interval = 10

    def __init__(
        self,
        *,
        n_simulations: int = 30,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.value_net = ValueNet()
        self.mcts = MCTSPlayer(self.value_net, n_simulations=n_simulations)
        self._self_play_count: int = 0
        self._x_wins: int = 0
        self._o_wins: int = 0
        self._draws: int = 0
        self._last_game: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #

    def choose_move(self, board: TicTacToeBoard | None = None) -> int:
        board = board or TicTacToeBoard.initial()
        return self.mcts.choose_move(board)

    def self_play(self, n_games: int = 1) -> dict[str, Any]:
        """自我对弈 n 局，更新价值网络。"""
        for _ in range(n_games):
            board = TicTacToeBoard.initial()
            history: list[tuple[TicTacToeBoard, int]] = []  # (board, mover)
            while not board.is_terminal():
                move = self.mcts.choose_move(board)
                history.append((board, board.to_move))
                board = board.apply(move)
            w = board.winner()
            self._self_play_count += 1
            if w == _X:
                self._x_wins += 1
            elif w == _O:
                self._o_wins += 1
            else:
                self._draws += 1
            # 训练价值网络：每个状态的目标 = 最终赢家
            for state, mover in history:
                if w == 0:
                    target = 0.0
                elif w == mover:
                    target = 1.0
                else:
                    target = -1.0
                self.value_net.update(state, target)
            self._last_game = {
                "winner": int(w) if w is not None else 0,
                "n_moves": len(history),
                "final_board": board.to_string(),
            }
        return {
            "games_played": self._self_play_count,
            "x_wins": self._x_wins,
            "o_wins": self._o_wins,
            "draws": self._draws,
            "value_net": self.value_net.params(),
        }

    # ------------------------------------------------------------------ #
    # SkillBase 接口
    # ------------------------------------------------------------------ #

    def process(self, ctx: SkillContext) -> SkillResult:
        """默认演示：1 局自我对弈。"""
        stats = self.self_play(1)
        return SkillResult(
            name=self.name,
            data={
                "games_played": stats["games_played"],
                "x_wins": stats["x_wins"],
                "o_wins": stats["o_wins"],
                "draws": stats["draws"],
                "value_net": stats["value_net"],
                "last_game": self._last_game,
            },
        )

    def snapshot(self) -> dict[str, Any]:
        base = super().snapshot()
        base["data"] = base.get("data", {}) | {
            "games_played": self._self_play_count,
            "x_wins": self._x_wins,
            "o_wins": self._o_wins,
            "draws": self._draws,
            "value_net": self.value_net.params(),
        }
        return base
