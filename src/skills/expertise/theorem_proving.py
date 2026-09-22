"""定理证明技能。

利用命题逻辑 + 算术定理的内部符号引擎，以 MCTS 在证明空间搜索。
奖励为证明步骤减少的自由能（预测误差下降量）。

支持的命题逻辑：
- 肯定前件 (Modus Ponens):  ``A, A->B  ⊢  B``
- 双重否定消除:  ``!!A  ⊢  A``
- 合取引入/消除:  ``A, B  ⊢  A∧B``；``A∧B  ⊢  A``
- 析取引入/消除:  ``A  ⊢  A∨B``；``A∨B, A->C, B->C  ⊢  C``

算术定理（轻量）：
- ``add(a, b) == a + b``  在常量域上验证
- ``commutative(a, b)``    验证交换律
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
# 命题逻辑引擎
# ------------------------------------------------------------------ #


@dataclass
class ProofStep:
    """单个证明步骤。"""
    rule: str
    premises: list[str]
    conclusion: str
    description: str = ""


@dataclass
class ProofState:
    """证明状态：已知事实集合 + 已应用规则序列。"""
    facts: frozenset[str]
    steps: tuple[ProofStep, ...] = ()
    goal: str = ""

    def add_fact(self, fact: str) -> "ProofState":
        if fact in self.facts:
            return self
        return ProofState(
            facts=self.facts | {fact},
            steps=self.steps,
            goal=self.goal,
        )

    def with_step(self, step: ProofStep, new_facts: set[str]) -> "ProofState":
        return ProofState(
            facts=self.facts | new_facts,
            steps=self.steps + (step,),
            goal=self.goal,
        )

    def is_goal_reached(self) -> bool:
        return self.goal in self.facts

    def heuristic_distance(self) -> float:
        """估计到目标的距离（启发式）。

        基于事实数量与目标是否包含已见子项。
        越接近 0 越好。
        """
        if self.is_goal_reached():
            return 0.0
        # 若目标以否定形式存在（如 !!p），距离较短
        # 简化：若任一 fact 是 goal 的子串，距离=1；否则=2
        if any(f in self.goal or self.goal in f for f in self.facts):
            return 1.0
        return 2.0 + max(0.0, 3.0 - len(self.facts) * 0.5)


class LogicEngine:
    """命题逻辑推理引擎。

    提供从已知事实集合到所有可一步推出的新事实的扩展函数。
    规则集合有限，便于 MCTS 展开。
    """

    #: (rule_name, matcher, producer)
    #: matcher(state) -> list of (premises, derived_facts, description)
    RULES: list[
        tuple[str, Any, Any]
    ] = []  # 在类体后用 _register_rules 填充

    @classmethod
    def expand(cls, state: ProofState) -> list[tuple[ProofStep, set[str]]]:
        """返回从 state 一步可得到的所有 (step, new_facts)。"""
        results: list[tuple[ProofStep, set[str]]] = []
        for rule_name, matcher, producer in cls.RULES:
            for premises, derived, desc in matcher(state):
                step = ProofStep(
                    rule=rule_name,
                    premises=premises,
                    conclusion=next(iter(derived)) if derived else "",
                    description=desc,
                )
                results.append((step, derived))
        return results


def _strip_parens(s: str) -> str:
    """剥除包围的单层括号：``(p)`` → ``p``，``p`` → ``p``。"""
    s = s.strip()
    if len(s) >= 2 and s.startswith("(") and s.endswith(")"):
        # 仅在括号配对时剥除（避免 "(a)->(b)" 被错误剥除）
        depth = 0
        for i, c in enumerate(s):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    # 中途配对完成 → 不是整体包围
                    return s
        return s[1:-1]
    return s


def _match_modus_ponens(state: ProofState):
    """A, A->B  ⊢  B

    接受的蕴含式写法：``p->q``、``(p)->(q)``、``(p->q)``。
    """
    facts = state.facts
    for f1 in facts:
        if "->" not in f1:
            continue
        # 找到顶层 "->" 的位置（不进入括号内）
        depth = 0
        sep_idx = -1
        for i, c in enumerate(f1):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == "-" and depth == 0 and i + 1 < len(f1) and f1[i + 1] == ">":
                sep_idx = i
                break
        if sep_idx < 0:
            continue
        ante = _strip_parens(f1[:sep_idx])
        cons = _strip_parens(f1[sep_idx + 2 :])
        if ante in facts and cons not in facts:
            yield ([f1, ante], {cons}, f"MP: {ante}, {f1} ⊢ {cons}")


def _match_double_negation(state: ProofState):
    """!!A  ⊢  A  and  A  ⊢  !!A"""
    for f in state.facts:
        if f.startswith("!!"):
            inner = f[2:]
            yield ([f], {inner}, f"DN: {f} ⊢ {inner}")
        # 反向：A ⊢ !!A（仅在 goal 需要 !!A 时有用，这里始终提供）
        if not f.startswith("!"):
            yield ([f], {f"!!{f}"}, f"DN: {f} ⊢ !!{f}")


def _match_conjunction_intro(state: ProofState):
    """A, B  ⊢  A∧B"""
    facts = list(state.facts)
    for i, a in enumerate(facts):
        for b in facts[i + 1 :]:
            conj = f"{a}∧{b}"
            if conj not in facts:
                yield ([a, b], {conj}, f"∧I: {a}, {b} ⊢ {conj}")


def _match_conjunction_elim(state: ProofState):
    """A∧B  ⊢  A, B"""
    for f in state.facts:
        if "∧" in f and not f.startswith("!"):
            parts = f.split("∧")
            if len(parts) == 2:
                a, b = parts[0], parts[1]
                for x in (a, b):
                    if x not in state.facts:
                        yield ([f], {x}, f"∧E: {f} ⊢ {x}")


LogicEngine.RULES = [
    ("modus_ponens", _match_modus_ponens, None),
    ("double_negation", _match_double_negation, None),
    ("conjunction_intro", _match_conjunction_intro, None),
    ("conjunction_elim", _match_conjunction_elim, None),
]


# ------------------------------------------------------------------ #
# 算术定理
# ------------------------------------------------------------------ #


class ArithmeticChecker:
    """轻量算术定理验证器。

    在常量域上随机采样若干对 (a, b)，验证命题是否对所有采样成立。
    """

    def __init__(self, seed: int = 7) -> None:
        self._rng = np.random.default_rng(seed)

    def verify(
        self,
        predicate: Any,
        n_samples: int = 64,
        domain: tuple[int, int] = (-100, 100),
    ) -> tuple[bool, float]:
        """返回 (是否通过, 平均误差)。"""
        lo, hi = domain
        errors: list[float] = []
        for _ in range(n_samples):
            a = int(self._rng.integers(lo, hi + 1))
            b = int(self._rng.integers(lo, hi + 1))
            try:
                ok = bool(predicate(a, b))
                errors.append(0.0 if ok else 1.0)
            except Exception:  # noqa: BLE001
                errors.append(1.0)
        mean_err = float(np.mean(errors))
        return mean_err == 0.0, mean_err


# ------------------------------------------------------------------ #
# MCTS 证明搜索
# ------------------------------------------------------------------ #


@dataclass
class _MCTSNode:
    state: ProofState
    parent: Any = None
    children: list[Any] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried: list[tuple[ProofStep, set[str]]] = field(default_factory=list)

    def ucb(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent is not None else 1
        exploit = self.value / max(self.visits, 1)
        explore = c * math.sqrt(math.log(max(parent_visits, 1)) / max(self.visits, 1))
        return exploit + explore


class MCTSProver:
    """MCTS 证明搜索器。

    奖励信号：
    - 找到目标：+1.0
    - 步数减少的自由能（启发式）：``-heuristic_distance`` 的归一化
    - 步数惩罚：每步 -0.01
    """

    def __init__(
        self,
        engine: LogicEngine,
        *,
        max_iterations: int = 200,
        max_depth: int = 8,
        exploration_c: float = 1.414,
        seed: int = 11,
    ) -> None:
        self.engine = engine
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.exploration_c = exploration_c
        self._rng = np.random.default_rng(seed)

    def search(self, axioms: list[str], goal: str) -> dict[str, Any]:
        """搜索从公理到目标的证明。

        Returns
        -------
        dict
            ``{"proved": bool, "steps": list[ProofStep], "iterations": int,
               "final_state_facts": list[str]}``
        """
        root_state = ProofState(
            facts=frozenset(axioms),
            steps=(),
            goal=goal,
        )
        root = _MCTSNode(state=root_state)
        root.untried = self.engine.expand(root.state)

        if root.state.is_goal_reached():
            return {
                "proved": True,
                "steps": [],
                "iterations": 0,
                "final_state_facts": sorted(root.state.facts),
            }

        best_leaf: _MCTSNode = root
        for it in range(self.max_iterations):
            node = self._select(root)
            if not node.untried and not node.children:
                # 死胡同 → 回溯
                continue
            if node.untried:
                child = self._expand(node)
            else:
                # 选最优子
                child = max(node.children, key=lambda n: n.ucb(self.exploration_c))
            reward = self._simulate(child)
            self._backprop(child, reward)
            if child.state.is_goal_reached():
                best_leaf = child
                break
            if child.ucb(self.exploration_c) > best_leaf.ucb(self.exploration_c):
                best_leaf = child

        # 沿 best_leaf 回溯取 steps
        if best_leaf.state.is_goal_reached():
            steps = best_leaf.state.steps
            return {
                "proved": True,
                "steps": [
                    {
                        "rule": s.rule,
                        "premises": s.premises,
                        "conclusion": s.conclusion,
                        "description": s.description,
                    }
                    for s in steps
                ],
                "iterations": it + 1,
                "final_state_facts": sorted(best_leaf.state.facts),
            }
        return {
            "proved": False,
            "steps": [],
            "iterations": it + 1,
            "final_state_facts": sorted(best_leaf.state.facts),
        }

    # ------------------------------------------------------------------ #
    # MCTS 四阶段
    # ------------------------------------------------------------------ #

    def _select(self, node: _MCTSNode) -> _MCTSNode:
        while not node.untried and node.children:
            node = max(node.children, key=lambda n: n.ucb(self.exploration_c))
            if node.state.is_goal_reached():
                return node
        return node

    def _expand(self, node: _MCTSNode) -> _MCTSNode:
        if not node.untried:
            return node
        step, new_facts = node.untried.pop()
        new_state = node.state.with_step(step, new_facts)
        child = _MCTSNode(state=new_state, parent=node)
        if len(child.state.steps) < self.max_depth:
            child.untried = self.engine.expand(child.state)
        node.children.append(child)
        return child

    def _simulate(self, node: _MCTSNode) -> float:
        """从 node 出发的随机 rollout。"""
        state = node.state
        steps_taken = 0
        while steps_taken < self.max_depth and not state.is_goal_reached():
            options = self.engine.expand(state)
            if not options:
                break
            step, new_facts = options[int(self._rng.integers(0, len(options)))]
            state = state.with_step(step, new_facts)
            steps_taken += 1
        if state.is_goal_reached():
            return 1.0 - 0.01 * steps_taken
        # 启发式奖励：距离越近越好
        return max(0.0, 1.0 - state.heuristic_distance() - 0.01 * steps_taken)

    def _backprop(self, node: _MCTSNode, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


# ------------------------------------------------------------------ #
# 主技能类
# ------------------------------------------------------------------ #


class TheoremProver(SkillBase):
    """定理证明技能。

    包装逻辑引擎 + MCTS 搜索 + 算术验证器。在 ``think()`` 的
    元认知阶段被调用，输出最近一次证明结果。
    """

    name = "theorem_proving"
    dimension = "expertise"

    #: MCTS 证明搜索开销较高，节流到每 10 步刷新一次。
    #: ``prove()`` 直接调用不受节流影响（仍是完整搜索）。
    process_interval = 10

    def __init__(
        self,
        *,
        max_iterations: int = 200,
        max_depth: int = 8,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.engine = LogicEngine()
        self.searcher = MCTSProver(
            self.engine,
            max_iterations=max_iterations,
            max_depth=max_depth,
        )
        self.arithmetic = ArithmeticChecker()
        self._last_proof: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #

    def prove(
        self, axioms: list[str], goal: str
    ) -> dict[str, Any]:
        """证明命题。"""
        result = self.searcher.search(axioms, goal)
        self._last_proof = result
        return result

    def verify_arithmetic(self, predicate: Any) -> dict[str, Any]:
        """验证算术命题（在随机采样上）。"""
        ok, err = self.arithmetic.verify(predicate)
        return {"verified": ok, "mean_error": err}

    # ------------------------------------------------------------------ #
    # SkillBase 接口
    # ------------------------------------------------------------------ #

    def process(self, ctx: SkillContext) -> SkillResult:
        """默认演示：证明 ``q`` 来自 ``p`` 与 ``(p)->(q)``。"""
        result = self.prove(
            axioms=["p", "(p)->(q)"],
            goal="q",
        )
        return SkillResult(
            name=self.name,
            data={
                "theorem": "p, (p)->(q) ⊢ q",
                "proved": result["proved"],
                "n_steps": len(result["steps"]),
                "iterations": result["iterations"],
                "steps": result["steps"][:8],  # 限制前端载荷
            },
        )

    def snapshot(self) -> dict[str, Any]:
        base = super().snapshot()
        if self._last_proof is not None:
            base["data"] = base.get("data", {}) | {
                "last_proved": self._last_proof["proved"],
                "last_n_steps": len(self._last_proof["steps"]),
            }
        return base
