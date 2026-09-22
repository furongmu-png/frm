"""叙事理解与故事生成技能。

使用层次化 PCN 的高层状态捕捉叙事弧线，
文本解码器（检索 + Markov）生成续写。
故事结构解析：人物、目标、冲突、高潮、结局。
"""
from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)

#: 叙事结构要素
NARRATIVE_ELEMENTS = ("characters", "goals", "conflicts", "climax", "resolution")


class SimpleTextEncoder:
    """简单的 bag-of-characters 文本编码器。"""

    def __init__(self, dim: int = 32, seed: int = 42) -> None:
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._proj = self._rng.standard_normal((256, dim)).astype(np.float64) * (
            1.0 / 16.0
        )

    def encode(self, text: str) -> np.ndarray:
        hist = np.zeros(256, dtype=np.float64)
        for byte in text.encode("utf-8", errors="replace"):
            hist[byte] += 1
        total = hist.sum()
        if total > 0:
            hist /= total
        latent = hist @ self._proj
        norm = np.linalg.norm(latent)
        if norm > 1e-10:
            latent = latent / norm
        return latent


class MarkovTextGenerator:
    """简单 Markov 链文本生成器。"""

    def __init__(self, n: int = 2, seed: int = 42) -> None:
        self.n = n
        self._rng = np.random.default_rng(seed)
        self._chain: dict[str, list[str]] = {}

    def train(self, texts: list[str]) -> None:
        for text in texts:
            tokens = text.split()
            for i in range(len(tokens) - self.n):
                key = " ".join(tokens[i : i + self.n])
                if key not in self._chain:
                    self._chain[key] = []
                self._chain[key].append(tokens[i + self.n])

    def generate(self, seed_text: str = "", max_tokens: int = 50) -> str:
        tokens = seed_text.split()
        if not tokens:
            if self._chain:
                start = self._rng.choice(list(self._chain.keys()))
                tokens = start.split()
            else:
                return ""

        for _ in range(max_tokens):
            key = " ".join(tokens[-self.n :]) if len(tokens) >= self.n else " ".join(tokens)
            if key in self._chain:
                next_token = self._rng.choice(self._chain[key])
                tokens.append(next_token)
            else:
                # 回退到随机选择
                if self._chain:
                    random_key = self._rng.choice(list(self._chain.keys()))
                    tokens.append(self._rng.choice(self._chain[random_key]))
                else:
                    break
        return " ".join(tokens)


class NarrativeModule(SkillBase):
    """叙事理解与生成模块。

    - 输入：故事文本流
    - 输出：故事结构解析 + 续写文本
    - 评估：故事连贯性通过预测误差降低幅度衡量
    """

    name = "narrative"
    dimension = "cognition"

    def __init__(
        self,
        *,
        latent_dim: int = 32,
        max_history: int = 100,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self._encoder = SimpleTextEncoder(dim=latent_dim)
        self._generator = MarkovTextGenerator(n=2)
        self.latent_dim = latent_dim
        self.story_history: deque[str] = deque(maxlen=max_history)
        self._narrative_latents: deque[np.ndarray] = deque(maxlen=max_history)
        self._current_structure: dict[str, Any] = {}

    def ingest_text(self, text: str) -> None:
        """摄入故事文本。"""
        self.story_history.append(text)
        latent = self._encoder.encode(text)
        self._narrative_latents.append(latent)
        self._generator.train([text])

    def parse_structure(self, text: str) -> dict[str, Any]:
        """解析故事结构：人物、目标、冲突、高潮、结局。"""
        # 简单规则解析
        sentences = re.split(r"[.!?。！？]", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        characters: set[str] = set()
        for sent in sentences:
            words = sent.split()
            for w in words:
                # 大写开头的词可能是人物名
                if w and w[0].isupper() and len(w) > 1 and w.isalpha():
                    characters.add(w)

        # 检测冲突关键词
        conflict_words = [
            "but",
            "however",
            "conflict",
            "fight",
            "against",
            "struggle",
            "但是",
            "然而",
            "冲突",
            "斗争",
        ]
        conflicts = [s for s in sentences if any(c in s.lower() for c in conflict_words)]

        # 检测高潮
        climax_words = [
            "suddenly",
            "finally",
            "climax",
            "peak",
            "turning",
            "突然",
            "终于",
            "高潮",
            "转折",
        ]
        climax = [s for s in sentences if any(c in s.lower() for c in climax_words)]

        # 结局
        resolution = sentences[-1] if sentences else ""

        structure = {
            "characters": sorted(characters)[:10],
            "goals": sentences[:3],  # 前 3 句作为目标线索
            "conflicts": conflicts[:3],
            "climax": climax[:2],
            "resolution": resolution,
            "n_sentences": len(sentences),
        }
        self._current_structure = structure
        return structure

    def continue_story(self, seed: str = "", max_tokens: int = 50) -> str:
        """生成故事续写。"""
        if not seed and self.story_history:
            seed = self.story_history[-1][-100:]
        return self._generator.generate(seed, max_tokens=max_tokens)

    def evaluate_coherence(self) -> float:
        """评估故事连贯性：通过叙事 latent 的预测误差降低幅度。

        连贯的故事其 latent 序列应平滑变化（低预测误差）。
        """
        if len(self._narrative_latents) < 2:
            return 1.0
        latents = list(self._narrative_latents)
        errors = []
        for i in range(1, len(latents)):
            # 用前一个 latent 预测当前（简单线性外推）
            pred = latents[i - 1]
            error = float(np.linalg.norm(latents[i] - pred))
            errors.append(error)

        mean_error = float(np.mean(errors))
        # 连贯性 = 1 / (1 + mean_error)，误差越低连贯性越高
        coherence = 1.0 / (1.0 + mean_error)
        return coherence

    def process(self, ctx: SkillContext) -> SkillResult:
        # 从 context 提取文本
        text = None
        if ctx.raw_observation is not None:
            if isinstance(ctx.raw_observation, str):
                text = ctx.raw_observation
            elif isinstance(ctx.raw_observation, dict):
                text = ctx.raw_observation.get("text")

        if text is None:
            # 用 belief 生成一个模拟文本
            text = ""

        if text:
            self.ingest_text(text)
            structure = self.parse_structure(text)
        else:
            structure = self._current_structure

        continuation = self.continue_story(seed=text[-50:] if text else "")
        coherence = self.evaluate_coherence()

        # 叙事 latent 用于前端
        if self._narrative_latents:
            latent = self._narrative_latents[-1]
            latent_preview = latent.tolist()[:16]
        else:
            latent_preview = [0.0] * 16

        return SkillResult(
            name=self.name,
            data={
                "structure": structure,
                "continuation": continuation,
                "coherence": coherence,
                "n_stories": len(self.story_history),
                "narrative_latent": latent_preview,
            },
        )
