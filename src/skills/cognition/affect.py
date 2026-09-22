"""幽默与情感理解技能。

对文本或场景进行情感分类（积极/消极/中性）和幽默检测。
情感向量整合到自由能计算中，作为偏好先验。
"""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)

#: 情感类别
AFFECT_CATEGORIES = ("positive", "negative", "neutral")

#: 积极词汇表（英文 + 中文）
POSITIVE_WORDS = frozenset(
    {
        "good", "great", "happy", "wonderful", "excellent", "love", "joy",
        "beautiful", "amazing", "fantastic", "perfect", "best", "win",
        "success", "hope", "smile", "kind", "brave", "bright", "warm",
        "好", "棒", "开心", "快乐", "美好", "优秀", "爱", "喜悦",
        "美丽", "惊人", "完美", "最好", "胜利", "成功", "希望", "微笑",
    }
)

#: 消极词汇表
NEGATIVE_WORDS = frozenset(
    {
        "bad", "terrible", "sad", "awful", "hate", "angry", "fear",
        "dark", "wrong", "fail", "loss", "pain", "hurt", "broken",
        "evil", "death", "cry", "lonely", "cold",
        "坏", "差", "伤心", "糟糕", "恨", "愤怒", "恐惧",
        "黑暗", "错误", "失败", "损失", "痛苦", "受伤", "破碎",
        "邪恶", "死亡", "哭泣", "孤独", "寒冷",
    }
)

#: 幽默标记词
HUMOR_MARKERS = frozenset(
    {
        "lol", "haha", "funny", "joke", "hilarious", "laugh",
        "哈哈", "搞笑", "笑话", "笑死", "有趣",
    }
)


class AffectModule(SkillBase):
    """情感与幽默理解模块。

    对文本进行情感分类和幽默检测，生成情感向量，
    可作为偏好先验整合到自由能计算中。
    """

    name = "affect"
    dimension = "cognition"

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.latent_dim = latent_dim
        self._rng = np.random.default_rng(321)
        self._proj = self._rng.standard_normal(
            (3, latent_dim)  # 3 类情感 → latent
        ).astype(np.float64) * (1.0 / np.sqrt(3))
        self._last_sentiment: str = "neutral"
        self._last_humor: float = 0.0
        self._last_affect_vector: np.ndarray = np.zeros(latent_dim)

    def _tokenize(self, text: str) -> list[str]:
        """简单分词：英文按空格，中文按字。"""
        # 英文单词
        en_words = re.findall(r"[a-zA-Z]+", text.lower())
        # 中文字符
        cn_chars = re.findall(r"[\u4e00-\u9fff]+", text)
        # 中文按 2-gram
        cn_words = []
        for seg in cn_chars:
            if len(seg) >= 2:
                cn_words.extend([seg[i : i + 2] for i in range(len(seg) - 1)])
            else:
                cn_words.append(seg)
        return en_words + cn_words

    def classify_sentiment(self, text: str) -> tuple[str, np.ndarray]:
        """分类情感。

        Returns
        -------
        (category, probabilities)
            category ∈ {"positive", "negative", "neutral"}
            probabilities: shape (3,)
        """
        tokens = self._tokenize(text)
        if not tokens:
            return "neutral", np.array([0.33, 0.33, 0.34])

        pos_count = sum(1 for t in tokens if t in POSITIVE_WORDS)
        neg_count = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        total = len(tokens)

        # 分数：积极/消极词密度，中性为剩余但有权重衰减
        pos_score = pos_count / total
        neg_score = neg_count / total
        # 中性分：基础分 + 极少情感词时的高值，但情感词多时被压制
        neu_score = max(0.0, 1.0 - pos_score - neg_score) * 0.5

        scores = np.array([pos_score, neg_score, neu_score])
        # softmax 放大差异
        scores = np.exp(scores * 4.0)
        probs = scores / scores.sum()

        category = AFFECT_CATEGORIES[int(np.argmax(probs))]
        return category, probs

    def detect_humor(self, text: str) -> float:
        """检测幽默程度（0-1）。"""
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        humor_count = sum(1 for t in tokens if t in HUMOR_MARKERS)
        # 也检测反转结构（"but"/"突然" + 意外结果）
        inversion_patterns = ["but then", "unexpectedly", "suddenly"]
        inv_count = sum(1 for p in inversion_patterns if p in text.lower())

        humor_score = min(1.0, (humor_count * 0.5 + inv_count * 0.3) / max(len(tokens), 1) * 10)
        return float(humor_score)

    def generate_affect_vector(self, probs: np.ndarray, humor: float) -> np.ndarray:
        """生成情感向量。

        积极情感使向量偏向"趋近"方向，消极情感偏向"回避"方向，
        幽默增加向量的"趣味"分量。
        """
        affect = probs @ self._proj  # (latent_dim,)
        # 加入幽默分量
        humor_vec = np.ones(self.latent_dim) * humor * 0.3
        affect = affect + humor_vec
        # L2 归一化
        norm = np.linalg.norm(affect)
        if norm > 1e-10:
            affect = affect / norm
        return affect

    def compute_preference_prior(self, probs: np.ndarray) -> float:
        """计算偏好先验值（用于自由能计算）。

        积极情感 → 负偏好值（降低自由能，鼓励趋近）
        消极情感 → 正偏好值（增加自由能，鼓励回避）

        Parameters
        ----------
        probs : np.ndarray
            情感概率 [positive, negative, neutral]。
        """
        # 积极概率 - 消极概率，取负使积极→负偏好
        return float(-(probs[0] - probs[1]) * 0.5)

    def analyze(self, text: str) -> dict[str, Any]:
        """完整情感分析。"""
        category, probs = self.classify_sentiment(text)
        humor = self.detect_humor(text)
        affect_vec = self.generate_affect_vector(probs, humor)
        preference = self.compute_preference_prior(probs)

        self._last_sentiment = category
        self._last_humor = humor
        self._last_affect_vector = affect_vec

        return {
            "sentiment": category,
            "probabilities": {
                AFFECT_CATEGORIES[i]: float(probs[i])
                for i in range(3)
            },
            "humor": humor,
            "affect_vector": affect_vec.tolist(),
            "preference_prior": preference,
        }

    def process(self, ctx: SkillContext) -> SkillResult:
        # 提取文本
        text = ""
        if ctx.raw_observation is not None:
            if isinstance(ctx.raw_observation, str):
                text = ctx.raw_observation
            elif isinstance(ctx.raw_observation, dict):
                text = ctx.raw_observation.get("text", "")

        if not text:
            # 从 belief 生成模拟情感
            if ctx.belief is not None and len(ctx.belief) > 0:
                # belief 均值正 → 积极
                b_mean = float(np.mean(ctx.belief[:8]))
                if b_mean > 0.1:
                    category = "positive"
                    probs = np.array([0.6, 0.1, 0.3])
                elif b_mean < -0.1:
                    category = "negative"
                    probs = np.array([0.1, 0.6, 0.3])
                else:
                    category = "neutral"
                    probs = np.array([0.3, 0.3, 0.4])
                humor = max(0.0, float(np.std(ctx.belief[:8])) * 0.5)
                affect_vec = self.generate_affect_vector(probs, humor)
                preference = self.compute_preference_prior(probs)
                self._last_sentiment = category
                self._last_humor = humor
                self._last_affect_vector = affect_vec
            else:
                category = "neutral"
                probs = np.array([0.33, 0.33, 0.34])
                humor = 0.0
                affect_vec = np.zeros(self.latent_dim)
                preference = 0.0
        else:
            result = self.analyze(text)
            category = result["sentiment"]
            probs = np.array(
                [result["probabilities"][c] for c in AFFECT_CATEGORIES]
            )
            humor = result["humor"]
            affect_vec = np.array(result["affect_vector"])
            preference = result["preference_prior"]

        return SkillResult(
            name=self.name,
            data={
                "sentiment": category,
                "probabilities": {
                    AFFECT_CATEGORIES[i]: float(probs[i])
                    for i in range(3)
                },
                "humor": humor,
                "affect_vector": affect_vec.tolist(),
                "preference_prior": preference,
            },
        )
