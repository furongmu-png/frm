"""自然语言对话技能。

使用文本解码器 + 检索式生成（从语料中检索相关语句再改写）。
对话管理：维护对话状态（意图、槽位），利用元认知决定何时澄清或提问。
"""
from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)

#: 意图类别
INTENTS = (
    "question",
    "statement",
    "request",
    "greeting",
    "farewell",
    "clarification",
)

#: 默认回复语料
DEFAULT_RESPONSES = {
    "question": [
        "Let me think about that. {topic} is interesting.",
        "Regarding {topic}, I believe it relates to perception.",
        "That's a good question about {topic}.",
    ],
    "statement": [
        "I see, you mentioned {topic}. Tell me more.",
        "Interesting point about {topic}.",
        "Noted. {topic} seems important to you.",
    ],
    "request": [
        "I'll try to help with {topic}.",
        "Let me work on {topic} for you.",
    ],
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! What would you like to explore?",
    ],
    "farewell": [
        "Goodbye! It was nice talking to you.",
        "See you later!",
    ],
    "clarification": [
        "Could you clarify what you mean by {topic}?",
        "I'm not sure I understand. Can you elaborate on {topic}?",
    ],
}


class DialogueState:
    """对话状态：意图、槽位、对话历史。"""

    def __init__(self) -> None:
        self.intent: str = "statement"
        self.slots: dict[str, str] = {}
        self.history: deque[dict[str, str]] = deque(maxlen=50)
        self.turn_count: int = 0
        self.last_topic: str | None = None

    def update(self, role: str, text: str, intent: str, topic: str | None) -> None:
        self.intent = intent
        self.turn_count += 1
        if topic:
            self.last_topic = topic
            self.slots["topic"] = topic
        self.history.append({"role": role, "text": text, "intent": intent})

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "slots": dict(self.slots),
            "turn_count": self.turn_count,
            "last_topic": self.last_topic,
            "recent_history": list(self.history)[-5:],
        }


class DialogueAgent(SkillBase):
    """对话代理：检索式生成 + 对话状态管理。

    输入：对话历史（用户+模型轮流发言）
    输出：下一轮回复
    """

    name = "dialogue"
    dimension = "interaction"

    def __init__(
        self,
        *,
        latent_dim: int = 32,
        max_context: int = 10,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self._state = DialogueState()
        self.max_context = max_context
        self._rng = np.random.default_rng(111)

        # 简单文本编码器（bag-of-chars + 随机投影）
        self._latent_dim = latent_dim
        self._proj = self._rng.standard_normal(
            (256, latent_dim)
        ).astype(np.float64) * (1.0 / 16.0)

        # 检索池
        self._retrieval_pool: list[tuple[np.ndarray, str]] = []
        self._last_response: str = ""
        self._clarification_needed: bool = False

    def _encode(self, text: str) -> np.ndarray:
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

    def classify_intent(self, text: str) -> str:
        """分类用户意图。"""
        lower = text.lower().strip()
        # 问题：以 ? 结尾或包含疑问词（优先检查，避免 "hi" in "this" 误匹配）
        if lower.endswith("?") or any(
            w in lower for w in ["what", "why", "how", "when", "where", "谁", "什么", "为什么", "怎么"]
        ):
            return "question"
        if any(w in lower for w in ["hello", "hey", "你好", "嗨"]):
            return "greeting"
        # "hi" 作为独立词检查（避免匹配 "this", "high" 等）
        if re.search(r"\bhi\b", lower):
            return "greeting"
        if any(w in lower for w in ["bye", "goodbye", "see you", "再见"]):
            return "farewell"
        if any(w in lower for w in ["please", "can you", "do this", "请", "帮我"]):
            return "request"
        if any(w in lower for w in ["what do you mean", "clarify", "不明白", "没懂"]):
            return "clarification"
        return "statement"

    def extract_topic(self, text: str) -> str | None:
        """提取对话主题（简单关键词提取）。"""
        # 移除常见停用词
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "do", "does",
            "did", "what", "why", "how", "when", "where", "can", "you",
            "i", "me", "my", "we", "they", "it", "to", "of", "in", "on",
        }
        words = re.findall(r"[a-zA-Z]+", text.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        if keywords:
            return keywords[0]
        # 中文关键词
        cn = re.findall(r"[\u4e00-\u9fff]+", text)
        if cn:
            return cn[0][:4]
        return None

    def ingest_corpus(self, texts: list[str]) -> None:
        """摄入语料到检索池。"""
        for text in texts:
            latent = self._encode(text)
            self._retrieval_pool.append((latent, text))

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """检索相关语句。"""
        if not self._retrieval_pool:
            return []
        q_latent = self._encode(query)
        scores = []
        for latent, text in self._retrieval_pool:
            sim = float(np.dot(q_latent, latent))
            scores.append((sim, text))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in scores[:k]]

    def generate_response(self, user_text: str) -> str:
        """生成回复。"""
        intent = self.classify_intent(user_text)
        topic = self.extract_topic(user_text) or self._state.last_topic or "that"

        # 元认知：如果置信度低且意图模糊，请求澄清
        if intent == "statement" and len(user_text.split()) < 3:
            self._clarification_needed = True
            intent = "clarification"

        # 检索相关语句
        retrieved = self.retrieve(user_text, k=1)
        if retrieved and intent in ("question", "statement"):
            # 改写检索结果
            base = retrieved[0]
            if len(base) > 10:
                response = base[:100] + " ..."
            else:
                response = base
        else:
            # 从模板生成
            templates = DEFAULT_RESPONSES.get(intent, DEFAULT_RESPONSES["statement"])
            template = self._rng.choice(templates)
            response = template.format(topic=topic)

        # 更新状态
        self._state.update("user", user_text, intent, topic)
        self._state.update("model", response, intent, topic)
        self._last_response = response
        return response

    def process(self, ctx: SkillContext) -> SkillResult:
        # 从 context 提取用户输入
        user_text = None
        if ctx.raw_observation is not None:
            if isinstance(ctx.raw_observation, str):
                user_text = ctx.raw_observation
            elif isinstance(ctx.raw_observation, dict):
                user_text = ctx.raw_observation.get("dialogue") or ctx.raw_observation.get("text")

        if user_text is None:
            return SkillResult(
                name=self.name,
                data={
                    "ready": False,
                    "state": self._state.to_dict(),
                    "last_response": self._last_response,
                },
            )

        response = self.generate_response(user_text)

        return SkillResult(
            name=self.name,
            data={
                "ready": True,
                "user_input": user_text,
                "response": response,
                "intent": self._state.intent,
                "state": self._state.to_dict(),
                "clarification_needed": self._clarification_needed,
                "retrieval_pool_size": len(self._retrieval_pool),
            },
        )
