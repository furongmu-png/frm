"""通信通道：支持符号语言的涌现与学习。"""

from __future__ import annotations

import threading

import numpy as np


class CommunicationChannel:
    """离散符号通信通道，带有可学习的嵌入与“语言涌现”统计。

    每个符号对应一个 ``embed_dim`` 维的可学习嵌入向量。发送方通过
    :meth:`encode_message` 把符号映射为向量；接收方通过
    :meth:`decode_observation` 用余弦相似度反查最近符号。通道还会记录
    符号与事件类型的共现统计，用于检测“涌现含义”。
    """

    def __init__(
        self,
        vocab_size: int = 10,
        embed_dim: int = 8,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self._rng = np.random.default_rng(seed)
        # 可重入锁：保护 embeddings / usage_counts /
        # symbol_event_correlation 的读写。
        self._lock = threading.RLock()
        # 可学习的符号嵌入表，初始化为小方差正态分布。
        self.embeddings = self._rng.normal(0, 0.1, (vocab_size, embed_dim))
        # 每个符号被使用的次数。
        self.usage_counts = np.zeros(vocab_size, dtype=int)
        # 符号 -> {事件类型: 出现次数}，用于统计符号与事件的共现。
        self.symbol_event_correlation: dict[int, dict[str, int]] = {
            i: {} for i in range(vocab_size)
        }

    # ------------------------------------------------------------------
    # 编 / 解码
    # ------------------------------------------------------------------
    def encode_message(self, symbol: int) -> np.ndarray:
        """把符号编码为嵌入向量。

        返回嵌入的副本，避免调用方原地修改污染内部 ``embeddings`` 表。
        """
        if not isinstance(symbol, (int, np.integer)):
            raise TypeError(f"symbol 必须为整数，收到 {type(symbol)!r}")
        if symbol < 0 or symbol >= self.vocab_size:
            raise IndexError(
                f"symbol 越界：{symbol} 不在 [0, {self.vocab_size})"
            )
        with self._lock:
            # 返回副本，防止调用方通过返回值反向修改内部嵌入表。
            return self.embeddings[symbol].copy()

    def decode_observation(self, received: np.ndarray) -> int:
        """用余弦相似度把接收向量解码为最近的符号索引。"""
        with self._lock:
            received = np.asarray(received, dtype=float).ravel()
            if received.shape != (self.embed_dim,) or not np.all(
                np.isfinite(received)
            ):
                raise ValueError(
                    f"received 必须是长度 {self.embed_dim} 的有限实向量"
                )
            emb = self.embeddings
            # 余弦相似度：对每行归一化后做内积。
            norms = np.linalg.norm(emb, axis=1)
            received_norm = float(np.linalg.norm(received))
            if received_norm == 0.0:
                return int(np.argmax(norms))
            safe = np.where(norms > 0, norms, 1.0)
            normalized = emb / safe[:, None]
            sims = normalized @ (received / received_norm)
            return int(np.argmax(sims))

    # ------------------------------------------------------------------
    # 使用统计
    # ------------------------------------------------------------------
    def record_usage(self, symbol: int, event_type: str = "") -> None:
        """记录一次符号使用，并（可选地）登记其关联事件类型。"""
        if not isinstance(symbol, (int, np.integer)):
            raise TypeError(f"symbol 必须为整数，收到 {type(symbol)!r}")
        if symbol < 0 or symbol >= self.vocab_size:
            raise IndexError(
                f"symbol 越界：{symbol} 不在 [0, {self.vocab_size})"
            )
        with self._lock:
            self.usage_counts[symbol] += 1
            if event_type:
                bucket = self.symbol_event_correlation[symbol]
                bucket[event_type] = bucket.get(event_type, 0) + 1

    def compute_communication_reward(
        self,
        sender_symbol: int,
        receiver_prediction_error_before: float,
        receiver_prediction_error_after: float,
    ) -> float:
        """计算通信奖励：接收方预测误差下降得越多，奖励越大。

        若误差下降（``before > after``），奖励为正；误差上升则为负；
        相等为零。该信号用于鼓励“有用”的通信。
        """
        # sender_symbol 当前仅作为 API 上下文保留，不参与计算；保留签名
        # 以便上层将奖励归因到具体发送符号。
        del sender_symbol
        with self._lock:
            return float(
                receiver_prediction_error_before
                - receiver_prediction_error_after
            ) * 0.1

    # ------------------------------------------------------------------
    # 学习
    # ------------------------------------------------------------------
    def update_embeddings(
        self, symbol: int, target: np.ndarray, lr: float = 0.01
    ) -> None:
        """Hebbian 式更新：把符号嵌入向 target 拉近一步。"""
        if not isinstance(symbol, (int, np.integer)):
            raise TypeError(f"symbol 必须为整数，收到 {type(symbol)!r}")
        if symbol < 0 or symbol >= self.vocab_size:
            raise IndexError(
                f"symbol 越界：{symbol} 不在 [0, {self.vocab_size})"
            )
        with self._lock:
            target_arr = np.asarray(target, dtype=float)
            if target_arr.shape != (self.embed_dim,):
                raise ValueError(
                    f"target 形状应为 ({self.embed_dim},)，收到 {target_arr.shape}"
                )
            self.embeddings[symbol] += lr * (
                target_arr - self.embeddings[symbol]
            )

    # ------------------------------------------------------------------
    # 涌现含义
    # ------------------------------------------------------------------
    def get_emergent_meanings(self) -> dict[int, str]:
        """返回已涌现出稳定含义的符号 -> 事件类型映射。

        判据：某符号的某个关联事件类型计数 > 5，且占该符号总关联计数的
        50% 以上时，认为该符号“意味着”该事件类型。
        """
        with self._lock:
            meanings: dict[int, str] = {}
            for symbol, bucket in self.symbol_event_correlation.items():
                if not bucket:
                    continue
                total = sum(bucket.values())
                if total == 0:
                    continue
                event_type, count = max(bucket.items(), key=lambda kv: kv[1])
                if count > 5 and count > 0.5 * total:
                    meanings[symbol] = event_type
            return meanings

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict:
        """通道的汇总统计。"""
        with self._lock:
            emergent = self.get_emergent_meanings()
            return {
                "vocab_size": self.vocab_size,
                "total_usage": int(self.usage_counts.sum()),
                "n_emergent_meanings": len(emergent),
            }

    # ------------------------------------------------------------------
    # Phase 3 扩展：策略梯度更新、互信息分析、通信网络
    # ------------------------------------------------------------------
    def policy_gradient_update(
        self,
        symbol: int,
        reward: float,
        lr: float = 0.01,
    ) -> None:
        """REINFORCE 式策略梯度更新。

        奖励为正时增强该符号的嵌入（使其更可能被选中），
        奖励为负时减弱。

        Parameters
        ----------
        symbol:
            被奖励的符号索引。
        reward:
            标量奖励（正=有效通信，负=有害通信）。
        lr:
            学习率。
        """
        if not isinstance(symbol, (int, np.integer)):
            raise TypeError(f"symbol 必须为整数，收到 {type(symbol)!r}")
        if symbol < 0 or symbol >= self.vocab_size:
            raise IndexError(
                f"symbol 越界：{symbol} 不在 [0, {self.vocab_size})"
            )
        with self._lock:
            # 策略梯度：嵌入向自身方向移动（增强）或远离（减弱）。
            # sign(reward) * lr * embedding
            direction = np.sign(reward)
            self.embeddings[symbol] += direction * lr * self.embeddings[symbol]
            # 归一化防止嵌入爆炸。
            norm = float(np.linalg.norm(self.embeddings[symbol]))
            if norm > 1e-6:
                self.embeddings[symbol] /= norm

    def compute_mutual_information(
        self,
        symbol_usage: list[int],
        event_labels: list[str],
    ) -> float:
        """计算符号使用与事件标签之间的互信息 I(symbol; event)。

        用于检测符号是否形成了稳定指称（高 MI = 符号携带事件信息）。

        Parameters
        ----------
        symbol_usage:
            每步使用的符号索引列表。
        event_labels:
            对应步的事件标签列表（与 symbol_usage 等长）。

        Returns
        -------
        float
            互信息（nat），0 = 无关联。
        """
        with self._lock:
            if len(symbol_usage) != len(event_labels) or len(symbol_usage) == 0:
                return 0.0
            n = len(symbol_usage)
            # 联合分布 P(symbol, event)
            joint: dict[tuple[int, str], int] = {}
            p_sym: dict[int, int] = {}
            p_evt: dict[str, int] = {}
            for s, e in zip(symbol_usage, event_labels, strict=True):
                joint[(s, e)] = joint.get((s, e), 0) + 1
                p_sym[s] = p_sym.get(s, 0) + 1
                p_evt[e] = p_evt.get(e, 0) + 1
            mi = 0.0
            for (s, e), count in joint.items():
                p_se = count / n
                p_s = p_sym[s] / n
                p_e = p_evt[e] / n
                if p_s > 0 and p_e > 0 and p_se > 0:
                    mi += p_se * np.log(p_se / (p_s * p_e))
            return float(max(0.0, mi))

    def get_communication_network(
        self, sender_receiver_pairs: list[tuple[int, int]],
    ) -> dict[tuple[int, int], int]:
        """统计通信网络：谁跟谁说话了多少次。

        Parameters
        ----------
        sender_receiver_pairs:
            (sender_id, receiver_id) 元组列表。

        Returns
        -------
        dict[tuple[int, int], int]
            边权重 = 通信次数。
        """
        with self._lock:
            net: dict[tuple[int, int], int] = {}
            for pair in sender_receiver_pairs:
                net[pair] = net.get(pair, 0) + 1
            return net
