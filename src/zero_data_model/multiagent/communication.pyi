from __future__ import annotations

import threading
from typing import Any

import numpy as np

class CommunicationChannel:
    """离散符号通信通道，带有可学习的嵌入与"语言涌现"统计。"""

    vocab_size: int
    embed_dim: int
    _rng: np.random.Generator
    _lock: threading.RLock
    embeddings: np.ndarray
    usage_counts: np.ndarray
    symbol_event_correlation: dict[int, dict[str, int]]

    def __init__(
        self,
        vocab_size: int = 10,
        embed_dim: int = 8,
        seed: int = 42,
    ) -> None: ...

    def encode_message(self, symbol: int) -> np.ndarray: ...

    def decode_observation(self, received: np.ndarray) -> int: ...

    def record_usage(self, symbol: int, event_type: str = "") -> None: ...

    def compute_communication_reward(
        self,
        sender_symbol: int,
        receiver_prediction_error_before: float,
        receiver_prediction_error_after: float,
    ) -> float: ...

    def update_embeddings(
        self, symbol: int, target: np.ndarray, lr: float = 0.01
    ) -> None: ...

    def get_emergent_meanings(self) -> dict[int, str]: ...

    @property
    def stats(self) -> dict[str, Any]: ...

    # Phase 3 扩展
    def policy_gradient_update(
        self, symbol: int, reward: float, lr: float = 0.01
    ) -> None: ...

    def compute_mutual_information(
        self, symbol_usage: list[int], event_labels: list[str]
    ) -> float: ...

    def get_communication_network(
        self, sender_receiver_pairs: list[tuple[int, int]]
    ) -> dict[tuple[int, int], int]: ...


__all__: list[str] = ["CommunicationChannel"]
