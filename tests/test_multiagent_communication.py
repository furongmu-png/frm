from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.multiagent.communication import CommunicationChannel


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #


def test_construct_default():
    ch = CommunicationChannel()
    assert ch.vocab_size == 10
    assert ch.embed_dim == 8
    assert ch.embeddings.shape == (10, 8)
    assert ch.usage_counts.shape == (10,)
    assert np.all(ch.usage_counts == 0)


def test_construct_with_args():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    assert ch.vocab_size == 5
    assert ch.embed_dim == 4
    assert ch.embeddings.shape == (5, 4)


# --------------------------------------------------------------------------- #
# encode_message / decode_observation / record_usage / compute_communication_reward /
# update_embeddings / get_emergent_meanings
# --------------------------------------------------------------------------- #


def test_encode_message_returns_vector():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    v = ch.encode_message(0)
    assert v.shape == (4,)


def test_encode_message_out_of_range_raises_index_error():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    with pytest.raises(IndexError):
        ch.encode_message(5)
    with pytest.raises(IndexError):
        ch.encode_message(-1)


def test_encode_message_rejects_non_int_symbol():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    with pytest.raises(TypeError):
        ch.encode_message(0.5)
    with pytest.raises(TypeError):
        ch.encode_message("0")


def test_decode_observation_returns_int_symbol():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    # 直接用一个符号的 embedding 来解码 → 应解码回相同符号
    v = ch.encode_message(2)
    decoded = ch.decode_observation(v)
    assert isinstance(decoded, int)
    assert decoded == 2


def test_decode_observation_rejects_wrong_shape():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    with pytest.raises(ValueError):
        ch.decode_observation(np.zeros(3))  # 形状错误


def test_decode_observation_rejects_nan():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    with pytest.raises(ValueError):
        ch.decode_observation(np.array([np.nan, 0.0, 0.0, 0.0]))


def test_decode_observation_zero_vector_returns_argmax_norm():
    """零向量时返回 argmax(norm)。"""
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    decoded = ch.decode_observation(np.zeros(4))
    expected = int(np.argmax(np.linalg.norm(ch.embeddings, axis=1)))
    assert decoded == expected


def test_record_usage_increments_count():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    ch.record_usage(2)
    ch.record_usage(2)
    assert ch.usage_counts[2] == 2


def test_record_usage_with_event_type():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    ch.record_usage(2, event_type="alarm")
    ch.record_usage(2, event_type="alarm")
    assert ch.symbol_event_correlation[2]["alarm"] == 2


def test_record_usage_out_of_range_raises_index_error():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    with pytest.raises(IndexError):
        ch.record_usage(5)


def test_compute_communication_reward_positive_for_error_drop():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    r = ch.compute_communication_reward(
        sender_symbol=0,
        receiver_prediction_error_before=2.0,
        receiver_prediction_error_after=1.0,
    )
    # (2.0 - 1.0) * 0.1 = 0.1
    assert r == pytest.approx(0.1)


def test_compute_communication_reward_negative_for_error_increase():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    r = ch.compute_communication_reward(
        sender_symbol=0,
        receiver_prediction_error_before=1.0,
        receiver_prediction_error_after=2.0,
    )
    assert r < 0.0


def test_compute_communication_reward_zero_for_no_change():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    r = ch.compute_communication_reward(
        sender_symbol=0,
        receiver_prediction_error_before=1.0,
        receiver_prediction_error_after=1.0,
    )
    assert r == 0.0


def test_update_embeddings_moves_embedding_toward_target():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    original = ch.embeddings[0].copy()
    target = np.array([1.0, 0.0, 0.0, 0.0])
    ch.update_embeddings(0, target, lr=0.5)
    # 应向 target 移动
    moved = ch.embeddings[0]
    assert np.linalg.norm(moved - target) < np.linalg.norm(original - target)


def test_update_embeddings_rejects_wrong_shape():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    with pytest.raises(ValueError):
        ch.update_embeddings(0, np.zeros(3))


def test_get_emergent_meanings_empty_without_usage():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    assert ch.get_emergent_meanings() == {}


def test_get_emergent_meanings_after_sufficient_usage():
    """符号 0 与 'alarm' 共现 6 次（>5）→ 涌现 'alarm' 含义。"""
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    for _ in range(6):
        ch.record_usage(0, event_type="alarm")
    # 也给其他符号一些使用以稀释其他桶
    for _ in range(2):
        ch.record_usage(0, event_type="other")
    meanings = ch.get_emergent_meanings()
    assert 0 in meanings
    assert meanings[0] == "alarm"


def test_get_emergent_meanings_below_threshold_not_emergent():
    """5 次以下不视为涌现。"""
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    for _ in range(5):  # 不 > 5
        ch.record_usage(0, event_type="alarm")
    meanings = ch.get_emergent_meanings()
    assert 0 not in meanings


# --------------------------------------------------------------------------- #
# 军事级修复点：encode_message 返回副本
# --------------------------------------------------------------------------- #


def test_encode_message_returns_copy_not_view():
    """核心修复点：encode_message 返回副本（非视图），修改不污染内部 embeddings。"""
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    v = ch.encode_message(0)
    # 修改返回值
    v[0] = 999.0
    # 内部 embeddings[0] 不应被修改
    assert ch.embeddings[0, 0] != 999.0


def test_encode_message_returns_distinct_arrays():
    """每次 encode_message 调用返回独立数组。"""
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    v1 = ch.encode_message(0)
    v2 = ch.encode_message(0)
    assert v1 is not v2
    # 修改 v1 不影响 v2
    v1[0] = 999.0
    assert v2[0] != 999.0


# --------------------------------------------------------------------------- #
# 线程安全
# --------------------------------------------------------------------------- #


def test_concurrent_encode_decode_does_not_crash():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)

    def encoder() -> None:
        for _ in range(50):
            v = ch.encode_message(0)
            ch.decode_observation(v)

    threads = [Thread(target=encoder) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_concurrent_update_embeddings_does_not_crash():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)
    target = np.array([1.0, 0.0, 0.0, 0.0])

    def worker() -> None:
        for _ in range(30):
            ch.update_embeddings(0, target, lr=0.01)

    threads = [Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 内部 embeddings 应有限
    assert np.isfinite(ch.embeddings).all()


def test_concurrent_record_usage_does_not_crash():
    ch = CommunicationChannel(vocab_size=5, embed_dim=4, seed=42)

    def worker(sym: int) -> None:
        for _ in range(50):
            ch.record_usage(sym, event_type="alarm")

    threads = [Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 总使用次数应是 5 × 50 = 250
    assert ch.usage_counts.sum() == 250
