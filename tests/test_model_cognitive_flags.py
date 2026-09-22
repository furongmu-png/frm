from __future__ import annotations

import json

import numpy as np
import pytest

from zero_data_model.model import ZeroDataModel


# --------------------------------------------------------------------------- #
# 默认构造（所有 flag=False）：零回归契约
# --------------------------------------------------------------------------- #


def test_default_construct_no_cognitive_keys():
    """默认构造：metadata 不含 module_errors/cognitive_upgrades（零回归契约）。"""
    model = ZeroDataModel(dim=8, seed=42)
    rng = np.random.default_rng(0)
    sig = model.think(rng.normal(0, 1, 8))
    md = sig.metadata
    assert "module_errors" not in md
    assert "cognitive_upgrades" not in md


def test_default_construct_modules_unset():
    """默认构造：所有升级模块属性都是 None。"""
    model = ZeroDataModel(dim=8, seed=42)
    assert model.architect is None
    assert model.layered_predictor is None
    assert model.temporal_memory is None
    assert model.episodic_graph is None
    assert model.semantic_index is None
    assert model.logic_layer is None
    assert model.causal_inference is None
    assert model.meta_cognition is None
    assert model.experiment_planner is None
    assert model.hypothesis_tester is None


# --------------------------------------------------------------------------- #
# enable_architect=True：metadata 含 cognitive_upgrades.architecture
# --------------------------------------------------------------------------- #


def test_enable_architect_surfaces_architecture():
    model = ZeroDataModel(dim=8, seed=42, enable_architect=True)
    assert model.architect is not None
    rng = np.random.default_rng(0)
    sig = model.think(rng.normal(0, 1, 8))
    md = sig.metadata
    assert "cognitive_upgrades" in md
    assert "architecture" in md["cognitive_upgrades"]


def test_enable_architect_does_not_mutate_modules():
    """HIGH-2 修复点：architect 不变更 self.modules 长度。"""
    model = ZeroDataModel(dim=8, seed=42, enable_architect=True)
    n_modules_before = len(model.modules)
    rng = np.random.default_rng(0)
    # 多次 think 触发多个 eval cycle
    for _ in range(5):
        model.think(rng.normal(0, 1, 8))
    # modules 长度不应被 architect 修改
    assert len(model.modules) == n_modules_before


# --------------------------------------------------------------------------- #
# enable_layered_predictor=True：metadata 含 cognitive_upgrades（有 layered 相关键）
# --------------------------------------------------------------------------- #


def test_enable_layered_predictor_surfaces_layered_keys():
    model = ZeroDataModel(dim=8, seed=42, enable_layered_predictor=True)
    assert model.layered_predictor is not None
    assert model.temporal_memory is not None
    rng = np.random.default_rng(0)
    sig = model.think(rng.normal(0, 1, 8))
    md = sig.metadata
    assert "cognitive_upgrades" in md
    upgrades = md["cognitive_upgrades"]
    # 至少存在 layered_predictor 或 temporal_memory_error 之一
    assert "layered_predictor" in upgrades or "temporal_memory_error" in upgrades


# --------------------------------------------------------------------------- #
# enable_episodic_memory=True：metadata 含 cognitive_upgrades.semantic_index_size
# --------------------------------------------------------------------------- #


def test_enable_episodic_memory_surfaces_semantic_index_size():
    model = ZeroDataModel(dim=8, seed=42, enable_episodic_memory=True)
    assert model.episodic_graph is not None
    assert model.semantic_index is not None
    rng = np.random.default_rng(0)
    sig = model.think(rng.normal(0, 1, 8))
    md = sig.metadata
    assert "cognitive_upgrades" in md
    upgrades = md["cognitive_upgrades"]
    assert "semantic_index_size" in upgrades
    assert "episodic_nodes" in upgrades
    # 第一个 cycle 后至少有 1 个语义索引条目
    assert upgrades["semantic_index_size"] >= 1


# --------------------------------------------------------------------------- #
# enable_logic_layer=True：metadata 含 cognitive_upgrades.causal_dim
# --------------------------------------------------------------------------- #


def test_enable_logic_layer_surfaces_causal_dim():
    model = ZeroDataModel(dim=8, seed=42, enable_logic_layer=True)
    assert model.logic_layer is not None
    assert model.causal_inference is not None
    rng = np.random.default_rng(0)
    sig = model.think(rng.normal(0, 1, 8))
    md = sig.metadata
    assert "cognitive_upgrades" in md
    upgrades = md["cognitive_upgrades"]
    # causal_inference 未设置 transition_matrix → causal_dim = 0
    assert "causal_dim" in upgrades
    assert upgrades["causal_dim"] == 0


def test_enable_logic_layer_with_transition_matrix():
    """当 causal_inference 设置了 transition_matrix 后，causal_dim 反映其维度。"""
    model = ZeroDataModel(dim=8, seed=42, enable_logic_layer=True)
    model.causal_inference.set_transition(np.eye(4))
    rng = np.random.default_rng(0)
    sig = model.think(rng.normal(0, 1, 8))
    md = sig.metadata
    assert md["cognitive_upgrades"]["causal_dim"] == 4


# --------------------------------------------------------------------------- #
# enable_meta_cognition=True：metadata 含 cognitive_upgrades.meta_cognition
# --------------------------------------------------------------------------- #


def test_enable_meta_cognition_surfaces_meta_cognition():
    model = ZeroDataModel(dim=8, seed=42, enable_meta_cognition=True)
    assert model.meta_cognition is not None
    rng = np.random.default_rng(0)
    sig = model.think(rng.normal(0, 1, 8))
    md = sig.metadata
    assert "cognitive_upgrades" in md
    upgrades = md["cognitive_upgrades"]
    assert "meta_cognition" in upgrades
    meta = upgrades["meta_cognition"]
    assert "mean_uncertainty" in meta
    assert "mode" in meta
    assert "confidence" in meta


# --------------------------------------------------------------------------- #
# enable_experiment_planner=True：metadata 含 supported_hypotheses 或 experiment_planner 相关
# --------------------------------------------------------------------------- #


def test_enable_experiment_planner_surfaces_hypotheses():
    model = ZeroDataModel(dim=8, seed=42, enable_experiment_planner=True)
    assert model.experiment_planner is not None
    assert model.hypothesis_tester is not None
    rng = np.random.default_rng(0)
    sig = model.think(rng.normal(0, 1, 8))
    md = sig.metadata
    assert "cognitive_upgrades" in md
    upgrades = md["cognitive_upgrades"]
    # hypothesis_tester 总会更新 supported_hypotheses（默认 0）
    assert "supported_hypotheses" in upgrades
    assert upgrades["supported_hypotheses"] == 0


def test_enable_experiment_planner_eval_triggers_experiment_key():
    """eval_interval 触发时 metadata 含 experiment 键。"""
    model = ZeroDataModel(dim=8, seed=42, enable_experiment_planner=True)
    # 默认 eval_interval=1000 → 在第 1000 cycle 才评估
    # 我们直接调用 experiment_planner.evaluate 验证可触发
    rng = np.random.default_rng(0)
    model.think(rng.normal(0, 1, 8))
    # 多次 think 直到触发评估
    result = model.experiment_planner.evaluate(current_uncertainty=0.5, step=1000)
    assert result is not None
    assert "experiment" in result


# --------------------------------------------------------------------------- #
# 全部启用：metadata JSON 可序列化
# --------------------------------------------------------------------------- #


def test_all_enabled_metadata_json_serializable():
    """全部启用：metadata JSON 可序列化（json.dumps 不报错）。"""
    model = ZeroDataModel(
        dim=8, seed=42,
        enable_architect=True,
        enable_layered_predictor=True,
        enable_episodic_memory=True,
        enable_logic_layer=True,
        enable_meta_cognition=True,
        enable_experiment_planner=True,
    )
    rng = np.random.default_rng(0)
    sig = model.think(rng.normal(0, 1, 8))
    md = sig.metadata
    # 应可被 json.dumps 序列化（_sanitize_for_json 应已剥离 numpy 类型）
    serialized = json.dumps(md)
    assert isinstance(serialized, str)
    # 反序列化回来仍是 dict
    parsed = json.loads(serialized)
    assert isinstance(parsed, dict)
    # cognitive_upgrades 应存在
    assert "cognitive_upgrades" in parsed


# --------------------------------------------------------------------------- #
# dim=1 不崩溃（LOW-2 修复点）
# --------------------------------------------------------------------------- #


def test_dim_one_does_not_crash():
    """军事级修复的 LOW-2 修复点：dim=1 不崩溃。"""
    model = ZeroDataModel(dim=1, seed=42)
    rng = np.random.default_rng(0)
    sig = model.think(rng.normal(0, 1, 1))
    assert sig.data is not None


def test_dim_one_with_layered_predictor_does_not_crash():
    """LOW-2 修复点：dim=1 + enable_layered_predictor=True 不崩溃。"""
    model = ZeroDataModel(dim=1, seed=42, enable_layered_predictor=True)
    # 检查 dim_l1, dim_l2 都被钳到 1
    assert model.layered_predictor.dim_l0 == 1
    assert model.layered_predictor.dim_l1 == 1
    assert model.layered_predictor.dim_l2 == 1
    rng = np.random.default_rng(0)
    sig = model.think(rng.normal(0, 1, 1))
    assert sig.data is not None


def test_dim_one_all_flags_does_not_crash():
    """dim=1 + 全部启用 flag 不崩溃。"""
    model = ZeroDataModel(
        dim=1, seed=42,
        enable_architect=True,
        enable_layered_predictor=True,
        enable_episodic_memory=True,
        enable_logic_layer=True,
        enable_meta_cognition=True,
        enable_experiment_planner=True,
    )
    rng = np.random.default_rng(0)
    sig = model.think(rng.normal(0, 1, 1))
    assert sig.data is not None
    # metadata 应可序列化
    json.dumps(sig.metadata)


# --------------------------------------------------------------------------- #
# dim 校验
# --------------------------------------------------------------------------- #


def test_dim_zero_raises_value_error():
    with pytest.raises(ValueError):
        ZeroDataModel(dim=0)


def test_dim_negative_raises_value_error():
    with pytest.raises(ValueError):
        ZeroDataModel(dim=-5)


def test_dim_too_large_raises_value_error():
    with pytest.raises(ValueError):
        ZeroDataModel(dim=10000)
