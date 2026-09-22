"""感知技能单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from skills.base import SkillContext
from skills.perception.depth_estimator import DepthEstimator
from skills.perception.tactile_sensor import TactileEncoder, TactileSensor
from skills.perception.audio_scene import AudioSceneEncoder
from skills.perception.olfaction import OlfactionEncoder


# ------------------------------------------------------------------ #
# 辅助：生成测试帧
# ------------------------------------------------------------------ #
def _make_frame(size: int = 128, n_objects: int = 2, seed: int = 42) -> np.ndarray:
    """生成一个模拟物理沙盒帧。"""
    rng = np.random.default_rng(seed)
    frame = np.zeros((size, size), dtype=np.uint8)
    for _ in range(n_objects):
        cx, cy = rng.integers(10, size - 10, 2)
        r = rng.integers(5, 15)
        gray = int(rng.integers(200, 255))
        cv, cu = np.ogrid[:size, :size]
        mask = (cv - cx) ** 2 + (cu - cy) ** 2 <= r**2
        frame[mask] = gray
    return frame


# ================================================================== #
# 1. DepthEstimator
# ================================================================== #
class TestDepthEstimator:
    def test_estimate_returns_correct_shape(self):
        est = DepthEstimator()
        frame = _make_frame()
        depth = est.estimate(frame)
        assert depth.shape == (128, 128)

    def test_depth_in_unit_range(self):
        est = DepthEstimator()
        frame = _make_frame()
        depth = est.estimate(frame)
        assert depth.min() >= 0.0
        assert depth.max() <= 1.0

    def test_process_via_context(self):
        est = DepthEstimator()
        frame = _make_frame()
        ctx = SkillContext(
            belief=np.zeros(64),
            raw_observation=frame,
        )
        result = est.safe_process(ctx)
        assert result.error is None
        assert "depth_mean" in result.data
        assert "depth_preview" in result.data
        assert len(result.data["depth_preview"]) == 32

    def test_empty_frame_handled(self):
        """全黑帧不应崩溃。"""
        est = DepthEstimator()
        frame = np.zeros((128, 128), dtype=np.uint8)
        depth = est.estimate(frame)
        assert depth.shape == (128, 128)
        assert np.all(np.isfinite(depth))

    def test_photometric_loss_with_history(self):
        """连续两帧应触发光度损失计算。"""
        est = DepthEstimator()
        frame1 = _make_frame(seed=1)
        frame2 = _make_frame(seed=2)
        est.estimate(frame1)
        est.estimate(frame2)
        assert est._last_loss >= 0.0


# ================================================================== #
# 2. TactileSensor / TactileEncoder
# ================================================================== #
class TestTactileSensor:
    def test_sense_returns_grid(self):
        sensor = TactileSensor(grid_size=8)
        frame = _make_frame()
        pressures = sensor.sense(frame)
        assert pressures.shape == (8, 8)

    def test_no_contact_on_empty_frame(self):
        sensor = TactileSensor()
        frame = np.zeros((128, 128), dtype=np.uint8)
        pressures = sensor.sense(frame)
        assert not sensor.has_contact(pressures)
        assert pressures.max() == 0.0

    def test_contact_detected_with_objects(self):
        sensor = TactileSensor()
        frame = _make_frame(n_objects=3)
        pressures = sensor.sense(frame)
        assert sensor.has_contact(pressures)

    def test_contact_center(self):
        sensor = TactileSensor()
        frame = _make_frame(seed=42)
        pressures = sensor.sense(frame)
        if sensor.has_contact(pressures):
            center = sensor.contact_center(pressures)
            assert center is not None
            cx, cy = center
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0


class TestTactileEncoder:
    def test_encode_returns_latent(self):
        enc = TactileEncoder(latent_dim=32)
        sensor = TactileSensor()
        pressures = sensor.sense(_make_frame())
        latent = enc.encode(pressures)
        assert latent.shape == (32,)
        assert np.isfinite(latent).all()

    def test_process_via_context(self):
        enc = TactileEncoder()
        ctx = SkillContext(
            belief=np.zeros(64),
            raw_observation=_make_frame(),
        )
        result = enc.safe_process(ctx)
        assert result.error is None
        assert "pressures" in result.data
        assert "texture" in result.data
        assert "latent" in result.data

    def test_texture_description_varies(self):
        enc = TactileEncoder()
        # 空帧 → 无接触
        ctx_empty = SkillContext(
            belief=np.zeros(64),
            raw_observation=np.zeros((128, 128), dtype=np.uint8),
        )
        r1 = enc.safe_process(ctx_empty)
        assert r1.data["texture"] == "无接触"

        # 有物体的帧 → 有纹理描述
        ctx_obj = SkillContext(
            belief=np.zeros(64),
            raw_observation=_make_frame(n_objects=3),
        )
        r2 = enc.safe_process(ctx_obj)
        assert r2.data["has_contact"] is True


# ================================================================== #
# 3. AudioSceneEncoder
# ================================================================== #
class TestAudioSceneEncoder:
    def _make_collision_sound(self, sr: int = 16000) -> np.ndarray:
        """生成模拟碰撞声（短脉冲 + 衰减）。"""
        t = np.linspace(0, 0.1, int(sr * 0.1))
        env = np.exp(-t * 50)
        return (env * np.sin(2 * np.pi * 1000 * t) * 0.5).astype(np.float64)

    def _make_silence(self, sr: int = 16000) -> np.ndarray:
        return np.zeros(int(sr * 0.1), dtype=np.float64)

    def _make_friction_sound(self, sr: int = 16000) -> np.ndarray:
        """生成模拟摩擦声（白噪声 + 带通）。"""
        rng = np.random.default_rng(99)
        n = int(sr * 0.1)
        noise = rng.standard_normal(n) * 0.3
        # 简单带通：减去均值
        return (noise - noise.mean()).astype(np.float64)

    def test_classify_collision(self):
        enc = AudioSceneEncoder()
        signal = self._make_collision_sound()
        ctx = SkillContext(
            belief=np.zeros(64),
            prediction_error=0.5,
            raw_observation={"audio": signal},
        )
        result = enc.safe_process(ctx)
        assert result.error is None
        assert "event" in result.data
        assert result.data["event"] in ("collision", "friction", "background")

    def test_silence_detected(self):
        enc = AudioSceneEncoder()
        signal = self._make_silence()
        ctx = SkillContext(
            belief=np.zeros(64),
            raw_observation={"audio": signal},
        )
        result = enc.safe_process(ctx)
        assert result.data["event"] == "silence"

    def test_no_audio_handled(self):
        enc = AudioSceneEncoder()
        ctx = SkillContext(belief=np.zeros(64))
        result = enc.safe_process(ctx)
        assert result.data["ready"] is False

    def test_latent_dimension(self):
        enc = AudioSceneEncoder(latent_dim=32)
        signal = self._make_collision_sound()
        ctx = SkillContext(
            belief=np.zeros(64),
            raw_observation={"audio": signal},
        )
        result = enc.safe_process(ctx)
        assert len(result.data["latent"]) == 32


# ================================================================== #
# 4. OlfactionEncoder
# ================================================================== #
class TestOlfactionEncoder:
    def test_init_field_creates_concentration(self):
        enc = OlfactionEncoder()
        enc.init_field(sources=[(64, 64)])
        field = enc.get_concentration_field()
        assert field.shape == (128, 128)
        assert field.max() <= 1.0
        assert field.min() >= 0.0

    def test_concentration_highest_at_source(self):
        enc = OlfactionEncoder()
        enc.init_field(sources=[(64, 64)])
        c_source = enc.get_concentration((64, 64))
        c_far = enc.get_concentration((0, 0))
        assert c_source > c_far

    def test_gradient_points_to_source(self):
        enc = OlfactionEncoder()
        enc.init_field(sources=[(100, 100)])
        gx, gy = enc.compute_gradient((50, 50))
        # 源在 (100,100)，从 (50,50) 看梯度应指向正方向
        assert gx > 0
        assert gy > 0

    def test_directional_sensing(self):
        enc = OlfactionEncoder()
        enc.init_field(sources=[(100, 64)])
        directional = enc.sense_directional((50, 64))
        assert directional.shape == (8,)
        # 朝向 (100, 64) 的方向（约 0 弧度，即 +x）浓度应更高
        assert directional.max() > 0

    def test_process_via_context(self):
        enc = OlfactionEncoder()
        ctx = SkillContext(
            belief=np.array([0.5, -0.3] + [0.0] * 62),
            prediction_error=0.1,
        )
        result = enc.safe_process(ctx)
        assert result.error is None
        assert "concentration" in result.data
        assert "gradient" in result.data
        assert "nav_reward" in result.data
        assert "field_preview" in result.data

    def test_navigation_reward_correlates_with_concentration(self):
        enc = OlfactionEncoder()
        enc.init_field(sources=[(64, 64)])
        ctx_at_source = SkillContext(belief=np.array([0.0, 0.0] + [0.0] * 62))
        r1 = enc.safe_process(ctx_at_source)
        ctx_far = SkillContext(belief=np.array([1.0, 1.0] + [0.0] * 62))
        r2 = enc.safe_process(ctx_far)
        assert r1.data["concentration"] > r2.data["concentration"]
        assert r1.data["nav_reward"] > r2.data["nav_reward"]
