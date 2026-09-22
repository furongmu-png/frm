"""音频场景理解技能。

扩展 AudioEncoder 为 AudioSceneEncoder：
- 环境声分类（碰撞声、摩擦声、背景噪声）
- Mel 频谱 + 统计特征 + 投影
- 声音-事件因果关联（碰撞声预测误差高时触发注意）
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)

#: 环境声类别
SCENE_CATEGORIES = ("collision", "friction", "background", "silence")


class AudioSceneEncoder(SkillBase):
    """音频场景编码器：环境声分类 + 事件因果关联。

    接受原始音频波形（1D float array），输出：
    - Mel 频谱统计特征向量
    - 场景分类概率
    - 事件标签（碰撞/摩擦/背景/静默）
    """

    name = "audio_scene"
    dimension = "perception"

    def __init__(
        self,
        *,
        latent_dim: int = 32,
        sample_rate: int = 16000,
        n_mels: int = 16,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.latent_dim = latent_dim
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self._rng = np.random.default_rng(456)

        # 固定随机投影：mel features → latent
        self._proj = self._rng.standard_normal(
            (n_mels * 4, latent_dim)  # 4 统计量 × n_mels
        ).astype(np.float64) * (1.0 / np.sqrt(n_mels * 4))

        # Mel 滤波器组（预计算）
        self._mel_filters = self._build_mel_filters()

        self._last_event: str = "silence"
        self._last_probs: np.ndarray | None = None
        self._prev_prediction_error: float = 0.0
        self._causal_attention: float = 0.0

    def _build_mel_filters(self) -> np.ndarray:
        """构建简单的三角 Mel 滤波器组。"""
        n_fft = 256
        n_mels = self.n_mels
        filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
        mel_points = np.linspace(
            self._hz_to_mel(0), self._hz_to_mel(self.sample_rate / 2), n_mels + 2
        )
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])
        bin_points = np.floor(hz_points / self.sample_rate * n_fft).astype(int)
        bin_points = np.clip(bin_points, 0, n_fft // 2)

        for m in range(n_mels):
            left = bin_points[m]
            center = bin_points[m + 1]
            right = bin_points[m + 2]
            if right == left:
                right = left + 1
            for k in range(left, center):
                if center > left:
                    filters[m, k] = (k - left) / (center - left)
            for k in range(center, right):
                if right > center:
                    filters[m, k] = (right - k) / (right - center)
        return filters

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _compute_mel_features(self, signal: np.ndarray) -> np.ndarray:
        """计算 Mel 频谱统计特征。"""
        n_fft = 256
        hop = 128
        if len(signal) < n_fft:
            signal = np.pad(signal, (0, n_fft - len(signal)))

        # 分帧 + FFT
        n_frames = max(1, (len(signal) - n_fft) // hop + 1)
        frames = np.array(
            [
                signal[i * hop : i * hop + n_fft] * np.hamming(n_fft)
                for i in range(n_frames)
            ]
        )
        if frames.size == 0:
            return np.zeros(self.n_mels * 4)

        spectrum = np.abs(np.fft.rfft(frames, axis=1))  # (n_frames, n_fft//2+1)
        mel_spec = spectrum @ self._mel_filters.T  # (n_frames, n_mels)
        mel_spec = np.log(mel_spec + 1e-10)

        # 4 个统计量 × n_mels
        features = np.concatenate(
            [
                mel_spec.mean(axis=0),
                mel_spec.std(axis=0),
                mel_spec.max(axis=0),
                mel_spec.min(axis=0),
            ]
        )
        return features

    def classify_scene(self, features: np.ndarray) -> tuple[str, np.ndarray]:
        """分类音频场景。

        Returns
        -------
        (event_label, probabilities)
        """
        # 基于特征的简单规则分类
        energy = float(np.mean(features[: self.n_mels]))
        spectral_centroid = float(
            np.mean(features[self.n_mels : 2 * self.n_mels])
        )
        spectral_flux = float(np.mean(features[2 * self.n_mels : 3 * self.n_mels]))

        scores = np.zeros(len(SCENE_CATEGORIES), dtype=np.float64)

        if energy < -5.0:
            # 静默
            scores[3] = 1.0
        elif spectral_flux > 2.0 and energy > -2.0:
            # 碰撞声：高能量 + 高频谱通量
            scores[0] = spectral_flux + energy + 2.0
            scores[1] = 0.3
            scores[2] = 0.1
            scores[3] = 0.0
        elif spectral_centroid > 0.5 and energy > -3.0:
            # 摩擦声：中等能量 + 宽频谱
            scores[1] = spectral_centroid + 0.5
            scores[0] = 0.2
            scores[2] = 0.3
            scores[3] = 0.0
        else:
            # 背景噪声
            scores[2] = 1.0
            scores[0] = 0.1
            scores[1] = 0.1
            scores[3] = 0.0

        # softmax 归一化
        scores = np.exp(scores - scores.max())
        probs = scores / scores.sum()
        label = SCENE_CATEGORIES[int(np.argmax(probs))]
        return label, probs

    def encode(self, signal: np.ndarray) -> np.ndarray:
        """编码音频为 latent 向量。"""
        features = self._compute_mel_features(signal)
        latent = features @ self._proj
        norm = np.linalg.norm(latent)
        if norm > 1e-10:
            latent = latent / norm
        return latent

    def process(self, ctx: SkillContext) -> SkillResult:
        # 从 context 提取音频信号
        # raw_observation 可能是帧（无音频），也可能附带音频
        audio_signal = None
        if hasattr(ctx, "audio") and ctx.audio is not None:
            audio_signal = ctx.audio
        elif ctx.raw_observation is not None and isinstance(
            ctx.raw_observation, dict
        ):
            audio_signal = ctx.raw_observation.get("audio")

        if audio_signal is None:
            # 无音频输入：生成静默特征
            return SkillResult(
                name=self.name,
                data={"ready": False, "event": "no_audio"},
            )

        audio_signal = np.asarray(audio_signal, dtype=np.float64).flatten()
        features = self._compute_mel_features(audio_signal)
        label, probs = self.classify_scene(features)
        latent = self.encode(audio_signal)

        # 声音-事件因果关联：碰撞声 + 高预测误差 → 触发注意
        pe = ctx.prediction_error
        causal_attention = 0.0
        if label == "collision" and pe > self._prev_prediction_error * 1.5:
            causal_attention = min(1.0, pe)
        self._prev_prediction_error = pe
        self._causal_attention = causal_attention
        self._last_event = label
        self._last_probs = probs

        return SkillResult(
            name=self.name,
            data={
                "event": label,
                "probabilities": {
                    SCENE_CATEGORIES[i]: float(probs[i])
                    for i in range(len(SCENE_CATEGORIES))
                },
                "latent": latent.tolist(),
                "energy": float(features[: self.n_mels].mean()),
                "causal_attention": causal_attention,
                "spectral_centroid": float(
                    features[self.n_mels : 2 * self.n_mels].mean()
                ),
            },
        )
