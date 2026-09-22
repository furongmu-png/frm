"""EmbodiedBody — first-person active-perception body environment.

理论基础
========
具身认知 (embodied cognition) 认为智能体的认知必须通过一个可主动控制的
身体与物理环境交互才能涌现。本模块给 ZeroDataModel 提供一个简化的"身体"：

  - **主动视觉** : move_gaze(yaw, pitch) 改变第一人称相机视角
  - **主动触觉** : touch_probe(x, y) 在视场内指定位置进行触觉探测
  - **主动施力** : apply_force(object_id, force_vector) 对物体施加力

身体内部维持一个 2D 俯视场景（复用 PhysicsSandbox 的物理，叠加相机/触觉
模型），返回多通道观测：
  - visual_frame  : 第一人称视角裁剪 + 旋转变换后的帧 (W×W uint8)
  - tactile       : 触觉读数向量（每个 probe 点的压力值）
  - proprioception: 本体感觉 (joint_angles, joint_velocities, gaze_yaw, gaze_pitch)

重置时身体姿态（位置、朝向、关节角度）随机初始化，保证探索多样性。

主动感知动作被编码为扩展动作空间中的离散/连续维度，与原有物理动作
（左/右/上/无操作）拼接，供 CuriosityPolicy 选择。

自由能原则映射
--------------
- 每个主动感知动作 = 主动推理中的 "action"
- 感知预测误差（视觉/触觉）= 自由能的感官分量
- 本体感觉预测误差 = 身体图式 (body schema) 的自洽性度量
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# 复用已有物理沙盒（同进程内 import，无外部依赖）
import sys
from pathlib import Path

# 让 experiments/ 可被导入（与 demo_cognitive_origin.py 一致的策略）
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "experiments"))

try:
    from physics_sandbox import PhysicsSandbox, Body, Shape  # type: ignore
    _HAS_SANDBOX = True
except Exception:  # pragma: no cover - 沙盒不可用时降级
    PhysicsSandbox = None  # type: ignore
    Body = None  # type: ignore
    Shape = None  # type: ignore
    _HAS_SANDBOX = False


# ------------------------------------------------------------------ #
# 常量
# ------------------------------------------------------------------ #
DEFAULT_VIEW_SIZE = 64       # 第一人称视角裁剪大小（像素）
DEFAULT_TACTILE_GRID = 4     # 触觉网格 4x4 = 16 个 probe 点
N_PROPRIO = 8                # 本体感觉向量维度
# 主动感知动作枚举（与原沙盒 4 个动作拼接）
PERCEPTUAL_ACTIONS = [
    "move_gaze_left",
    "move_gaze_right",
    "move_gaze_up",
    "move_gaze_down",
    "touch_center",
    "touch_left",
    "touch_right",
    "apply_force_forward",
    "apply_force_left",
    "apply_force_right",
]
N_PERCEPTUAL_ACTIONS = len(PERCEPTUAL_ACTIONS)
# 原沙盒动作数
N_PHYSICS_ACTIONS = 4
# 扩展动作空间总数
N_TOTAL_ACTIONS = N_PHYSICS_ACTIONS + N_PERCEPTUAL_ACTIONS


# ------------------------------------------------------------------ #
# 数据结构
# ------------------------------------------------------------------ #
@dataclass
class EmbodiedObservation:
    """具身多通道观测。"""
    visual_frame: np.ndarray          # (W, W) uint8 第一人称视角
    tactile: np.ndarray               # (n_probe,) float64 触觉读数
    proprioception: np.ndarray       # (N_PROPRIO,) float64 本体感觉
    gaze_yaw: float                   # 当前视线 yaw（弧度）
    gaze_pitch: float                 # 当前视线 pitch（弧度）
    body_position: np.ndarray         # (2,) 身体在场景中的位置
    touched_object_id: int            # 最近触摸的物体索引（-1 = 无）
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual_frame": self.visual_frame,
            "tactile": self.tactile,
            "proprioception": self.proprioception,
            "gaze_yaw": self.gaze_yaw,
            "gaze_pitch": self.gaze_pitch,
            "body_position": self.body_position,
            "touched_object_id": self.touched_object_id,
            "metadata": self.metadata,
        }


@dataclass
class BodyAction:
    """一个主动感知动作的解析结果。"""
    action_type: str          # "physics" | "move_gaze" | "touch_probe" | "apply_force"
    params: dict[str, Any]    # 动作参数


# ------------------------------------------------------------------ #
# 主类
# ------------------------------------------------------------------ #
class EmbodiedBody:
    """具身主动感知身体环境。

    在 PhysicsSandbox 之上叠加：
      - 相机模型（yaw/pitch 旋转载剪裁出第一人称视角）
      - 触觉模型（在视场内采样 probe 点，根据像素亮度返回压力）
      - 本体感觉（关节角度/速度 + 视线角度）

    Parameters
    ----------
    view_size : int
        第一人称视角裁剪大小（默认 64）
    tactile_grid : int
        触觉网格边长（默认 4 → 16 个 probe 点）
    gaze_step : float
        单次 move_gaze 动作的角度增量（弧度，默认 0.3）
    num_objects : int
        场景物体数（传给 PhysicsSandbox，1-3）
    seed : int | None
        随机种子
    """

    def __init__(
        self,
        view_size: int = DEFAULT_VIEW_SIZE,
        tactile_grid: int = DEFAULT_TACTILE_GRID,
        gaze_step: float = 0.3,
        num_objects: int = 2,
        seed: Optional[int] = 42,
    ):
        if view_size < 8:
            raise ValueError(f"view_size must be >= 8, got {view_size}")
        if tactile_grid < 1:
            raise ValueError(f"tactile_grid must be >= 1, got {tactile_grid}")
        if gaze_step <= 0:
            raise ValueError(f"gaze_step must be > 0, got {gaze_step}")

        self.view_size = int(view_size)
        self.tactile_grid = int(tactile_grid)
        self.n_probe = self.tactile_grid * self.tactile_grid
        self.gaze_step = float(gaze_step)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        # 物理沙盒（若可用）
        if _HAS_SANDBOX:
            self.sandbox: Optional[PhysicsSandbox] = PhysicsSandbox(
                num_objects=num_objects, seed=seed
            )
        else:
            self.sandbox = None
            self._synthetic_frame = np.zeros(
                (view_size, view_size), dtype=np.uint8
            )

        # 身体状态
        self.gaze_yaw = 0.0
        self.gaze_pitch = 0.0
        self.body_position = np.zeros(2, dtype=np.float64)
        self.joint_angles = np.zeros(4, dtype=np.float64)   # 4 个虚拟关节
        self.joint_velocities = np.zeros(4, dtype=np.float64)
        self.touched_object_id = -1
        self._step_count = 0
        # B3 修复：记录最近一次沙盒同步错误供诊断（None = 无错误）
        self._last_body_sync_error: Optional[str] = None

        # 本体感觉归一化目标范围
        self._proprio_dim = N_PROPRIO

        self.reset()

    # ---------------------------------------------------------------- #
    # 重置
    # ---------------------------------------------------------------- #
    def reset(self, seed: Optional[int] = None) -> EmbodiedObservation:
        """重置环境，身体姿态随机初始化。"""
        with self._lock:
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            if self.sandbox is not None:
                self.sandbox.reset(seed=seed)
            # 随机初始化身体姿态
            self.gaze_yaw = float(self._rng.uniform(-1.0, 1.0))
            self.gaze_pitch = float(self._rng.uniform(-0.5, 0.5))
            self.body_position = self._rng.uniform(10, 118, size=2).astype(
                np.float64
            )
            self.joint_angles = self._rng.uniform(-0.5, 0.5, size=4)
            self.joint_velocities = np.zeros(4, dtype=np.float64)
            self.touched_object_id = -1
            self._step_count = 0
            self._last_body_sync_error = None
            return self._build_observation()

    # ---------------------------------------------------------------- #
    # 动作解析
    # ---------------------------------------------------------------- #
    def parse_action(self, action: int) -> BodyAction:
        """将扩展动作空间中的整数动作解析为 BodyAction。

        动作空间 = [0..3] 物理动作 + [4..13] 主动感知动作
        """
        if not (0 <= action < N_TOTAL_ACTIONS):
            raise ValueError(
                f"action must be in [0, {N_TOTAL_ACTIONS}), got {action}"
            )
        if action < N_PHYSICS_ACTIONS:
            return BodyAction(action_type="physics", params={"raw": action})
        perceptual_idx = action - N_PHYSICS_ACTIONS
        name = PERCEPTUAL_ACTIONS[perceptual_idx]
        if name.startswith("move_gaze"):
            direction = name.split("_")[-1]
            return BodyAction(
                action_type="move_gaze",
                params={"direction": direction, "step": self.gaze_step},
            )
        if name.startswith("touch"):
            # 触觉 probe 位置由 direction 决定
            direction = name.split("_")[-1]
            offsets = {
                "center": (0.0, 0.0),
                "left": (-0.25, 0.0),
                "right": (0.25, 0.0),
            }
            dx, dy = offsets.get(direction, (0.0, 0.0))
            return BodyAction(
                action_type="touch_probe",
                params={"dx": dx, "dy": dy},
            )
        if name.startswith("apply_force"):
            direction = name.split("_")[-1]
            force_map = {
                "forward": np.array([0.0, 1.0]),
                "left": np.array([-1.0, 0.0]),
                "right": np.array([1.0, 0.0]),
            }
            fv = force_map.get(direction, np.array([0.0, 1.0]))
            return BodyAction(
                action_type="apply_force",
                params={"object_id": 0, "force_vector": fv * 2.0},
            )
        # B2 修复：原代码默认返回 physics action 3（no-op），这会
        # 静默吞下未识别的动作，使 PERCEPTUAL_ACTIONS 表与解析逻辑
        # 脱节时不报错。改为显式抛 ValueError，让调用方立即发现
        # 配置错误（军事级：失败显式而非静默）。
        raise ValueError(
            f"unrecognized perceptual action name: {name!r} "
            f"(idx={perceptual_idx}, action={action})"
        )

    # ---------------------------------------------------------------- #
    # 主动感知动作
    # ---------------------------------------------------------------- #
    def move_gaze(self, yaw_delta: float, pitch_delta: float) -> None:
        """改变相机视角。"""
        with self._lock:
            self.gaze_yaw = float(np.clip(self.gaze_yaw + yaw_delta, -2.0, 2.0))
            self.gaze_pitch = float(
                np.clip(self.gaze_pitch + pitch_delta, -1.0, 1.0)
            )

    def touch_probe(self, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        """在视场内偏移 (dx, dy) 处进行触觉探测，返回触觉读数向量。

        触觉模型：在第一人称视角帧上采样 tactile_grid×tactile_grid 个点，
        用像素亮度作为压力代理（亮 = 接触物体，暗 = 空气）。
        """
        with self._lock:
            frame = self._render_first_person()
            n = self.tactile_grid
            h, w = frame.shape
            # 在帧中心 ± offset 采样网格
            cx, cy = w // 2 + int(dx * w), h // 2 + int(dy * h)
            # B1 修复：原代码用 `range(-half, half)` 产生 2*half 个点，
            # 对奇数 tactile_grid（如 n=5）→ half=2 → 4 个点而非 5 个，
            # 漏采最外层。改用 np.linspace 对称采样恰好 n 个点。
            # 同时确保偏移不超出帧范围。
            n_eff = min(n, w, h)
            # 在 [-half, +half] 上取 n_eff 个对称点（含端点）
            half = (n_eff - 1) / 2.0
            offsets = np.linspace(-half, half, n_eff)
            readings = np.zeros(self.n_probe, dtype=np.float64)
            idx = 0
            for i in offsets:
                for j in offsets:
                    px = int(np.clip(cx + int(round(j)), 0, w - 1))
                    py = int(np.clip(cy + int(round(i)), 0, h - 1))
                    if idx < self.n_probe:
                        readings[idx] = float(frame[py, px]) / 255.0
                        idx += 1
            # 检测是否触摸到物体（亮度 > 阈值）
            if readings.mean() > 0.6 and self.sandbox is not None:
                self.touched_object_id = 0
            return readings

    def apply_force(
        self, object_id: int, force_vector: np.ndarray
    ) -> dict[str, Any]:
        """对指定物体施加力。"""
        with self._lock:
            if self.sandbox is None:
                return {"applied": False, "reason": "no_sandbox"}
            try:
                bodies = getattr(self.sandbox, "bodies", [])
                if not (0 <= object_id < len(bodies)):
                    return {"applied": False, "reason": "bad_object_id"}
                fv = np.asarray(force_vector, dtype=np.float64).flatten()
                if fv.size >= 2:
                    bodies[object_id].vx += float(fv[0])
                    bodies[object_id].vy += float(fv[1])
                return {"applied": True, "object_id": object_id}
            except Exception as exc:
                return {"applied": False, "reason": str(exc)}

    # ---------------------------------------------------------------- #
    # step
    # ---------------------------------------------------------------- #
    def step(self, action: int) -> EmbodiedObservation:
        """执行一个扩展动作并返回多通道观测。

        Parameters
        ----------
        action : int
            [0..3] 物理动作, [4..13] 主动感知动作
        """
        with self._lock:
            self._step_count += 1
            parsed = self.parse_action(action)

            if parsed.action_type == "physics":
                if self.sandbox is not None:
                    self.sandbox.step(parsed.params["raw"])
                # 身体跟随 agent（若有沙盒，body_position = agent 位置）
                self._update_body_from_sandbox()
                # 关节速度衰减
                self.joint_velocities *= 0.9

            elif parsed.action_type == "move_gaze":
                direction = parsed.params["direction"]
                step = parsed.params["step"]
                deltas = {
                    "left": (-step, 0.0),
                    "right": (step, 0.0),
                    "up": (0.0, step),
                    "down": (0.0, -step),
                }
                dyaw, dpitch = deltas.get(direction, (0.0, 0.0))
                self.move_gaze(dyaw, dpitch)
                # 移动视线时关节角度联动（模拟头颈）
                self.joint_velocities[0] = dyaw / step if step > 0 else 0.0
                self.joint_velocities[1] = dpitch / step if step > 0 else 0.0
                self.joint_angles[0] += dyaw
                self.joint_angles[1] += dpitch

            elif parsed.action_type == "touch_probe":
                readings = self.touch_probe(
                    parsed.params["dx"], parsed.params["dy"]
                )
                # 触觉动作消耗关节能量
                self.joint_velocities[2] = float(readings.mean())
                self.joint_angles[2] += 0.1

            elif parsed.action_type == "apply_force":
                self.apply_force(
                    parsed.params["object_id"],
                    parsed.params["force_vector"],
                )
                if self.sandbox is not None:
                    self.sandbox.step(3)  # no-op 让物理推进
                self._update_body_from_sandbox()
                self.joint_velocities[3] = 0.5
                self.joint_angles[3] += 0.05

            return self._build_observation()

    # ---------------------------------------------------------------- #
    # 内部：构建观测
    # ---------------------------------------------------------------- #
    def _render_first_person(self) -> np.ndarray:
        """渲染第一人称视角帧（裁剪 + 旋转近似）。"""
        if self.sandbox is not None:
            full = self.sandbox.render()
        else:
            full = self._synthetic_frame
        h, w = full.shape
        cx = int(np.clip(self.body_position[0], self.view_size // 2, w - self.view_size // 2))
        cy = int(np.clip(self.body_position[1], self.view_size // 2, h - self.view_size // 2))
        half = self.view_size // 2
        # 用 gaze_yaw 做简单的水平偏移（模拟转头）
        yaw_offset = int(self.gaze_yaw * 10)
        x0 = int(np.clip(cx - half + yaw_offset, 0, w - self.view_size))
        y0 = int(np.clip(cy - half, 0, h - self.view_size))
        frame = full[y0:y0 + self.view_size, x0:x0 + self.view_size]
        if frame.shape != (self.view_size, self.view_size):
            frame = np.resize(frame, (self.view_size, self.view_size))
        return frame.astype(np.uint8)

    def _update_body_from_sandbox(self) -> None:
        """从沙盒同步身体位置（agent = bodies[0]）。"""
        if self.sandbox is None:
            return
        # B3 修复：原代码用 `except Exception: pass` 吞下所有错误，
        # 导致沙盒状态损坏时无法诊断。改为记录到 metadata.last_error
        # 供前端面板显示，同时保留 body_position 的上一个有效值
        # （军事级：错误可见但行为降级，不静默失败）。
        try:
            bodies = getattr(self.sandbox, "bodies", [])
            if bodies:
                new_pos = np.array(
                    [bodies[0].cx, bodies[0].cy], dtype=np.float64
                )
                if np.all(np.isfinite(new_pos)):
                    self.body_position = new_pos
                else:
                    self._last_body_sync_error = "non_finite_position"
        except Exception as exc:
            self._last_body_sync_error = f"{type(exc).__name__}: {exc}"

    def _build_observation(self) -> EmbodiedObservation:
        """组装多通道观测。"""
        visual = self._render_first_person()
        # 触觉读数（无显式 touch 时用帧中心采样作为默认）
        tactile = self._sample_tactile_from_frame(visual)
        proprio = self._encode_proprioception()
        return EmbodiedObservation(
            visual_frame=visual,
            tactile=tactile,
            proprioception=proprio,
            gaze_yaw=self.gaze_yaw,
            gaze_pitch=self.gaze_pitch,
            body_position=self.body_position.copy(),
            touched_object_id=self.touched_object_id,
            metadata={"step": self._step_count},
        )

    def _sample_tactile_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """从帧中心采样触觉网格。"""
        n = self.tactile_grid
        h, w = frame.shape
        cx, cy = w // 2, h // 2
        # B1 修复：与 touch_probe 一致，用 linspace 采样恰好 n 个点。
        n_eff = min(n, w, h)
        half = (n_eff - 1) / 2.0
        offsets = np.linspace(-half, half, n_eff)
        readings = np.zeros(self.n_probe, dtype=np.float64)
        idx = 0
        for i in offsets:
            for j in offsets:
                if idx < self.n_probe:
                    px = int(np.clip(cx + int(round(j)), 0, w - 1))
                    py = int(np.clip(cy + int(round(i)), 0, h - 1))
                    readings[idx] = float(frame[py, px]) / 255.0
                    idx += 1
        return readings

    def _encode_proprioception(self) -> np.ndarray:
        """编码本体感觉为 N_PROPRIO 维向量。

        [joint_angles(4), joint_velocities(4)] 归一化到 [-1, 1]
        """
        vec = np.zeros(self._proprio_dim, dtype=np.float64)
        vec[:4] = np.tanh(self.joint_angles)
        vec[4:8] = np.tanh(self.joint_velocities)
        # NaN 防护
        if not np.all(np.isfinite(vec)):
            vec = np.zeros(self._proprio_dim, dtype=np.float64)
        return vec

    # ---------------------------------------------------------------- #
    # 查询接口
    # ---------------------------------------------------------------- #
    def get_state_snapshot(self) -> dict[str, Any]:
        """返回身体当前状态快照（用于可视化）。"""
        with self._lock:
            return {
                "gaze_yaw": self.gaze_yaw,
                "gaze_pitch": self.gaze_pitch,
                "body_position": self.body_position.tolist(),
                "joint_angles": self.joint_angles.tolist(),
                "joint_velocities": self.joint_velocities.tolist(),
                "touched_object_id": self.touched_object_id,
                "step": self._step_count,
                "sandbox_attached": self.sandbox is not None,
                "n_total_actions": N_TOTAL_ACTIONS,
                "n_perceptual_actions": N_PERCEPTUAL_ACTIONS,
                "last_body_sync_error": self._last_body_sync_error,
            }

    @property
    def action_space_size(self) -> int:
        return N_TOTAL_ACTIONS

    @property
    def observation_dim(self) -> int:
        """平坦化后的观测总维度（视觉 + 触觉 + 本体感觉）。"""
        return (
            self.view_size * self.view_size + self.n_probe + self._proprio_dim
        )

    # B4 修复：hierarchical_model 调用 `get_snapshot()` 而非
    # `get_state_snapshot()`，导致 body 状态在 metadata 中永远为 `{}`。
    # 添加别名以同时兼容两个调用名（军事级：API 契约对齐）。
    def get_snapshot(self) -> dict[str, Any]:
        """``get_state_snapshot`` 的别名，供 ``hierarchical_model`` 调用。"""
        return self.get_state_snapshot()
