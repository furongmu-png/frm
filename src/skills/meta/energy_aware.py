"""能耗感知计算：动态调整 think() 迭代次数和模块精度。

设计：
- 监控 CPU/内存/电量（在测试环境或不可用时退化为模拟值）。
- ``EnergyPolicy``：根据当前能耗模式给出 think 迭代数上限、隐空间降维系数等建议。
- ``EnergyMonitor``：暴露启用/禁用节能模式的开关。

与 ZeroDataModel 集成：
- 在 ``think()`` 进入前查询 ``recommend()``，将建议应用到该步的运行配置。
- 不直接修改模型结构，只通过参数限制工作负载。
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# 能耗模式
# ------------------------------------------------------------------ #


@dataclass
class EnergyPolicy:
    """能耗策略：根据当前模式给出运行配置建议。"""
    mode: str = "balanced"  # performance / balanced / power_saver / critical
    max_iterations: int = 10
    latent_dim_scale: float = 1.0  # 隐空间缩放系数
    precision: str = "float64"  # float64 / float32 / float16
    enable_curiosity: bool = True
    enable_consolidation: bool = True
    enable_skills: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "max_iterations": self.max_iterations,
            "latent_dim_scale": self.latent_dim_scale,
            "precision": self.precision,
            "enable_curiosity": self.enable_curiosity,
            "enable_consolidation": self.enable_consolidation,
            "enable_skills": self.enable_skills,
        }


def _policy_for(mode: str) -> EnergyPolicy:
    """根据能耗模式生成策略。"""
    presets = {
        "performance": EnergyPolicy(
            mode="performance",
            max_iterations=20,
            latent_dim_scale=1.0,
            precision="float64",
            enable_curiosity=True,
            enable_consolidation=True,
            enable_skills=True,
        ),
        "balanced": EnergyPolicy(
            mode="balanced",
            max_iterations=10,
            latent_dim_scale=1.0,
            precision="float64",
            enable_curiosity=True,
            enable_consolidation=True,
            enable_skills=True,
        ),
        "power_saver": EnergyPolicy(
            mode="power_saver",
            max_iterations=5,
            latent_dim_scale=0.5,
            precision="float32",
            enable_curiosity=False,
            enable_consolidation=False,
            enable_skills=True,
        ),
        "critical": EnergyPolicy(
            mode="critical",
            max_iterations=2,
            latent_dim_scale=0.25,
            precision="float16",
            enable_curiosity=False,
            enable_consolidation=False,
            enable_skills=False,
        ),
    }
    return presets.get(mode, presets["balanced"])


# ------------------------------------------------------------------ #
# 系统监控
# ------------------------------------------------------------------ #


def _read_cpu_usage() -> float:
    """读取 CPU 使用率（%）。

    单次读取需要两次采样并 ``sleep(0.05)``（约 50ms）。为避免每次
    ``think()`` 都阻塞 50ms，使用基于时间戳的缓存：若上次真实读取
    在 ``CPU_CACHE_SECONDS`` 内，直接返回缓存值。
    """
    now = time.monotonic()
    cached_at, cached_val = _CPU_CACHE
    if now - cached_at < CPU_CACHE_SECONDS:
        return cached_val
    val = _read_cpu_usage_raw()
    _CPU_CACHE[0] = now
    _CPU_CACHE[1] = val
    return val


def _read_cpu_usage_raw() -> float:
    """真正的 CPU 使用率采样（含 50ms sleep）。"""
    try:
        # Linux: /proc/stat
        with open("/proc/stat") as f:
            line1 = f.readline()
        fields = line1.split()[1:]
        if len(fields) < 4:
            return -1.0
        # user + nice + system = busy, idle = idle
        vals = [int(v) for v in fields[:4]]
        total = sum(vals)
        idle = vals[3]
        time.sleep(0.05)
        with open("/proc/stat") as f:
            line2 = f.readline()
        fields2 = line2.split()[1:]
        vals2 = [int(v) for v in fields2[:4]]
        total2 = sum(vals2)
        idle2 = vals2[3]
        if total2 == total:
            return 0.0
        busy = (total2 - idle2) - (total - idle)
        return float(busy) / max(total2 - total, 1) * 100.0
    except (OSError, ValueError, IndexError):
        return -1.0


#: CPU 读取的时间缓存：[last_read_monotonic, last_value]。
#: 两次真实采样之间复用上次结果，避免每次 think() 阻塞 50ms。
CPU_CACHE_SECONDS: float = 1.0
_CPU_CACHE: list = [0.0, -1.0]


def _read_memory_usage() -> float:
    """读取内存使用率（%）。失败返回 -1。"""
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        info = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        if total <= 0:
            return -1.0
        used = total - avail
        return used / total * 100.0
    except (OSError, ValueError, IndexError):
        return -1.0


def _read_battery_level() -> float:
    """读取电池电量（%）。失败返回 100（视为插电）。"""
    # Linux: /sys/class/power_supply/BAT*/capacity
    try:
        import glob
        paths = glob.glob("/sys/class/power_supply/BAT*/capacity")
        if not paths:
            return 100.0
        with open(paths[0]) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return 100.0


# ------------------------------------------------------------------ #
# EnergyMonitor
# ------------------------------------------------------------------ #


class EnergyMonitor(SkillBase):
    """能耗监控与策略建议。

    工作流：
    1. ``update()``：从系统读取 CPU/内存/电量，存入历史。
    2. ``decide_mode()``：根据当前系统状态决定能耗模式。
    3. ``recommend()``：返回 :class:`EnergyPolicy`。
    4. ``set_force_mode(mode)``：手动覆盖。
    """

    name = "energy_aware"
    dimension = "meta"

    def __init__(
        self,
        *,
        cpu_threshold_high: float = 85.0,
        cpu_threshold_low: float = 30.0,
        mem_threshold_high: float = 90.0,
        battery_threshold_low: float = 30.0,
        battery_threshold_critical: float = 10.0,
        history_size: int = 60,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.cpu_threshold_high = cpu_threshold_high
        self.cpu_threshold_low = cpu_threshold_low
        self.mem_threshold_high = mem_threshold_high
        self.battery_threshold_low = battery_threshold_low
        self.battery_threshold_critical = battery_threshold_critical
        self.history_size = history_size
        self._cpu_history: list[float] = []
        self._mem_history: list[float] = []
        self._battery_history: list[float] = []
        self._mode: str = "balanced"
        self._forced_mode: str | None = None
        self._current_policy: EnergyPolicy = _policy_for("balanced")
        self._cpu_reader: Callable[[], float] = _read_cpu_usage
        self._mem_reader: Callable[[], float] = _read_memory_usage
        self._battery_reader: Callable[[], float] = _read_battery_level
        self._step: int = 0

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #

    def set_readers(
        self,
        cpu: Callable[[], float] | None = None,
        mem: Callable[[], float] | None = None,
        battery: Callable[[], float] | None = None,
    ) -> None:
        """注入测试用 reader。"""
        if cpu is not None:
            self._cpu_reader = cpu
        if mem is not None:
            self._mem_reader = mem
        if battery is not None:
            self._battery_reader = battery

    def update(self) -> dict[str, float]:
        """读取当前系统状态并更新历史。"""
        cpu = float(self._cpu_reader())
        mem = float(self._mem_reader())
        battery = float(self._battery_reader())
        self._cpu_history.append(cpu)
        self._mem_history.append(mem)
        self._battery_history.append(battery)
        if len(self._cpu_history) > self.history_size:
            self._cpu_history = self._cpu_history[-self.history_size :]
            self._mem_history = self._mem_history[-self.history_size :]
            self._battery_history = self._battery_history[-self.history_size :]
        self._step += 1
        return {"cpu": cpu, "memory": mem, "battery": battery}

    def decide_mode(self) -> str:
        """根据当前系统状态决定能耗模式。"""
        if self._forced_mode is not None:
            return self._forced_mode
        if not self._battery_history:
            return "balanced"
        battery = self._battery_history[-1]
        cpu = self._cpu_history[-1] if self._cpu_history else 0.0
        mem = self._mem_history[-1] if self._mem_history else 0.0

        # 电池优先级最高
        if battery < self.battery_threshold_critical:
            return "critical"
        if battery < self.battery_threshold_low:
            return "power_saver"
        # 高负载 → 平衡
        if cpu > self.cpu_threshold_high or mem > self.mem_threshold_high:
            return "balanced"
        # 低负载 → 性能模式
        if cpu < self.cpu_threshold_low:
            return "performance"
        return "balanced"

    def recommend(self) -> EnergyPolicy:
        """返回当前能耗策略。"""
        mode = self.decide_mode()
        if mode != self._mode:
            self._mode = mode
            self._current_policy = _policy_for(mode)
            logger.info("energy mode -> %s", mode)
        return self._current_policy

    def set_force_mode(self, mode: str | None) -> None:
        """手动覆盖能耗模式。传 None 取消覆盖。"""
        if mode is not None and mode not in (
            "performance",
            "balanced",
            "power_saver",
            "critical",
        ):
            raise ValueError(f"unknown energy mode: {mode!r}")
        self._forced_mode = mode

    def current_mode(self) -> str:
        return self._forced_mode or self.decide_mode()

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.current_mode(),
            "cpu": self._cpu_history[-1] if self._cpu_history else -1.0,
            "memory": self._mem_history[-1] if self._mem_history else -1.0,
            "battery": self._battery_history[-1] if self._battery_history else 100.0,
            "avg_cpu": (
                float(np.mean(self._cpu_history)) if self._cpu_history else 0.0
            ),
            "forced": self._forced_mode is not None,
            "step": self._step,
        }

    def estimate_battery_remaining_min(self) -> float | None:
        """根据电量历史粗略估算续航（分钟）。

        若电量稳定不变（插电状态），返回 None。
        """
        if len(self._battery_history) < 5:
            return None
        recent = self._battery_history[-10:]
        if len(recent) < 2:
            return None
        slope = float(np.polyfit(range(len(recent)), recent, 1)[0])
        if slope >= -1e-3:
            return None  # 电量稳定或上升
        current = recent[-1]
        # 剩余分钟 = 当前电量 / 单步电量下降速率 * 单步时间（假设 1 秒一步）
        return max(0.0, current / abs(slope))

    # ------------------------------------------------------------------ #
    # SkillBase 接口
    # ------------------------------------------------------------------ #

    def process(self, ctx: SkillContext) -> SkillResult:
        self.update()
        policy = self.recommend()
        return SkillResult(
            name=self.name,
            data={
                "stats": self.stats(),
                "policy": policy.to_dict(),
                "battery_remaining_min": self.estimate_battery_remaining_min(),
            },
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "ready": True,
            "data": {
                "mode": self.current_mode(),
                "battery": (
                    self._battery_history[-1] if self._battery_history else 100.0
                ),
                "cpu": self._cpu_history[-1] if self._cpu_history else -1.0,
                "memory": self._mem_history[-1] if self._mem_history else -1.0,
                "forced": self._forced_mode is not None,
                "step": self._step,
            },
        }
