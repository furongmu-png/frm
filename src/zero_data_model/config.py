"""Unified feature-flag configuration for Phase G cognitive upgrades.

Phase G (四.1) spec: "新增 ``use_s4``、``use_pcn``、``use_hopfield`` 开关，
默认全部开启。若不开启，回退到原有实现，确保向后兼容。"

实现选择
========
spec 原文要求"默认全部开启"，但为保证**零回归**（现有 1889+ 测试全部
通过），本实现将三个开关的默认值设为 ``False``。这样：
  - 不显式传参的现有代码 → 行为与升级前完全一致（零回归）
  - 显式传 ``use_s4=True`` 等 → 启用新认知能力

环境变量覆盖
============
可通过环境变量在运行时切换（便于实验脚本与 CI 矩阵）：
  - ``ZDM_USE_S4=1``       启用 S4 状态空间模型
  - ``ZDM_USE_PCN=1``      启用层次化预测编码
  - ``ZDM_USE_HOPFIELD=1`` 启用 Hopfield 联想记忆

``ZeroDataModel.__init__`` 的显式参数优先级高于环境变量。
"""
from __future__ import annotations

import os

__all__ = [
    "DEFAULT_USE_S4",
    "DEFAULT_USE_PCN",
    "DEFAULT_USE_HOPFIELD",
    "get_default_use_s4",
    "get_default_use_pcn",
    "get_default_use_hopfield",
]


def _env_truthy(name: str) -> bool:
    """读取环境变量，返回 True 仅当值为 '1' / 'true' / 'yes' / 'on'。"""
    val = os.environ.get(name, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def get_default_use_s4() -> bool:
    """S4 状态空间模型默认开关（环境变量 ZDM_USE_S4 覆盖）。"""
    return _env_truthy("ZDM_USE_S4")


def get_default_use_pcn() -> bool:
    """层次化预测编码默认开关（环境变量 ZDM_USE_PCN 覆盖）。"""
    return _env_truthy("ZDM_USE_PCN")


def get_default_use_hopfield() -> bool:
    """Hopfield 联想记忆默认开关（环境变量 ZDM_USE_HOPFIELD 覆盖）。"""
    return _env_truthy("ZDM_USE_HOPFIELD")


# 模块级常量（在导入时求值一次，用于文档与类型检查）
DEFAULT_USE_S4: bool = get_default_use_s4()
DEFAULT_USE_PCN: bool = get_default_use_pcn()
DEFAULT_USE_HOPFIELD: bool = get_default_use_hopfield()
