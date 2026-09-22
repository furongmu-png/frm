#!/usr/bin/env python3
"""认知起源完整演示：一键运行并自动记录素材。

四阶段闭环：
  1. 物理直觉涌现（沙盒探索 ~5000 步）
  2. 文本宇宙探索（百科阅读 ~5000 步）
  3. 跨模态统一（对齐训练 ~3000 步）
  4. 自我提问与知识整合（百科问答 ~2000 步）

所有可视化素材（帧图像、自由能曲线、知识图谱快照、3D 隐空间截图、
问答日志）自动保存到 ``output_dir``，无需人工干预。

用法::

    python demo_cognitive_origin.py                    # 全量 ~15000 步
    python demo_cognitive_origin.py --steps 200        # 快速烟雾测试
    python demo_cognitive_origin.py --no-streamer       # 禁用 WebSocket

依赖：numpy, PIL (Pillow)。无 GPU 需求。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ #
# sys.path 配置：让 experiments/ 和 src/ 下的模块可被直接导入
# ------------------------------------------------------------------ #
_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR / "experiments"))
sys.path.insert(0, str(_SCRIPT_DIR / "src"))

# 实验组件
from physics_sandbox import PhysicsSandbox          # noqa: E402
from text_stream import TextStream, N_NAV_ACTIONS     # noqa: E402
from text_encoder import TextEncoder                  # noqa: E402
from image_preprocessor import ImagePreprocessor       # noqa: E402
from multimodal_bridge import MultimodalBridge        # noqa: E402
from experience_buffer import ExperienceBuffer         # noqa: E402
from aligned_data_generator import EventDetector       # noqa: E402

# 核心模型
from zero_data_model import ZeroDataModel              # noqa: E402
from zero_data_model.deep_memory import DeepMemoryModule  # noqa: E402

# 可视化
from visualization import SnapshotCollector, WebSocketStreamer  # noqa: E402
from visualization.milestone_detector import MilestoneDetector  # noqa: E402

# QA 阶段（惰性导入，避免在物理阶段就加载）
# from active_inquiry import InquiryLoop
# from text_decoder import TextDecoder
# from knowledge_graph_builder import KnowledgeGraphBuilder
# from encyclopedia_reader import EncyclopediaReader


# ------------------------------------------------------------------ #
# 好奇心驱动策略
# ------------------------------------------------------------------ #
class CuriosityPolicy:
    """好奇心驱动的动作选择（β 衰减 + 访问计数探索奖励）。

    模型 ``think()`` 返回 ``Signal`` 对象而非 (action, info) 元组，
    因此动作选择由本类独立完成。策略：
      - 维护每个动作的 Q 值（移动平均奖励 = 负自由能）
      - softmax 策略，温度由 β 控制（随步数衰减 → 逐步从探索转向利用）
      - 访问计数探索奖励：久未访问的动作获得探索加成
    """

    def __init__(self, n_actions: int, beta: float = 1.0,
                 beta_decay: float = 0.9999, seed: int = 42):
        self.n_actions = n_actions
        self.beta = beta
        self.beta_decay = beta_decay
        self._rng = np.random.default_rng(seed)
        self._q = np.zeros(n_actions)
        self._visit = np.zeros(n_actions, dtype=np.int64)

    def select(self, prediction_error: float = 0.0) -> int:
        self.beta *= self.beta_decay
        # 探索奖励：prediction_error 越高 → 探索欲望越强
        exploration_bonus = prediction_error / (self._visit + 1.0) * 0.5
        logits = self.beta * self._q + exploration_bonus
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        action = int(self._rng.choice(self.n_actions, p=probs))
        self._visit[action] += 1
        return action

    def update(self, action: int, reward: float) -> None:
        """reward = -free_energy（FE 越低 → 奖励越高）"""
        alpha = 0.1
        self._q[action] = (1 - alpha) * self._q[action] + alpha * reward


# ------------------------------------------------------------------ #
# 主演示系统
# ------------------------------------------------------------------ #
class CognitiveOriginDemo:
    """认知起源四阶段演示。

    Parameters
    ----------
    output_dir : str
        素材输出目录。
    dim : int
        模型隐空间维度（所有编码器输出维度需匹配）。
    streamer : bool
        是否启动 WebSocket 实时推流（前端可视化）。
    """

    def __init__(self, output_dir: str = "demo_output",
                 dim: int = 64, streamer: bool = True,
                 use_deep_memory: bool = True):
        self.output_dir = Path(output_dir)
        self._mkdirs()

        # ---- 核心模型 ---- #
        self.model = ZeroDataModel(dim=dim, seed=42)
        self.dim = dim

        # ---- 深度认知记忆（S4 + PCN + Hopfield 协调器，第一阶段升级） ---- #
        # 独立于 ZeroDataModel 的 use_s4/use_pcn/use_hopfield 开关：
        # DeepMemoryModule 是一个外壳，把三个已实现的组件协调成统一循环。
        self.deep_memory: DeepMemoryModule | None = None
        if use_deep_memory:
            self.deep_memory = DeepMemoryModule(
                dim=dim,
                s4_state_dim=2 * dim,
                hopfield_capacity=2048,
                novelty_threshold=0.7,
                store_fe_threshold=2.0,
                lr=0.01,
                seed=42,
            )
        # 深度记忆统计追踪
        self.dm_stats_history: list[dict] = []

        # ---- 物理沙盒 ---- #
        self.sandbox = PhysicsSandbox(num_objects=2, seed=42)
        self.img_pre = ImagePreprocessor(output_dim=dim, seed=42)

        # ---- 文本流 ---- #
        wiki_path = _SCRIPT_DIR / "experiments" / "simplewiki.txt"
        self.text_stream = TextStream(
            file_path=str(wiki_path), block_size=128, seed=42
        )
        self.text_encoder = TextEncoder(
            block_size=128, output_dim=dim, seed=42
        )

        # ---- 跨模态桥 ---- #
        self.bridge = MultimodalBridge(dim=dim, seed=42)
        self.event_detector = EventDetector(num_objects=2, seed=42)

        # ---- 经验缓冲 ---- #
        self.buffer = ExperienceBuffer(capacity=5000, seed=42)

        # ---- 可视化 ---- #
        self.collector = SnapshotCollector()
        self.milestone_detector = MilestoneDetector()
        self.streamer: WebSocketStreamer | None = None
        if streamer:
            self.streamer = WebSocketStreamer(port=8765)
            try:
                self.streamer.start()
            except Exception:
                self.streamer = None  # 降级：无 WebSocket 依赖

        # ---- 策略 ---- #
        self.physics_policy = CuriosityPolicy(
            n_actions=4, beta=1.0, seed=42
        )
        self.text_policy = CuriosityPolicy(
            n_actions=N_NAV_ACTIONS, beta=1.0, seed=43
        )

        # ---- 状态追踪 ---- #
        self.milestones: list[dict] = []
        self.step_counter = 0
        self.fe_history: list[float] = []
        self.stage_labels = ["physics", "text", "crossmodal", "qa"]
        self.current_stage = "physics"

        # ---- QA 组件（惰性初始化） ---- #
        self._inquiry_loop = None
        self._kg_builder = None

    def _mkdirs(self) -> None:
        """创建素材子目录。"""
        for sub in ("frames", "graphs", "latent", "text", "curves", "qa"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 工具：提取自由能
    # ------------------------------------------------------------------ #
    def _extract_fe(self, signal) -> float:
        """从 Signal 或模型状态中提取自由能。"""
        # 优先从 metadata 提取
        md = getattr(signal, "metadata", {}) or {}
        if "free_energy" in md:
            return float(md["free_energy"])
        # 回退到 active_inference 历史
        try:
            hist = self.model.active_inference.free_energy_history
            if len(hist) > 0:
                return float(hist[-1])
        except Exception:
            pass
        # 回退到 PCN 层误差（若启用了 PCN）
        try:
            pcn = md.get("pcn", {})
            errs = pcn.get("layer_errors", {})
            if errs:
                return float(sum(errs.values()))
        except Exception:
            pass
        return 0.0

    def _extract_pe(self, signal) -> float:
        """提取预测误差。"""
        md = getattr(signal, "metadata", {}) or {}
        if "prediction_error" in md:
            return float(md["prediction_error"])
        try:
            hist = self.model.active_inference.prediction_error_history
            if len(hist) > 0:
                return float(hist[-1])
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------ #
    # 深度记忆循环（S4 + PCN + Hopfield 第一阶段升级）
    # ------------------------------------------------------------------ #
    def _run_deep_memory(self, obs_vec: np.ndarray) -> dict:
        """运行一次深度记忆循环，返回统计字典。

        若未启用 deep_memory，返回空字典。统计会被累计到
        ``self.dm_stats_history`` 供报告生成使用。
        """
        if self.deep_memory is None:
            return {}
        try:
            dm_state = self.deep_memory.step(obs_vec)
            stats = {
                "step": self.step_counter,
                "stage": self.current_stage,
                "dm_free_energy": dm_state.free_energy,
                "dm_prediction_error": dm_state.prediction_error,
                "dm_hopfield_similarity": dm_state.hopfield_similarity,
                "dm_is_novel": int(dm_state.is_novel),
                "dm_s4_norm": dm_state.s4_norm,
                "dm_hopfield_size": dm_state.hopfield_size,
                "dm_pcn_l0_error": dm_state.pcn_layer_errors["l0"],
                "dm_pcn_l1_error": dm_state.pcn_layer_errors["l1"],
                "dm_pcn_l2_error": dm_state.pcn_layer_errors["l2"],
            }
            # 每 200 步记录一次快照（避免历史过长）
            if self.step_counter % 200 == 0:
                self.dm_stats_history.append(stats)
            return stats
        except Exception as exc:
            # 深度记忆失败不应阻断主流程
            return {"dm_error": str(exc)}

    def _save_deep_memory_curve(self) -> None:
        """保存深度记忆曲线（CSV）。"""
        if not self.dm_stats_history:
            return
        path = self.output_dir / "curves" / "deep_memory.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "step", "stage",
                "dm_free_energy", "dm_prediction_error",
                "dm_hopfield_similarity", "dm_is_novel",
                "dm_s4_norm", "dm_hopfield_size",
                "dm_pcn_l0_error", "dm_pcn_l1_error", "dm_pcn_l2_error",
            ])
            for s in self.dm_stats_history:
                w.writerow([
                    s["step"], s["stage"],
                    s["dm_free_energy"], s["dm_prediction_error"],
                    s["dm_hopfield_similarity"], s["dm_is_novel"],
                    s["dm_s4_norm"], s["dm_hopfield_size"],
                    s["dm_pcn_l0_error"], s["dm_pcn_l1_error"],
                    s["dm_pcn_l2_error"],
                ])

    # ------------------------------------------------------------------ #
    # 快照与素材保存
    # ------------------------------------------------------------------ #
    def _take_snapshot(self, signal, modality: str, action: int,
                       frame=None, text_block: str = "",
                       extra: dict | None = None) -> dict:
        """采集快照 → 推流 → 里程碑检测 → 返回 dict。"""
        fe = self._extract_fe(signal)
        pe = self._extract_pe(signal)
        self.fe_history.append(fe)

        snap = self.collector.collect(
            model=self.model,
            step=self.step_counter,
            modality=modality,
            frame=frame,
            text_block=text_block,
            action=action,
            free_energy=fe,
            prediction_error=pe,
            text_pos=self.text_stream.tell() if modality == "text" else -1,
            metadata=getattr(signal, "metadata", None),
        )
        snap_dict = snap.to_dict()
        if extra:
            snap_dict.update(extra)

        # 推流
        if self.streamer is not None:
            try:
                self.streamer.broadcast(snap_dict)
            except Exception:
                pass

        # 里程碑检测
        ms = self.milestone_detector.analyze(snap_dict)
        if ms is not None:
            ms_dict = {
                "step": ms.step,
                "type": ms.type,
                "title": ms.title,
                "description": ms.description,
                "stage": self.current_stage,
            }
            self.milestones.append(ms_dict)
            print(f"  ★ 里程碑: {ms.title}")

        return snap_dict

    def _save_frame(self, frame: np.ndarray, step: int, tag: str) -> None:
        """保存帧图像为 PNG。"""
        try:
            from PIL import Image
            img = Image.fromarray(frame.astype(np.uint8))
            path = self.output_dir / "frames" / f"{tag}_{step:06d}.png"
            img.save(str(path))
        except ImportError:
            np.save(
                str(self.output_dir / "frames" / f"{tag}_{step:06d}.npy"),
                frame,
            )

    def _save_text_snapshot(self, text: str, step: int) -> None:
        """保存文本快照。"""
        path = self.output_dir / "text" / f"snapshot_{step:06d}.txt"
        path.write_text(text, encoding="utf-8")

    def _save_latent(self, latent: np.ndarray, true_latent: np.ndarray,
                     step: int) -> None:
        """保存隐空间向量对（用于 3D 投影可视化）。"""
        np.savez(
            str(self.output_dir / "latent" / f"latent_{step:06d}.npz"),
            latent=latent,
            true_latent=true_latent,
        )

    def _save_kg_snapshot(self, step: int) -> None:
        """保存知识图谱快照。"""
        try:
            kg = self.collector.get_current_kg()
            path = self.output_dir / "graphs" / f"kg_{step:06d}.json"
            path.write_text(json.dumps(kg, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        except Exception:
            pass

    def _save_fe_curve(self) -> None:
        """保存自由能曲线数据（CSV）。"""
        path = self.output_dir / "curves" / "free_energy.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "free_energy", "stage"])
            # 标注每个 FE 值所属的阶段
            stage_offsets = self._stage_offsets
            for i, fe in enumerate(self.fe_history):
                stage = self._stage_for_step(i, stage_offsets)
                w.writerow([i, fe, stage])

    def _save_qa_log(self, q: str, a: str, step: int) -> None:
        """保存问答日志。"""
        path = self.output_dir / "qa" / f"qa_{step:06d}.txt"
        path.write_text(f"Q: {q}\n\nA: {a}\n", encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 阶段 1：物理直觉涌现
    # ------------------------------------------------------------------ #
    def run_physics_stage(self, steps: int = 5000) -> None:
        print(f"\n{'='*60}")
        print(f"阶段 1: 物理直觉涌现（沙盒探索，{steps} 步）")
        print(f"{'='*60}")
        self.current_stage = "physics"
        self._stage_start = self.step_counter

        obs = self.sandbox.reset(seed=42)
        self.event_detector.reset()

        for i in range(steps):
            obs_vec = self.img_pre.encode(obs)

            # 模型思考
            signal = self.model.think(obs_vec)
            fe = self._extract_fe(signal)
            pe = self._extract_pe(signal)

            # 深度记忆循环（S4+PCN+Hopfield）：用 obs_vec 驱动
            dm_stats = self._run_deep_memory(obs_vec)

            # 好奇心驱动动作选择
            action = self.physics_policy.select(pe)
            next_obs = self.sandbox.step(action)

            # 检测物理事件（用于跨模态对齐阶段）
            event = self.event_detector.detect(self.sandbox, action)

            # 经验记录
            next_vec = self.img_pre.encode(next_obs)
            self.buffer.add(obs_vec, action, next_vec, fe, step=self.step_counter)

            # 策略更新（奖励 = -FE）
            self.physics_policy.update(action, -fe)

            # 快照
            snap = self._take_snapshot(
                signal, modality="physics", action=action,
                frame=next_obs,
                extra={"event": event.label if event else "",
                       **dm_stats},
            )

            # 保存关键帧
            if i % 100 == 0:
                self._save_frame(next_obs, self.step_counter, "physics")

            # 进度报告
            if i % 500 == 0:
                print(f"  [step {self.step_counter}] FE={fe:.4f}, "
                      f"PE={pe:.4f}, β={self.physics_policy.beta:.4f}, "
                      f"event={event.label if event else 'none'}")

            obs = next_obs
            self.step_counter += 1

        # 阶段结束时保存 KG 和 FE 曲线
        self._save_kg_snapshot(self.step_counter)
        self._save_fe_curve()
        print(f"物理阶段完成（FE: {self.fe_history[-1]:.4f}）")

    # ------------------------------------------------------------------ #
    # 阶段 2：文本宇宙探索
    # ------------------------------------------------------------------ #
    def run_text_stage(self, steps: int = 5000) -> None:
        print(f"\n{'='*60}")
        print(f"阶段 2: 文本宇宙探索（百科阅读，{steps} 步）")
        print(f"{'='*60}")
        self.current_stage = "text"
        self._stage_start = self.step_counter

        text_block = self.text_stream.step()

        for i in range(steps):
            obs_vec = self.text_encoder.encode(text_block)

            # 模型思考
            signal = self.model.think(obs_vec)
            fe = self._extract_fe(signal)
            pe = self._extract_pe(signal)

            # 深度记忆循环
            dm_stats = self._run_deep_memory(obs_vec)

            # 文本导航动作
            action = self.text_policy.select(pe)
            next_block = self.text_stream.navigate(action)

            # 经验记录
            next_vec = self.text_encoder.encode(next_block)
            self.buffer.add(obs_vec, action, next_vec, fe, step=self.step_counter)

            # 策略更新
            self.text_policy.update(action, -fe)

            # 快照
            self._take_snapshot(
                signal, modality="text", action=action,
                text_block=next_block,
                extra=dm_stats,
            )

            # 保存文本快照
            if i % 200 == 0:
                self._save_text_snapshot(next_block, self.step_counter)

            # 进度报告
            if i % 500 == 0:
                pos = self.text_stream.tell()
                print(f"  [step {self.step_counter}] FE={fe:.4f}, "
                      f"PE={pe:.4f}, pos={pos}, "
                      f"block='{next_block[:40]}...'")

            text_block = next_block
            self.step_counter += 1

        self._save_kg_snapshot(self.step_counter)
        self._save_fe_curve()
        print(f"文本阶段完成（FE: {self.fe_history[-1]:.4f}）")

    # ------------------------------------------------------------------ #
    # 阶段 3：跨模态统一
    # ------------------------------------------------------------------ #
    def run_crossmodal_stage(self, steps: int = 3000) -> None:
        print(f"\n{'='*60}")
        print(f"阶段 3: 跨模态统一（对齐训练，{steps} 步）")
        print(f"{'='*60}")
        self.current_stage = "crossmodal"
        self._stage_start = self.step_counter

        # 重置沙盒以生成新的对齐数据
        obs = self.sandbox.reset(seed=99)
        self.event_detector.reset()

        for i in range(steps):
            # 生成物理帧 + 事件描述
            action = self.physics_policy.select(0.5)
            next_obs = self.sandbox.step(action)
            event = self.event_detector.detect(self.sandbox, action)
            text_desc = event.description if event else "objects move"

            # 随机模态掩码
            use_phys = np.random.random() < 0.5

            if use_phys:
                # 从物理预测文本
                h_phys = self.bridge.encode_physics(next_obs)
                h_text_true = self.bridge.encode_text(text_desc)
                obs_vec = h_phys
                h_pred = self.bridge.predict_text_from_physics(h_phys)
                cross_error = float(np.mean((h_text_true - h_pred) ** 2))
                # 更新桥
                self.bridge.update(h_phys, h_text_true)
            else:
                # 从文本预测物理
                h_text = self.bridge.encode_text(text_desc)
                h_phys_true = self.bridge.encode_physics(next_obs)
                obs_vec = h_text
                h_pred = self.bridge.predict_physics_from_text(h_text)
                cross_error = float(np.mean((h_phys_true - h_pred) ** 2))
                # 更新桥
                self.bridge.update(h_phys_true, h_text)

            # 模型思考（用融合后的隐表示）
            signal = self.model.think(obs_vec)
            fe = self._extract_fe(signal)

            # 深度记忆循环
            dm_stats = self._run_deep_memory(obs_vec)

            # 快照
            self._take_snapshot(
                signal, modality="crossmodal", action=action,
                frame=next_obs, text_block=text_desc,
                extra={"crossmodal_error": cross_error,
                       "masked_modality": "text" if use_phys else "physics",
                       **dm_stats},
            )

            # 保存关键帧和隐空间
            if i % 100 == 0:
                self._save_frame(next_obs, self.step_counter, "crossmodal")
                self._save_latent(obs_vec, h_pred, self.step_counter)

            # 进度报告
            if i % 500 == 0:
                print(f"  [step {self.step_counter}] FE={fe:.4f}, "
                      f"cross_err={cross_error:.6f}, "
                      f"mask={'text' if use_phys else 'phys'}, "
                      f"event={event.label if event else 'none'}")

            obs = next_obs
            self.step_counter += 1

        self._save_kg_snapshot(self.step_counter)
        self._save_fe_curve()
        print(f"跨模态阶段完成（cross_err: {cross_error:.6f}）")

    # ------------------------------------------------------------------ #
    # 阶段 4：自我提问与知识整合
    # ------------------------------------------------------------------ #
    def run_qa_stage(self, steps: int = 2000) -> None:
        print(f"\n{'='*60}")
        print(f"阶段 4: 自我提问与知识整合（百科问答，{steps} 步）")
        print(f"{'='*60}")
        self.current_stage = "qa"
        self._stage_start = self.step_counter

        # 惰性初始化 QA 组件
        self._init_qa_components()

        for i in range(steps):
            # 每 200 步触发主动提问
            if i > 0 and i % 200 == 0:
                print(f"  [step {self.step_counter}] 触发主动提问...")
                try:
                    summary = self._inquiry_loop.run(n_questions=2)
                    n_answered = summary.get("n_answered", 0)
                    n_total = summary.get("n_questions", 0)
                    print(f"    问答完成: {n_answered}/{n_total} 已回答")

                    # 保存问答日志
                    for qa in summary.get("qa", []):
                        q = qa.get("question", "")
                        a = qa.get("answer", "")
                        self._save_qa_log(q, a, self.step_counter)

                except Exception as exc:
                    print(f"    问答出错: {exc}")

            # 继续阅读文本
            text_block = self.text_stream.step()
            obs_vec = self.text_encoder.encode(text_block)

            signal = self.model.think(obs_vec)
            fe = self._extract_fe(signal)

            # 深度记忆循环
            dm_stats = self._run_deep_memory(obs_vec)

            # 更新知识图谱
            if self._kg_builder is not None and i % 50 == 0:
                try:
                    self._kg_builder.update_graph(
                        article_title=text_block[:20].strip(),
                        latent_state=signal.data,
                        free_energy=fe,
                        read_step=self.step_counter,
                    )
                except Exception:
                    pass

            self._take_snapshot(
                signal, modality="qa", action=0,
                text_block=text_block,
                extra=dm_stats,
            )

            if i % 500 == 0:
                print(f"  [step {self.step_counter}] FE={fe:.4f}")

            self.step_counter += 1

        self._save_kg_snapshot(self.step_counter)
        self._save_fe_curve()
        print(f"问答阶段完成")

    def _init_qa_components(self) -> None:
        """惰性初始化 InquiryLoop 及其依赖。"""
        if self._inquiry_loop is not None:
            return

        try:
            from text_decoder import TextDecoder
            from knowledge_graph_builder import KnowledgeGraphBuilder
            from active_inquiry import InquiryLoop

            decoder = TextDecoder(
                text_encoder=self.text_encoder, n=2, seed=42
            )
            self._kg_builder = KnowledgeGraphBuilder()

            # 尝试用 EncyclopediaReader 读取语料
            reader = None
            try:
                from encyclopedia_reader import EncyclopediaReader
                wiki_path = _SCRIPT_DIR / "experiments" / "simplewiki.txt"
                reader = EncyclopediaReader(
                    file_paths=str(wiki_path),
                    block_size=256,
                    seed=42,
                )
            except Exception as exc:
                print(f"  (EncyclopediaReader 不可用: {exc}; QA 将标记为无法回答)")

            self._inquiry_loop = InquiryLoop(
                model=self.model,
                text_encoder=self.text_encoder,
                decoder=decoder,
                kg_builder=self._kg_builder,
                reader=reader,
                max_read_blocks=8,
                seed=42,
            )
            print("  QA 组件初始化完成")
        except Exception as exc:
            print(f"  QA 组件初始化失败: {exc}")
            self._inquiry_loop = None

    # ------------------------------------------------------------------ #
    # 报告生成
    # ------------------------------------------------------------------ #
    @property
    def _stage_offsets(self) -> list[tuple[int, str]]:
        """计算各阶段在 fe_history 中的起始偏移。"""
        # 每阶段开始时记录的 self._stage_start
        # 简化：用 milestones 中的 stage 字段推断
        offsets = []
        prev_stage = None
        for i, fe in enumerate(self.fe_history):
            # 通过 _stage_marks（在 run_*_stage 中记录）
            pass
        return offsets

    def _stage_for_step(self, step: int,
                        offsets: list) -> str:
        """推断给定全局步属于哪个阶段。"""
        # 简化实现：按步数范围划分
        if not hasattr(self, "_stage_marks"):
            return "unknown"
        marks = self._stage_marks  # [(start_step, label), ...]
        result = "unknown"
        for start, label in marks:
            if step >= start:
                result = label
        return result

    def generate_report(self) -> None:
        """生成演示摘要和素材索引。"""
        # 收集素材列表
        frames_dir = self.output_dir / "frames"
        text_dir = self.output_dir / "text"
        latent_dir = self.output_dir / "latent"
        graphs_dir = self.output_dir / "graphs"
        qa_dir = self.output_dir / "qa"
        curves_dir = self.output_dir / "curves"

        frame_files = sorted(frames_dir.glob("*.png")) if frames_dir.exists() else []
        # 如果没有 PNG（无 PIL），统计 .npy
        if not frame_files:
            frame_files = sorted(frames_dir.glob("*.npy"))

        report = {
            "title": "ZeroDataModel 认知起源演示报告",
            "timestamp": datetime.now().isoformat(),
            "total_steps": self.step_counter,
            "model_dim": self.dim,
            "stages": {
                "physics": {"steps": "见自由能曲线", "status": "完成"},
                "text": {"steps": "见自由能曲线", "status": "完成"},
                "crossmodal": {"steps": "见自由能曲线", "status": "完成"},
                "qa": {"steps": "见自由能曲线", "status": "完成"},
            },
            "milestones": self.milestones,
            "milestone_count": len(self.milestones),
            "final_free_energy": self.fe_history[-1] if self.fe_history else None,
            "fe_history_length": len(self.fe_history),
            "deep_memory": self._report_deep_memory(),
            "material_index": {
                "frames": {
                    "count": len(frame_files),
                    "dir": str(frames_dir),
                    "samples": [f.name for f in frame_files[:10]],
                },
                "text_snapshots": {
                    "count": len(list(text_dir.glob("*.txt"))) if text_dir.exists() else 0,
                    "dir": str(text_dir),
                },
                "latent_snapshots": {
                    "count": len(list(latent_dir.glob("*.npz"))) if latent_dir.exists() else 0,
                    "dir": str(latent_dir),
                },
                "kg_snapshots": {
                    "count": len(list(graphs_dir.glob("*.json"))) if graphs_dir.exists() else 0,
                    "dir": str(graphs_dir),
                },
                "qa_logs": {
                    "count": len(list(qa_dir.glob("*.txt"))) if qa_dir.exists() else 0,
                    "dir": str(qa_dir),
                },
                "curves": {
                    "free_energy_csv": str(curves_dir / "free_energy.csv") if curves_dir.exists() else None,
                    "deep_memory_csv": str(curves_dir / "deep_memory.csv") if (curves_dir / "deep_memory.csv").exists() else None,
                },
            },
        }

        report_path = self.output_dir / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 里程碑独立文件
        ms_path = self.output_dir / "milestones.json"
        ms_path.write_text(
            json.dumps(self.milestones, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\n{'='*60}")
        print(f"演示报告已生成: {report_path}")
        print(f"里程碑记录: {ms_path}")
        print(f"总步数: {self.step_counter}")
        print(f"里程碑数: {len(self.milestones)}")
        if self.fe_history:
            print(f"最终自由能: {self.fe_history[-1]:.4f}")
        print(f"素材目录: {self.output_dir}")
        print(f"  帧图像: {report['material_index']['frames']['count']}")
        print(f"  文本快照: {report['material_index']['text_snapshots']['count']}")
        print(f"  隐空间快照: {report['material_index']['latent_snapshots']['count']}")
        print(f"  KG 快照: {report['material_index']['kg_snapshots']['count']}")
        print(f"  问答日志: {report['material_index']['qa_logs']['count']}")
        if self.deep_memory is not None:
            dm = report["deep_memory"]
            print(f"  深度记忆: S4 sr={dm['components']['s4']['spectral_radius']:.4f}, "
                  f"Hopfield={dm['components']['hopfield']['size']}/{dm['components']['hopfield']['capacity']}, "
                  f"novel={dm['stats']['novel_count']}")
        print(f"{'='*60}")

    def _report_deep_memory(self) -> dict:
        """生成深度记忆模块的统计报告。"""
        if self.deep_memory is None:
            return {"enabled": False}
        snap = self.deep_memory.get_snapshot()
        return {
            "enabled": True,
            "components": {
                "s4": snap["s4"],
                "pcn": snap["pcn"],
                "hopfield": snap["hopfield"],
            },
            "stats": snap["stats"],
            "snapshots_collected": len(self.dm_stats_history),
        }

    # ------------------------------------------------------------------ #
    # 一键运行
    # ------------------------------------------------------------------ #
    def run_all(self, physics_steps: int = 5000, text_steps: int = 5000,
                crossmodal_steps: int = 3000, qa_steps: int = 2000) -> None:
        """运行全部四个阶段。"""
        self._stage_marks: list[tuple[int, str]] = []

        t0 = time.time()
        self._stage_marks.append((0, "physics"))
        self.run_physics_stage(physics_steps)

        self._stage_marks.append((self.step_counter, "text"))
        self.run_text_stage(text_steps)

        self._stage_marks.append((self.step_counter, "crossmodal"))
        self.run_crossmodal_stage(crossmodal_steps)

        self._stage_marks.append((self.step_counter, "qa"))
        self.run_qa_stage(qa_steps)

        self._save_fe_curve()
        self._save_deep_memory_curve()
        self.generate_report()

        elapsed = time.time() - t0
        print(f"\n总运行时间: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print("=== 认知起源演示全部完成 ===")

    def cleanup(self) -> None:
        """清理资源。"""
        if self.streamer is not None:
            try:
                self.streamer.stop()
            except Exception:
                pass


# ------------------------------------------------------------------ #
# CLI 入口
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="ZeroDataModel 认知起源完整演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demo_cognitive_origin.py                      # 全量 ~15000 步
  python demo_cognitive_origin.py --steps 100          # 快速烟雾测试
  python demo_cognitive_origin.py --physics 1000       # 仅物理阶段
  python demo_cognitive_origin.py --no-streamer         # 禁用 WebSocket
        """,
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="快速模式：每阶段运行 N 步（覆盖各阶段独立设置）",
    )
    parser.add_argument("--physics", type=int, default=5000,
                        help="物理阶段步数 (默认 5000)")
    parser.add_argument("--text", type=int, default=5000,
                        help="文本阶段步数 (默认 5000)")
    parser.add_argument("--crossmodal", type=int, default=3000,
                        help="跨模态阶段步数 (默认 3000)")
    parser.add_argument("--qa", type=int, default=2000,
                        help="问答阶段步数 (默认 2000)")
    parser.add_argument("--output", type=str, default="demo_output",
                        help="素材输出目录 (默认 demo_output)")
    parser.add_argument("--dim", type=int, default=64,
                        help="模型隐空间维度 (默认 64)")
    parser.add_argument("--no-streamer", action="store_true",
                        help="禁用 WebSocket 实时推流")
    parser.add_argument("--no-deep-memory", action="store_true",
                        help="禁用 S4+PCN+Hopfield 深度认知记忆（第一阶段升级）")
    args = parser.parse_args()

    # 快速模式
    if args.steps is not None:
        physics_steps = text_steps = args.steps
        crossmodal_steps = max(args.steps // 2, 50)
        qa_steps = max(args.steps // 3, 20)
    else:
        physics_steps = args.physics
        text_steps = args.text
        crossmodal_steps = args.crossmodal
        qa_steps = args.qa

    demo = CognitiveOriginDemo(
        output_dir=args.output,
        dim=args.dim,
        streamer=not args.no_streamer,
        use_deep_memory=not args.no_deep_memory,
    )
    try:
        demo.run_all(
            physics_steps=physics_steps,
            text_steps=text_steps,
            crossmodal_steps=crossmodal_steps,
            qa_steps=qa_steps,
        )
    except KeyboardInterrupt:
        print("\n\n用户中断，保存已有素材...")
        demo._save_fe_curve()
        demo.generate_report()
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()
