#!/usr/bin/env python3
"""素材索引生成器 + 演示视频剪辑脚本建议。

扫描 ``demo_cognitive_origin.py`` 产生的素材目录，生成：

  1. ``manifest.json``         —— 全部素材的结构化清单（按阶段/类型分组）
  2. ``video_segments.json``   —— 视频片段建议（每段含起止步、推荐时长、字幕）
  3. ``assemble_video.sh``     —— ffmpeg 一键组装脚本（占位；用户按需调整）

用法::

    python build_material_index.py --input demo_output
    python build_material_index.py --input demo_output --fps 30 --out video
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------------ #
# 素材类型
# ------------------------------------------------------------------ #
FRAME_TAGS = ("physics", "crossmodal")  # 文件名前缀 → 阶段
STAGES = ("physics", "text", "crossmodal", "qa")


# ------------------------------------------------------------------ #
# 工具
# ------------------------------------------------------------------ #
def _step_from_name(name: str) -> Optional[int]:
    """从文件名 ``tag_000123.png`` 提取步数。"""
    m = re.search(r"_(\d{6,})\.", name)
    return int(m.group(1)) if m else None


def _tag_from_name(name: str) -> str:
    """提取阶段标签前缀。"""
    for tag in STAGES:
        if name.startswith(tag + "_"):
            return tag
    return "unknown"


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# ------------------------------------------------------------------ #
# Manifest 构建
# ------------------------------------------------------------------ #
@dataclass
class Manifest:
    input_dir: str
    generated_at: str
    total_steps: int = 0
    stages: dict = field(default_factory=dict)
    materials: dict = field(default_factory=dict)
    milestones: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "input_dir": self.input_dir,
            "generated_at": self.generated_at,
            "total_steps": self.total_steps,
            "stages": self.stages,
            "materials": self.materials,
            "milestones": self.milestones,
            "notes": self.notes,
        }


def build_manifest(input_dir: Path) -> Manifest:
    """扫描素材目录，构建 manifest。"""
    from datetime import datetime

    m = Manifest(
        input_dir=str(input_dir),
        generated_at=datetime.now().isoformat(),
    )

    # report.json
    report_path = input_dir / "report.json"
    if report_path.exists():
        try:
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            m.total_steps = rep.get("total_steps", 0)
            m.milestones = rep.get("milestones", [])
        except Exception as exc:
            m.notes.append(f"读取 report.json 失败: {exc}")

    # 阶段边界（从 milestones 推断，或从 _stage_marks 文件）
    stage_bounds = _infer_stage_bounds(input_dir, m.total_steps)
    m.stages = stage_bounds

    # 帧图像
    frames_dir = input_dir / "frames"
    frame_files = sorted(
        list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.npy"))
    ) if frames_dir.exists() else []
    m.materials["frames"] = _index_files(frame_files, with_tag=True)

    # 文本快照
    text_dir = input_dir / "text"
    text_files = sorted(text_dir.glob("*.txt")) if text_dir.exists() else []
    m.materials["text_snapshots"] = _index_files(text_files, with_tag=True)

    # 隐空间快照
    latent_dir = input_dir / "latent"
    latent_files = sorted(latent_dir.glob("*.npz")) if latent_dir.exists() else []
    m.materials["latent_snapshots"] = _index_files(latent_files, with_tag=True)

    # 知识图谱
    graphs_dir = input_dir / "graphs"
    kg_files = sorted(graphs_dir.glob("*.json")) if graphs_dir.exists() else []
    m.materials["kg_snapshots"] = _index_files(kg_files, with_tag=True)

    # 问答日志
    qa_dir = input_dir / "qa"
    qa_files = sorted(qa_dir.glob("*.txt")) if qa_dir.exists() else []
    m.materials["qa_logs"] = _index_files(qa_files, with_tag=True)

    # 自由能曲线
    curves_dir = input_dir / "curves"
    fe_csv = curves_dir / "free_energy.csv" if curves_dir.exists() else None
    if fe_csv and fe_csv.exists():
        m.materials["free_energy_curve"] = {
            "path": str(fe_csv),
            "rows": _count_csv_rows(fe_csv),
        }

    return m


def _infer_stage_bounds(input_dir: Path, total_steps: int) -> dict:
    """从 milestones.json 推断每个阶段的步数边界。"""
    ms_path = input_dir / "milestones.json"
    bounds: dict[str, dict] = {}
    if not ms_path.exists():
        return bounds

    try:
        milestones = json.loads(ms_path.read_text(encoding="utf-8"))
    except Exception:
        return bounds

    by_stage: dict[str, int] = {}
    for ms in milestones:
        stage = ms.get("stage", "unknown")
        if stage not in by_stage:
            by_stage[stage] = ms.get("step", 0)

    # 计算每阶段 [start, end)
    ordered = sorted(by_stage.items(), key=lambda kv: kv[1])
    for i, (stage, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else total_steps
        bounds[stage] = {"start_step": start, "end_step": end,
                         "duration_steps": end - start}
    return bounds


def _index_files(files: list[Path], with_tag: bool = False) -> dict:
    """构建某类素材的索引。"""
    if not files:
        return {"count": 0, "files": []}
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        step = _step_from_name(f.name)
        tag = _tag_from_name(f.name) if with_tag else "all"
        by_tag[tag].append({
            "name": f.name,
            "path": str(f),
            "step": step,
            "size_bytes": f.stat().st_size,
        })
    return {
        "count": len(files),
        "by_tag": dict(by_tag),
    }


def _count_csv_rows(path: Path) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in csv.reader(f)) - 1  # 去表头
    except Exception:
        return 0


# ------------------------------------------------------------------ #
# 视频片段建议
# ------------------------------------------------------------------ #
@dataclass
class Segment:
    stage: str
    start_step: int
    end_step: int
    recommended_seconds: float
    caption: str
    suggested_clips: list[dict] = field(default_factory=list)


def build_video_segments(manifest: Manifest, fps: int = 30,
                        target_total_seconds: int = 180) -> list[dict]:
    """根据阶段边界 + 里程碑，构建视频片段建议。"""
    segments: list[Segment] = []
    if not manifest.stages:
        return []

    stage_titles = {
        "physics": "阶段 1：物理直觉涌现（沙盒探索）",
        "text": "阶段 2：文本宇宙探索（百科阅读）",
        "crossmodal": "阶段 3：跨模态统一（对齐训练）",
        "qa": "阶段 4：自我提问与知识整合",
    }

    # 按阶段持续时间分配视频时长
    total_steps = sum(s["duration_steps"] for s in manifest.stages.values()) or 1
    for stage, bounds in manifest.stages.items():
        dur_steps = bounds["duration_steps"]
        seconds = max(20.0, target_total_seconds * dur_steps / total_steps)
        # 选取该阶段的关键帧作为片段
        frames = manifest.materials.get("frames", {}).get("by_tag", {}).get(stage, [])
        # 最多取 5 张作为关键帧
        key_frames = sorted(frames, key=lambda x: x["step"] or 0)[:5]
        seg = Segment(
            stage=stage,
            start_step=bounds["start_step"],
            end_step=bounds["end_step"],
            recommended_seconds=round(seconds, 1),
            caption=stage_titles.get(stage, stage),
            suggested_clips=[
                {"path": kf["path"], "step": kf["step"],
                 "hold_seconds": round(seconds / max(len(key_frames), 1), 2)}
                for kf in key_frames
            ],
        )
        segments.append(seg)

    # 在里程碑处插入标题卡
    milestone_cards = []
    for ms in manifest.milestones:
        milestone_cards.append({
            "step": ms.get("step"),
            "title": ms.get("title", ""),
            "description": ms.get("description", ""),
            "stage": ms.get("stage", ""),
            "hold_seconds": 2.0,
            "type": "title_card",
        })

    return {
        "fps": fps,
        "target_total_seconds": target_total_seconds,
        "segments": [s.__dict__ for s in segments],
        "milestone_title_cards": milestone_cards,
        "notes": [
            "建议使用 ffmpeg 的 concat filter 或 concat demuxer 拼接片段。",
            "每个 PNG 帧通过 -loop 1 -t <hold_seconds> 转成短视频片段。",
            "milestone_title_cards 可用 drawtext 滤镜叠加字幕。",
            "自由能曲线（free_energy.csv）建议用 matplotlib 单独渲染成曲线动画后插入。",
        ],
    }


# ------------------------------------------------------------------ #
# ffmpeg 组装脚本
# ------------------------------------------------------------------ #
def write_assemble_script(manifest: Manifest, segments_data: dict,
                          out_path: Path) -> None:
    """写出 ``assemble_video.sh``。"""
    lines: list[str] = [
        "#!/usr/bin/env bash",
        "# 由 build_material_index.py 自动生成",
        "# 演示视频组装脚本（基于 ffmpeg）",
        "",
        f"OUTPUT=\"${{1:-cognitive_origin_demo.mp4}}\"",
        f"FPS={segments_data['fps']}",
        "",
        "# 检查 ffmpeg",
        "if ! command -v ffmpeg >/dev/null 2>&1; then",
        "    echo \"错误：未找到 ffmpeg，请先安装\" >&2",
        "    exit 1",
        "fi",
        "",
        "# 构建 concat 列表文件",
        "LIST=$(mktemp)",
        "trap 'rm -f $LIST' EXIT",
        "",
    ]

    # 为每个 segment 的关键帧生成短片段
    for seg in segments_data["segments"]:
        lines.append(f"# === {seg['caption']} ===")
        for clip in seg["suggested_clips"]:
            p = clip["path"]
            h = clip["hold_seconds"]
            if Path(p).exists():
                lines.append(
                    f"echo \"file '{p}'\" >> \"$LIST\""
                )
                # 用 ffmpeg 把图片转成短片段（这里仅示意，实际需先转成 mp4）
                # 为简化，我们用 concat demuxer 直接拼接图片
                lines.append(
                    f"# 帧 {clip['step']} 显示 {h}s"
                )
        lines.append("")

    # milestone title cards（纯文字提示，建议手动渲染）
    if segments_data["milestone_title_cards"]:
        lines.append("# === 里程碑标题卡（建议手动用 drawtext 渲染） ===")
        for card in segments_data["milestone_title_cards"]:
            lines.append(
                f"# step={card['step']}: {card['title']} - {card['description'][:60]}"
            )
        lines.append("")

    # 自由能曲线（建议先用 matplotlib 渲染）
    fe_path = manifest.materials.get("free_energy_curve", {}).get("path")
    if fe_path:
        lines.append("# === 自由能曲线（先用 render_fe_curve.py 渲染为 PNG） ===")
        lines.append(f"# 源数据: {fe_path}")
        lines.append("# python render_fe_curve.py --input \"{}\" "
                     "--output fe_curve.png".format(fe_path))
        lines.append("")

    lines.extend([
        "# === 最终拼接 ===",
        "# 注意：图片直接拼接需要每张 -loop 1 -t <hold_seconds> 转成 mp4",
        "# 完整命令（示例，需按实际片段调整）：",
        "#",
        "# ffmpeg -framerate $FPS -loop 1 -t 3 -i frame1.png \\",
        "#        -framerate $FPS -loop 1 -t 3 -i frame2.png \\",
        "#        -filter_complex concat=n=2:v=1:a=0 out.mp4",
        "",
        "echo \"脚本已生成，请根据注释手动调整 ffmpeg 命令\"",
        "echo \"最终输出: $OUTPUT\"",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.chmod(0o755)


# ------------------------------------------------------------------ #
# 主入口
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="生成认知起源演示的素材索引和视频剪辑建议",
    )
    parser.add_argument("--input", type=str, default="demo_output",
                        help="demo_cognitive_origin.py 的输出目录")
    parser.add_argument("--fps", type=int, default=30,
                        help="输出视频帧率（默认 30）")
    parser.add_argument("--target-seconds", type=int, default=180,
                        help="目标视频总时长（秒，默认 180）")
    parser.add_argument("--out", type=str, default="video_assets",
                        help="索引/脚本输出目录")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"错误：输入目录不存在: {input_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"扫描素材目录: {input_dir}")
    manifest = build_manifest(input_dir)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"素材清单已生成: {manifest_path}")
    print(f"  帧图像: {manifest.materials['frames']['count']}")
    print(f"  文本快照: {manifest.materials['text_snapshots']['count']}")
    print(f"  隐空间快照: {manifest.materials['latent_snapshots']['count']}")
    print(f"  KG 快照: {manifest.materials['kg_snapshots']['count']}")
    print(f"  问答日志: {manifest.materials['qa_logs']['count']}")
    print(f"  里程碑: {len(manifest.milestones)}")

    segments = build_video_segments(
        manifest, fps=args.fps, target_total_seconds=args.target_seconds,
    )
    segments_path = out_dir / "video_segments.json"
    segments_path.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"视频片段建议: {segments_path}")

    assemble_path = out_dir / "assemble_video.sh"
    write_assemble_script(manifest, segments, assemble_path)
    print(f"组装脚本: {assemble_path}")

    if not _ffmpeg_available():
        print("\n注意：未检测到 ffmpeg，组装脚本仅供参考。"
              "安装 ffmpeg 后可直接运行。")

    print("\n=== 素材索引生成完成 ===")


if __name__ == "__main__":
    main()
