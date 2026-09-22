#!/usr/bin/env python3
"""认知起源演示视频自动生成器。

基于 ``demo_cognitive_origin.py`` 产生的素材，自动构建一段完整的演示视频：

  1. 标题卡（项目名 + 时间戳）
  2. 阶段 1 物理直觉涌现（帧序列 → mp4，叠加字幕）
  3. 阶段 2 文本宇宙探索（文本快照 → 图片帧 → mp4）
  4. 阶段 3 跨模态统一（帧 + 隐空间投影图）
  5. 阶段 4 自我提问（问答日志 → 字幕卡）
  6. 自由能曲线动画（matplotlib 渲染）
  7. 里程碑汇总（drawtext 叠加）
  8. 结尾卡

依赖：
  - ffmpeg (>= 4.0)
  - Python: numpy, pillow, matplotlib (可选，用于曲线渲染)

用法::

    python render_demo_video.py --input demo_output --out video_assets
    python render_demo_video.py --input demo_output --out video_assets --fps 24 --no-curves
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ------------------------------------------------------------------ #
# 工具函数
# ------------------------------------------------------------------ #
def _run(cmd: list[str], **kw) -> int:
    """运行外部命令，打印命令到 stderr。"""
    print("  $ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, check=True, **kw).returncode


def _ffmpeg_ok() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        return True
    except Exception:
        return False


def _step_from_name(name: str) -> int:
    m = re.search(r"_(\d{6,})\.", name)
    return int(m.group(1)) if m else 0


def _font(size: int = 28) -> ImageFont.FreeTypeFont:
    """寻找系统中可用字体。"""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ------------------------------------------------------------------ #
# 标题卡渲染
# ------------------------------------------------------------------ #
def render_title_card(path: Path, title: str, subtitle: str = "",
                     width: int = 1280, height: int = 720,
                     bg: str = "#0a0a1a", fg: str = "#ffffff") -> None:
    """渲染一张标题卡 PNG。"""
    img = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(img)
    title_font = _font(56)
    sub_font = _font(28)
    # 居中
    tw, th = d.textsize(title, font=title_font) if hasattr(d, "textsize") else \
             (title_font.getlength(title), title_font.size)
    d.text(((width - tw) / 2, height / 2 - 60), title,
           font=title_font, fill=fg)
    if subtitle:
        sw = sub_font.getlength(subtitle) if hasattr(sub_font, "getlength") else len(subtitle) * 14
        d.text(((width - sw) / 2, height / 2 + 20), subtitle,
               font=sub_font, fill="#aaaacc")
    img.save(str(path))


def render_text_card(path: Path, lines: list[str],
                     width: int = 1280, height: int = 720,
                     bg: str = "#101020", fg: str = "#e0e0e0") -> None:
    """渲染多行文本卡片。"""
    img = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(img)
    font = _font(24)
    y = 40
    for line in lines:
        d.text((40, y), line[:120], font=font, fill=fg)
        y += 32
    img.save(str(path))


# ------------------------------------------------------------------ #
# 自由能曲线渲染（matplotlib 可选）
# ------------------------------------------------------------------ #
def render_fe_curve(csv_path: Path, out_png: Path,
                    width: int = 1280, height: int = 720) -> bool:
    """用 matplotlib 渲染自由能曲线为 PNG。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib 不可用，跳过自由能曲线渲染")
        return False

    steps, fes, stages = [], [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            fes.append(float(row["free_energy"]))
            stages.append(row.get("stage", ""))

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    stage_colors = {
        "physics": "#4a90e2",
        "text": "#7ed957",
        "crossmodal": "#ff9f43",
        "qa": "#ee5a6f",
    }
    # 按阶段分段着色
    prev_stage = None
    seg_start = 0
    for i, st in enumerate(stages + [None]):
        if st != prev_stage and prev_stage is not None:
            color = stage_colors.get(prev_stage, "#888888")
            ax.plot(steps[seg_start:i], fes[seg_start:i],
                    color=color, linewidth=2, label=prev_stage)
            seg_start = i
        prev_stage = st

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Free Energy", fontsize=12)
    ax.set_title("Free Energy Curve — Cognitive Origin Demo", fontsize=14)
    ax.grid(True, alpha=0.3)
    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels)
            if not (l in seen or seen.add(l))]
    ax.legend(*zip(*uniq))
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=100)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ #
# 隐空间投影渲染（PCA → 2D 散点）
# ------------------------------------------------------------------ #
def render_latent_scatter(latent_dir: Path, out_png: Path,
                          width: int = 1280, height: int = 720) -> bool:
    """将所有 latent_*.npz 加载并做 PCA → 2D 散点图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        # 退化为 numpy 自实现 PCA
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return False
        PCA = None

    files = sorted(latent_dir.glob("latent_*.npz"))
    if not files:
        return False
    mat = []
    for f in files:
        try:
            d = np.load(str(f))
            mat.append(d["latent"].flatten())
        except Exception:
            continue
    if len(mat) < 2:
        return False
    X = np.stack(mat, axis=0)
    # 中心化
    X = X - X.mean(axis=0, keepdims=True)
    if PCA is not None:
        emb = PCA(n_components=2).fit_transform(X)
    else:
        # SVD 退化为 2D
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        emb = X @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.scatter(emb[:, 0], emb[:, 1], c=range(len(emb)),
               cmap="viridis", s=80, alpha=0.8)
    ax.set_title("Latent Space Trajectory (PCA 2D)", fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=100)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ #
# 视频片段生成（PNG 序列 → mp4）
# ------------------------------------------------------------------ #
def png_sequence_to_mp4(png_files: list[Path], out_mp4: Path,
                        fps: int = 10, duration_each: float = 0.5,
                        width: int = 1280, height: int = 720) -> bool:
    """将一组 PNG 合成 mp4（每张停留 duration_each 秒）。"""
    if not png_files:
        return False
    # 先把所有图归一化到同尺寸（ffmpeg concat 要求统一）
    tmp_dir = out_mp4.parent / f".tmp_norm_{out_mp4.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    norm_files: list[Path] = []
    for i, p in enumerate(png_files):
        try:
            img = Image.open(p).convert("RGB")
            img = img.resize((width, height), Image.LANCZOS)
            np_path = tmp_dir / f"frame_{i:06d}.png"
            img.save(str(np_path))
            norm_files.append(np_path)
        except Exception as exc:
            print(f"  [warn] 跳过 {p}: {exc}")

    if not norm_files:
        return False

    # concat 列表
    list_path = tmp_dir / "concat.txt"
    list_path.write_text(
        "\n".join(f"file '{f.absolute()}'" for f in norm_files) + "\n",
        encoding="utf-8",
    )

    # ffmpeg concat demuxer（每张停留 duration_each 秒）
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-vf", f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    # 注：concat demuxer 不支持每帧时长；改用 image2 + duration
    # 重写为 image2 模式
    list_path.write_text(
        "\n".join(
            f"file '{f.absolute()}'\nduration {duration_each}"
            for f in norm_files
        ) + f"\nfile '{norm_files[-1].absolute()}'\n",
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-vf", f"fps={fps},scale={width}:{height}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    try:
        _run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        print(f"  [warn] ffmpeg 失败: {exc.stderr.decode()[:300]}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


def single_png_to_mp4(png_path: Path, out_mp4: Path,
                      duration: float = 3.0, fps: int = 10,
                      width: int = 1280, height: int = 720) -> bool:
    """单张 PNG → mp4（持续 duration 秒）。"""
    if not png_path.exists():
        return False
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(fps),
        "-i", str(png_path),
        "-t", str(duration),
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(out_mp4),
    ]
    try:
        _run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  [warn] ffmpeg 失败: {exc.stderr.decode()[:300]}")
        return False


# ------------------------------------------------------------------ #
# 字幕叠加（drawtext）
# ------------------------------------------------------------------ #
def overlay_text(input_mp4: Path, output_mp4: Path,
                 text: str, fontsize: int = 24,
                 y_offset: int = 40) -> bool:
    """用 drawtext 滤镜在底部叠加字幕。"""
    # 转义特殊字符
    safe = (text.replace("\\", "\\\\")
                .replace(":", "\\:")
                .replace("'", "\\'")
                .replace("%", "\\%"))
    vf = (f"drawtext=text='{safe}':"
          f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
          f"fontcolor=white:fontsize={fontsize}:"
          f"box=1:boxcolor=black@0.6:boxborderw=8:"
          f"x=(w-text_w)/2:y=h-text_h-{y_offset}")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_mp4),
        "-vf", vf,
        "-c:a", "copy", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output_mp4),
    ]
    try:
        _run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        # 退化为无字幕
        shutil.copy(str(input_mp4), str(output_mp4))
        return False


# ------------------------------------------------------------------ #
# 视频拼接
# ------------------------------------------------------------------ #
def concat_mp4s(mp4_files: list[Path], out_mp4: Path) -> bool:
    """拼接多个 mp4（要求同尺寸/编码）。"""
    if not mp4_files:
        return False
    if len(mp4_files) == 1:
        shutil.copy(str(mp4_files[0]), str(out_mp4))
        return True
    list_path = out_mp4.parent / f".concat_{out_mp4.stem}.txt"
    list_path.write_text(
        "\n".join(f"file '{f.absolute()}'" for f in mp4_files) + "\n",
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(out_mp4),
    ]
    try:
        _run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        list_path.unlink(missing_ok=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  [warn] concat 失败，尝试 re-encode: {exc.stderr.decode()[:200]}")
        # 退化为 re-encode
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_mp4),
        ]
        try:
            _run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            list_path.unlink(missing_ok=True)
            return True
        except subprocess.CalledProcessError:
            list_path.unlink(missing_ok=True)
            return False


# ------------------------------------------------------------------ #
# 主流程
# ------------------------------------------------------------------ #
class DemoVideoBuilder:
    def __init__(self, input_dir: Path, out_dir: Path,
                 fps: int = 10, width: int = 1280, height: int = 720,
                 draw_curves: bool = True):
        self.input_dir = input_dir
        self.out_dir = out_dir
        self.tmp_dir = out_dir / ".tmp"
        self.fps = fps
        self.width = width
        self.height = height
        self.draw_curves = draw_curves
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # 加载 manifest
        report_path = input_dir / "report.json"
        self.report = {}
        if report_path.exists():
            self.report = json.loads(report_path.read_text(encoding="utf-8"))

    def build(self) -> Path:
        print(f"\n=== 开始构建演示视频 ===")
        if not _ffmpeg_ok():
            print("错误：ffmpeg 不可用", file=sys.stderr)
            sys.exit(1)

        segments: list[Path] = []

        # 1. 标题卡
        title_seg = self._build_title_segment()
        if title_seg:
            segments.append(title_seg)
            print("  [OK] 标题卡")

        # 2. 物理阶段
        seg = self._build_physics_segment()
        if seg:
            segments.append(seg)
            print("  [OK] 物理阶段片段")

        # 3. 文本阶段
        seg = self._build_text_segment()
        if seg:
            segments.append(seg)
            print("  [OK] 文本阶段片段")

        # 4. 跨模态阶段
        seg = self._build_crossmodal_segment()
        if seg:
            segments.append(seg)
            print("  [OK] 跨模态阶段片段")

        # 5. QA 阶段
        seg = self._build_qa_segment()
        if seg:
            segments.append(seg)
            print("  [OK] QA 阶段片段")

        # 6. 自由能曲线
        if self.draw_curves:
            seg = self._build_fe_curve_segment()
            if seg:
                segments.append(seg)
                print("  [OK] 自由能曲线片段")

        # 7. 隐空间投影
        seg = self._build_latent_segment()
        if seg:
            segments.append(seg)
            print("  [OK] 隐空间投影片段")

        # 8. 里程碑汇总
        seg = self._build_milestones_segment()
        if seg:
            segments.append(seg)
            print("  [OK] 里程碑片段")

        # 9. 结尾卡
        seg = self._build_end_card_segment()
        if seg:
            segments.append(seg)
            print("  [OK] 结尾卡")

        # 拼接
        print(f"\n拼接 {len(segments)} 个片段...")
        final_path = self.out_dir / "cognitive_origin_demo.mp4"
        if not segments:
            print("错误：无可用片段", file=sys.stderr)
            sys.exit(1)
        if not concat_mp4s(segments, final_path):
            print("错误：拼接失败", file=sys.stderr)
            sys.exit(1)

        # 清理临时文件
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        for seg in segments:
            seg.unlink(missing_ok=True)

        size_mb = final_path.stat().st_size / 1024 / 1024
        print(f"\n=== 视频已生成 ===")
        print(f"  路径: {final_path}")
        print(f"  大小: {size_mb:.2f} MB")
        return final_path

    # ---------------------------------------------------------------- #
    def _build_title_segment(self) -> Optional[Path]:
        title = "ZeroDataModel"
        subtitle = "认知起源演示 · " + self.report.get(
            "timestamp", "unknown"
        ).split("T")[0]
        png = self.tmp_dir / "title.png"
        render_title_card(png, title, subtitle,
                          width=self.width, height=self.height)
        mp4 = self.tmp_dir / "title.mp4"
        if single_png_to_mp4(png, mp4, duration=3.0, fps=self.fps,
                             width=self.width, height=self.height):
            return self._with_caption(mp4, "ZeroDataModel — Cognitive Origin")
        return None

    def _build_physics_segment(self) -> Optional[Path]:
        frames = sorted((self.input_dir / "frames").glob("physics_*.png"))
        if not frames:
            return None
        mp4 = self.tmp_dir / "physics.mp4"
        ok = png_sequence_to_mp4(
            frames, mp4, fps=self.fps, duration_each=0.5,
            width=self.width, height=self.height,
        )
        if not ok:
            return None
        return self._with_caption(mp4, "阶段 1：物理直觉涌现（沙盒探索）")

    def _build_text_segment(self) -> Optional[Path]:
        # 文本快照转 PNG
        text_files = sorted((self.input_dir / "text").glob("snapshot_*.txt"))
        if not text_files:
            return None
        pngs: list[Path] = []
        for tf in text_files[:30]:  # 最多 30 张
            try:
                content = tf.read_text(encoding="utf-8")
            except Exception:
                continue
            lines = content.splitlines()[:20]
            png = self.tmp_dir / f"text_{tf.stem}.png"
            render_text_card(png, lines,
                             width=self.width, height=self.height)
            pngs.append(png)
        if not pngs:
            return None
        mp4 = self.tmp_dir / "text.mp4"
        ok = png_sequence_to_mp4(
            pngs, mp4, fps=self.fps, duration_each=1.2,
            width=self.width, height=self.height,
        )
        if not ok:
            return None
        return self._with_caption(mp4, "阶段 2：文本宇宙探索（百科阅读）")

    def _build_crossmodal_segment(self) -> Optional[Path]:
        frames = sorted((self.input_dir / "frames").glob("crossmodal_*.png"))
        if not frames:
            return None
        mp4 = self.tmp_dir / "crossmodal.mp4"
        ok = png_sequence_to_mp4(
            frames, mp4, fps=self.fps, duration_each=0.6,
            width=self.width, height=self.height,
        )
        if not ok:
            return None
        return self._with_caption(mp4, "阶段 3：跨模态统一（对齐训练）")

    def _build_qa_segment(self) -> Optional[Path]:
        qa_files = sorted((self.input_dir / "qa").glob("qa_*.txt"))
        if not qa_files:
            # 用一个空提示卡
            png = self.tmp_dir / "qa_empty.png"
            render_text_card(png, ["（问答阶段未产生日志）"],
                             width=self.width, height=self.height)
            mp4 = self.tmp_dir / "qa.mp4"
            if single_png_to_mp4(png, mp4, duration=2.0, fps=self.fps,
                                 width=self.width, height=self.height):
                return self._with_caption(mp4, "阶段 4：自我提问与知识整合")
            return None
        pngs: list[Path] = []
        for qf in qa_files[:20]:
            try:
                content = qf.read_text(encoding="utf-8")
            except Exception:
                continue
            lines = content.splitlines()[:20]
            png = self.tmp_dir / f"qa_{qf.stem}.png"
            render_text_card(png, lines,
                             width=self.width, height=self.height)
            pngs.append(png)
        mp4 = self.tmp_dir / "qa.mp4"
        ok = png_sequence_to_mp4(
            pngs, mp4, fps=self.fps, duration_each=1.5,
            width=self.width, height=self.height,
        )
        if not ok:
            return None
        return self._with_caption(mp4, "阶段 4：自我提问与知识整合")

    def _build_fe_curve_segment(self) -> Optional[Path]:
        csv_path = self.input_dir / "curves" / "free_energy.csv"
        if not csv_path.exists():
            return None
        png = self.tmp_dir / "fe_curve.png"
        if not render_fe_curve(csv_path, png,
                               width=self.width, height=self.height):
            return None
        mp4 = self.tmp_dir / "fe_curve.mp4"
        if single_png_to_mp4(png, mp4, duration=4.0, fps=self.fps,
                             width=self.width, height=self.height):
            return self._with_caption(mp4, "自由能曲线 — 学习动力学")
        return None

    def _build_latent_segment(self) -> Optional[Path]:
        latent_dir = self.input_dir / "latent"
        if not latent_dir.exists():
            return None
        png = self.tmp_dir / "latent.png"
        if not render_latent_scatter(latent_dir, png,
                                     width=self.width, height=self.height):
            return None
        mp4 = self.tmp_dir / "latent.mp4"
        if single_png_to_mp4(png, mp4, duration=3.0, fps=self.fps,
                             width=self.width, height=self.height):
            return self._with_caption(mp4, "隐空间轨迹（PCA 2D）")
        return None

    def _build_milestones_segment(self) -> Optional[Path]:
        milestones = self.report.get("milestones", [])
        if not milestones:
            return None
        lines = ["里程碑记录：", ""]
        for ms in milestones[:15]:
            step = ms.get("step", "?")
            title = ms.get("title", "")
            stage = ms.get("stage", "")
            lines.append(f"[step {step}] [{stage}] {title}")
        png = self.tmp_dir / "milestones.png"
        render_text_card(png, lines,
                         width=self.width, height=self.height,
                         bg="#1a1010", fg="#ffd0a0")
        mp4 = self.tmp_dir / "milestones.mp4"
        if single_png_to_mp4(png, mp4, duration=4.0, fps=self.fps,
                             width=self.width, height=self.height):
            return self._with_caption(mp4, f"里程碑汇总（共 {len(milestones)} 个）")
        return None

    def _build_end_card_segment(self) -> Optional[Path]:
        png = self.tmp_dir / "end.png"
        total_steps = self.report.get("total_steps", 0)
        final_fe = self.report.get("final_free_energy", 0)
        subtitle = (f"Total steps: {total_steps}  |  "
                    f"Final FE: {final_fe:.4f}")
        render_title_card(png, "Demo Complete", subtitle,
                          width=self.width, height=self.height,
                          bg="#0a1a0a", fg="#a0ffa0")
        mp4 = self.tmp_dir / "end.mp4"
        if single_png_to_mp4(png, mp4, duration=3.0, fps=self.fps,
                             width=self.width, height=self.height):
            return self._with_caption(mp4, "ZeroDataModel Demo")
        return None

    # ---------------------------------------------------------------- #
    def _with_caption(self, src_mp4: Path, caption: str) -> Path:
        """叠加字幕，返回新 mp4 路径。"""
        dst = src_mp4.with_name(src_mp4.stem + "_cap.mp4")
        if overlay_text(src_mp4, dst, caption):
            src_mp4.unlink(missing_ok=True)
            return dst
        # 失败则保留原视频
        shutil.copy(str(src_mp4), str(dst))
        src_mp4.unlink(missing_ok=True)
        return dst


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="基于 demo_output 素材自动生成演示视频",
    )
    parser.add_argument("--input", type=str, default="demo_output",
                        help="demo_cognitive_origin.py 的输出目录")
    parser.add_argument("--out", type=str, default="video_assets",
                        help="视频输出目录")
    parser.add_argument("--fps", type=int, default=10,
                        help="输出视频帧率 (默认 10)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-curves", action="store_true",
                        help="跳过自由能曲线渲染（无需 matplotlib）")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"错误：输入目录不存在: {input_dir}", file=sys.stderr)
        print("请先运行: python demo_cognitive_origin.py --output "
              f"{args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = DemoVideoBuilder(
        input_dir=input_dir,
        out_dir=out_dir,
        fps=args.fps,
        width=args.width,
        height=args.height,
        draw_curves=not args.no_curves,
    )
    builder.build()


if __name__ == "__main__":
    main()
