# experiments/run_phase_g_comparison.py
"""Phase G 升级前后对比：物理沙盒 5000 步自由能下降对比。

对应 spec §五.2 的要求：
    在物理沙盒中运行5000步，对比升级前后的自由能下降速度和收敛水平。

流程：
1. 基线（baseline）：use_s4=False, use_pcn=False, use_hopfield=False
2. 升级后（phase_g）：use_s4=True, use_pcn=True, use_hopfield=True
3. 两次运行使用相同的随机种子与初始帧，确保可比性
4. 对比指标：
   - 初始自由能（前 100 步平均）
   - 收敛自由能（后 500 步平均）
   - 自由能下降速度（前 1000 步线性回归斜率）
   - 最终自由能
   - 每步平均延迟

输出：升级前后对比表（stdout + JSON 文件）。

Run with:  python experiments/run_phase_g_comparison.py [--steps 5000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make src/ + examples/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from zero_data_model.model import ZeroDataModel  # noqa: E402
from physics_sandbox import PhysicsSandbox  # noqa: E402


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
DEFAULT_DIM = 64
DEFAULT_STEPS = 5000
DEFAULT_SEED = 42
DEFAULT_NUM_OBJECTS = 2
PROGRESS_INTERVAL = 1000


@dataclass
class RunResult:
    """Per-run aggregated metrics."""
    label: str                       # "baseline" | "phase_g"
    steps: int
    initial_fe: float                # mean of first 100 steps
    final_fe: float                  # mean of last 500 steps
    fe_descent_rate: float           # linear slope of FE over first 1000 steps
    fe_history: list[float] = field(default_factory=list)
    mean_step_latency_ms: float = 0.0
    total_runtime_s: float = 0.0


def discretise_action(action_vec: np.ndarray) -> int:
    """Map continuous action vector to one of 4 sandbox actions.

    Mirrors run_sandbox_closed_loop.py's discretisation:
        dominant axis > 0.1  →  directional action
        else                 →  no-op (3)
    """
    if action_vec.size == 0:
        return 3
    a = action_vec.flatten()
    if a.size >= 2:
        ax, ay = float(a[0]), float(a[1])
    else:
        ax, ay = float(a[0]), 0.0
    if abs(ax) < 0.1 and abs(ay) < 0.1:
        return 3
    if abs(ax) > abs(ay):
        return 0 if ax > 0 else 1
    return 2


def run_one(
    label: str,
    use_s4: bool,
    use_pcn: bool,
    use_hopfield: bool,
    steps: int,
    dim: int,
    seed: int,
) -> RunResult:
    """Run ``steps`` cycles with the given upgrade flags and return metrics."""
    print(f"\n{'=' * 60}")
    print(f"  Running {label}: use_s4={use_s4}, use_pcn={use_pcn}, "
          f"use_hopfield={use_hopfield}, steps={steps}")
    print(f"{'=' * 60}")

    sandbox = PhysicsSandbox(num_objects=DEFAULT_NUM_OBJECTS, seed=seed)
    model = ZeroDataModel(
        dim=dim, seed=seed,
        use_s4=use_s4, use_pcn=use_pcn, use_hopfield=use_hopfield,
    )

    # Pre-populate Hopfield memory so retrieval isn't trivially empty
    # (only when use_hopfield=True). Mirror the demo's pattern: store
    # a few prototypical belief vectors.
    if use_hopfield:
        rng = np.random.default_rng(seed)
        for _ in range(5):
            v = rng.standard_normal(dim)
            v = v / (np.linalg.norm(v) + 1e-12)
            model.hopfield_memory.store(v, v)

    frame = sandbox.reset(seed=seed)
    fe_history: list[float] = []
    latencies_ms: list[float] = []

    t0 = time.perf_counter()
    for step in range(steps):
        # Encode frame → observation (flatten + downsample to dim)
        obs = frame.flatten().astype(np.float64)
        if obs.size > dim:
            # Stride-sample down to dim
            idx = np.linspace(0, obs.size - 1, dim).astype(int)
            obs = obs[idx]
        elif obs.size < dim:
            obs = np.pad(obs, (0, dim - obs.size))
        # Normalize to [-1, 1]
        if obs.max() > obs.min():
            obs = 2.0 * (obs - obs.min()) / (obs.max() - obs.min()) - 1.0

        # think() — measure latency
        t_step = time.perf_counter()
        signal = model.think(obs)
        latency_ms = (time.perf_counter() - t_step) * 1000.0
        latencies_ms.append(latency_ms)

        gm = model.active_inference.generative_model
        belief = gm.belief_state

        # Discretise action and step the sandbox
        action_vec = model.active_inference.select_action(belief, current_observation=obs)
        action = discretise_action(action_vec)
        frame = sandbox.step(action)

        # Compute pragmatic free energy (spec §五.2 metric)
        predicted = gm.predict_observation(belief)
        err = obs[: len(predicted)] - predicted[: len(obs)]
        if len(err) < gm.obs_dim:
            err = np.pad(err, (0, gm.obs_dim - len(err)))
        fe = float(np.dot(err, err)) / max(err.size, 1)
        fe_history.append(fe)

        if (step + 1) % PROGRESS_INTERVAL == 0:
            elapsed = time.perf_counter() - t0
            recent_fe = float(np.mean(fe_history[-PROGRESS_INTERVAL:]))
            print(
                f"  step {step + 1:5d}/{steps}  "
                f"FE={fe:.6f}  "
                f"recent_FE={recent_fe:.6f}  "
                f"latency={latency_ms:.2f}ms  "
                f"elapsed={elapsed:.1f}s"
            )

    total_runtime = time.perf_counter() - t0

    # Aggregate metrics
    fe_arr = np.array(fe_history)
    initial_fe = float(np.mean(fe_arr[:100])) if len(fe_arr) >= 100 else float(np.mean(fe_arr))
    final_fe = float(np.mean(fe_arr[-500:])) if len(fe_arr) >= 500 else float(np.mean(fe_arr[-100:]))
    # Linear descent rate over first 1000 steps (slope of best-fit line)
    n_fit = min(1000, len(fe_arr))
    x = np.arange(n_fit)
    y = fe_arr[:n_fit]
    if n_fit >= 2 and np.std(y) > 1e-12:
        slope, _ = np.polyfit(x, y, 1)
        fe_descent_rate = float(slope)
    else:
        fe_descent_rate = 0.0
    mean_latency = float(np.mean(latencies_ms))

    result = RunResult(
        label=label,
        steps=steps,
        initial_fe=initial_fe,
        final_fe=final_fe,
        fe_descent_rate=fe_descent_rate,
        fe_history=fe_history,
        mean_step_latency_ms=mean_latency,
        total_runtime_s=total_runtime,
    )
    print(
        f"\n  [{label}] DONE — initial_FE={initial_fe:.6f}  "
        f"final_FE={final_fe:.6f}  descent_rate={fe_descent_rate:.2e}/step  "
        f"mean_latency={mean_latency:.2f}ms  total={total_runtime:.1f}s"
    )
    return result


def print_comparison_table(baseline: RunResult, phase_g: RunResult) -> None:
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 72)
    print("  Phase G 升级前后对比表 (Phase G Before/After Comparison)")
    print("=" * 72)
    print(f"{'指标 (Metric)':<32} {'Baseline':>16} {'Phase G':>16} {'Δ':>8}")
    print("-" * 72)
    # Initial FE
    delta = phase_g.initial_fe - baseline.initial_fe
    print(f"{'初始自由能 (initial FE)':<32} {baseline.initial_fe:>16.6f} "
          f"{phase_g.initial_fe:>16.6f} {delta:>+8.4f}")
    # Final FE
    delta = phase_g.final_fe - baseline.final_fe
    pct = (delta / max(abs(baseline.final_fe), 1e-12)) * 100
    print(f"{'收敛自由能 (final FE)':<32} {baseline.final_fe:>16.6f} "
          f"{phase_g.final_fe:>16.6f} {delta:>+8.4f} ({pct:+.1f}%)")
    # Descent rate
    delta = phase_g.fe_descent_rate - baseline.fe_descent_rate
    print(f"{'下降速度 (descent rate)':<32} {baseline.fe_descent_rate:>16.4e} "
          f"{phase_g.fe_descent_rate:>16.4e} {delta:>+8.4e}")
    # Mean latency
    delta = phase_g.mean_step_latency_ms - baseline.mean_step_latency_ms
    pct = (delta / max(baseline.mean_step_latency_ms, 1e-12)) * 100
    print(f"{'平均延迟 (mean latency, ms)':<32} {baseline.mean_step_latency_ms:>16.2f} "
          f"{phase_g.mean_step_latency_ms:>16.2f} {delta:>+8.2f} ({pct:+.1f}%)")
    # Total runtime
    delta = phase_g.total_runtime_s - baseline.total_runtime_s
    pct = (delta / max(baseline.total_runtime_s, 1e-12)) * 100
    print(f"{'总耗时 (total runtime, s)':<32} {baseline.total_runtime_s:>16.1f} "
          f"{phase_g.total_runtime_s:>16.1f} {delta:>+8.1f} ({pct:+.1f}%)")
    print("-" * 72)

    # Verdict
    print("\n评估 (Verdict):")
    if phase_g.final_fe < baseline.final_fe:
        improvement = (baseline.final_fe - phase_g.final_fe) / max(abs(baseline.final_fe), 1e-12) * 100
        print(f"  ✓ Phase G 收敛自由能更低 (改善 {improvement:.1f}%)")
    else:
        print(f"  ✗ Phase G 收敛自由能未改善")
    if abs(phase_g.fe_descent_rate) > abs(baseline.fe_descent_rate):
        print(f"  ✓ Phase G 下降速度更快 (|slope| 更大)")
    else:
        print(f"  ✗ Phase G 下降速度未加快")
    latency_pct = (
        (phase_g.mean_step_latency_ms - baseline.mean_step_latency_ms)
        / max(baseline.mean_step_latency_ms, 1e-12) * 100
    )
    if latency_pct <= 20.0:
        print(f"  ✓ think() 延迟增加 {latency_pct:+.1f}% (满足 < 20% 要求)")
    else:
        print(f"  ✗ think() 延迟增加 {latency_pct:+.1f}% (超过 20% 阈值)")


def main():
    parser = argparse.ArgumentParser(
        description="Phase G 升级前后物理沙盒对比"
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"每个配置运行的步数 (default: {DEFAULT_STEPS})")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM,
                        help=f"模型维度 (default: {DEFAULT_DIM})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"随机种子 (default: {DEFAULT_SEED})")
    parser.add_argument("--output", type=str, default="phase_g_comparison.json",
                        help="JSON 输出文件名")
    args = parser.parse_args()

    # 1. Baseline (all upgrades off)
    baseline = run_one(
        label="baseline",
        use_s4=False, use_pcn=False, use_hopfield=False,
        steps=args.steps, dim=args.dim, seed=args.seed,
    )

    # 2. Phase G (all upgrades on)
    phase_g = run_one(
        label="phase_g",
        use_s4=True, use_pcn=True, use_hopfield=True,
        steps=args.steps, dim=args.dim, seed=args.seed,
    )

    # 3. Print comparison
    print_comparison_table(baseline, phase_g)

    # 4. Save JSON
    out_path = Path(__file__).resolve().parent / args.output
    report = {
        "baseline": {
            "initial_fe": baseline.initial_fe,
            "final_fe": baseline.final_fe,
            "fe_descent_rate": baseline.fe_descent_rate,
            "mean_step_latency_ms": baseline.mean_step_latency_ms,
            "total_runtime_s": baseline.total_runtime_s,
            "fe_history_sample": baseline.fe_history[::50],  # downsample for storage
        },
        "phase_g": {
            "initial_fe": phase_g.initial_fe,
            "final_fe": phase_g.final_fe,
            "fe_descent_rate": phase_g.fe_descent_rate,
            "mean_step_latency_ms": phase_g.mean_step_latency_ms,
            "total_runtime_s": phase_g.total_runtime_s,
            "fe_history_sample": phase_g.fe_history[::50],
        },
        "config": {
            "steps": args.steps,
            "dim": args.dim,
            "seed": args.seed,
        },
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nJSON 报告已保存: {out_path}")


if __name__ == "__main__":
    main()
