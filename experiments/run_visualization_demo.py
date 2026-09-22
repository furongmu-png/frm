# experiments/run_visualization_demo.py
"""Visualization demo — runs the sandbox loop with live WebSocket streaming.

This script demonstrates the "Window of Consciousness" backend:
  1. Starts a WebSocket streamer on ws://localhost:8765
  2. Starts a REST API on http://localhost:8000
  3. Runs a physics sandbox loop for N steps
  4. After each think(), collects a snapshot and broadcasts it to all
     connected frontend clients

Usage:
    python experiments/run_visualization_demo.py --steps 500

Then open the frontend (frontend/) in a browser to see live updates.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from physics_sandbox import PhysicsSandbox
from image_preprocessor import ImagePreprocessor
from experience_buffer import ExperienceBuffer
from run_closed_loop import CycleRecord, SANDBOX_ACTIONS, discretise_action
from zero_data_model.model import ZeroDataModel
from visualization.snapshot import SnapshotCollector
from visualization.streamer import WebSocketStreamer
from visualization.api_server import VisualizationAPI
from visualization.milestone_detector import MilestoneDetector


DEFAULT_STEPS = 500
DEFAULT_DIM = 32
DEFAULT_SEED = 42
DEFAULT_NUM_OBJECTS = 2
WS_PORT = 8765
REST_PORT = 8000
BELIEF_DECAY = 0.99


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualization demo: live model state streaming."
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_objects", type=int, default=DEFAULT_NUM_OBJECTS)
    parser.add_argument("--ws_port", type=int, default=WS_PORT)
    parser.add_argument("--rest_port", type=int, default=REST_PORT)
    parser.add_argument("--ws_host", type=str, default="localhost",
                        help="WebSocket bind host (use 0.0.0.0 for Docker).")
    parser.add_argument("--rest_host", type=str, default="localhost",
                        help="REST API bind host (use 0.0.0.0 for Docker).")
    parser.add_argument(
        "--modality", type=str, default="physics",
        choices=["physics", "text", "crossmodal"],
        help="Active modality label pushed into each snapshot.",
    )
    parser.add_argument(
        "--text_file", type=str, default="",
        help="Path to a text file (used when --modality=text). If empty, "
             "a bundled sample essay is generated.",
    )
    args = parser.parse_args()

    # --- Initialize model + encoders based on modality --- #
    model = ZeroDataModel(dim=args.dim, seed=args.seed)
    engine = model.active_inference

    sandbox = None
    text_stream = None
    image_encoder = None
    text_encoder = None
    sample_text_path = Path(__file__).resolve().parent / "output" / "_sample_text.txt"

    if args.modality == "text":
        # Text mode: import TextStream + TextEncoder lazily.
        from text_stream import TextStream  # type: ignore
        from text_encoder import TextEncoder  # type: ignore
        text_path = Path(args.text_file) if args.text_file else sample_text_path
        if not text_path.exists():
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(
                "Cognitive emergence is the process by which a system "
                "develops understanding from raw sensory experience. "
                "The ZeroDataModel begins with no prior knowledge and "
                "learns by minimising free energy across modalities. "
                "As the model reads more text, it builds a knowledge "
                "graph linking concepts that co-occur in the stream. "
                "Prediction error drops as familiar patterns recur. "
                "This is the essence of active inference: the model "
                "acts to reduce surprise, and surprise reduction is "
                "learning. Over thousands of cycles, structure that "
                "was invisible in any single observation becomes "
                "visible in the joint distribution of beliefs.",
                encoding="utf-8",
            )
        text_stream = TextStream(
            file_path=text_path,
            block_size=128, seed=args.seed,
        )
        text_encoder = TextEncoder(output_dim=args.dim, seed=args.seed)
    else:
        # Physics / crossmodal: use the image-pipeline sandbox.
        sandbox = PhysicsSandbox(num_objects=args.num_objects, seed=args.seed)
        image_encoder = ImagePreprocessor(
            output_dim=args.dim, img_size=128, seed=args.seed,
        )

    # --- Initialize visualization backend --- #
    collector = SnapshotCollector(max_history=5000)
    milestone_detector = MilestoneDetector()
    streamer = WebSocketStreamer(host=args.ws_host, port=args.ws_port)
    streamer.start()

    api = VisualizationAPI(
        collector=collector, streamer=streamer,
        milestone_detector=milestone_detector,
        host=args.rest_host, rest_port=args.rest_port,
    )
    api.start()

    print()
    print("=" * 64)
    print("Window of Consciousness — Visualization Demo")
    print("=" * 64)
    print(f"  WebSocket:  ws://localhost:{args.ws_port}")
    print(f"  REST API:   http://localhost:{args.rest_port}")
    print(f"  Steps:      {args.steps}")
    print(f"  Modality:   {args.modality}")
    print(f"  Dim:        {args.dim}")
    print(f"  Seed:       {args.seed}")
    print()
    print("  Open the frontend in a browser to see live updates:")
    print("    cd frontend && npm run dev")
    print()

    # --- Main loop --- #
    # Physics / crossmodal: track the rendered frame.
    # Text: track the text stream + character position.
    frame = sandbox.frame.copy() if sandbox is not None else None
    text_block = ""
    text_pos = -1
    t0 = time.perf_counter()

    for step in range(args.steps):
        # 0. Honour pause / single_step from the streamer.
        #    If paused, wait until either resume or a single_step signal.
        #    For single_step, we run exactly one cycle then re-pause.
        while streamer.paused and not streamer.single_step_pending:
            # Drain any pending commands (e.g. resume) while waiting.
            cmd = streamer.get_pending_command()
            if cmd:
                print(f"  [cmd] step {step}: {cmd}")
            time.sleep(0.05)
        # Consume the single_step signal if it was set.
        if streamer.single_step_pending:
            streamer.clear_single_step()

        # Drain any other pending commands (set_speed, inject_question, ...).
        cmd = streamer.get_pending_command()
        if cmd:
            print(f"  [cmd] step {step}: {cmd}")

        # 1. Encode observation + gather modality-specific context.
        if args.modality == "text" and text_stream is not None and text_encoder is not None:
            text_block = text_stream.step()
            text_pos = text_stream.tell()
            obs = text_encoder.encode(text_block)
        else:
            # physics / crossmodal
            assert image_encoder is not None and sandbox is not None
            obs = image_encoder.encode(frame)
            text_pos = -1
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Think.
        signal = model.think(obs)
        gm = engine.generative_model
        gm.belief_state = gm.belief_state * BELIEF_DECAY

        # 3. Select action (and step the sandbox for physics modalities).
        belief = gm.belief_state
        action_vec = engine.select_action(belief, current_observation=obs)
        action = discretise_action(action_vec)
        next_frame = sandbox.step(action) if sandbox is not None else None

        # 4. Compute pragmatic free energy.
        predicted = gm.predict_observation(belief)
        err = obs[:len(predicted)] - predicted[:len(obs)]
        if len(err) < gm.obs_dim:
            err = np.pad(err, (0, gm.obs_dim - len(err)))
        fe = float(np.dot(err, err)) / err.size
        pe = float(np.linalg.norm(obs - belief[:len(obs)]))

        # 5. Collect snapshot and push to streamer.
        #    `push` is an alias for `broadcast`; the streamer's broadcast
        #    loop will gate on paused/single_step and append to history.
        snap = collector.collect(
            model=model,
            step=step,
            modality=args.modality,
            frame=frame,
            text_block=text_block,
            text_pos=text_pos,
            free_energy=fe,
            prediction_error=pe,
            beta=float(getattr(engine, "_exploration_step", 0)),
            # Phase G (四.4): forward think()'s signal.metadata so the
            # snapshot collector can populate s4_state / layer_errors /
            # memory_retrieved from the cognitive-upgrade hooks.
            metadata=getattr(signal, "metadata", None),
        )
        streamer.push(snap.to_dict())

        milestone = milestone_detector.analyze(snap.to_dict())
        if milestone:
            print(f"  [milestone] step {step}: {milestone.title}")

        # 6. Advance.
        if next_frame is not None:
            frame = next_frame

        if (step + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = (step + 1) / max(elapsed, 1e-9)
            print(
                f"  step {step + 1:4d}/{args.steps}  "
                f"FE={fe:.6f}  PE={pe:.4f}  "
                f"belief_norm={float(np.linalg.norm(belief)):.4f}  "
                f"clients={streamer.n_clients}  "
                f"rate={rate:.1f} cyc/s"
            )

    elapsed = time.perf_counter() - t0
    print()
    print(f"Done: {args.steps} steps in {elapsed:.1f}s")
    print(f"  Final FE: {fe:.6f}")
    print(f"  History: {collector.history_size} snapshots")
    print(f"  Milestones: {len(collector.get_milestones())}")
    print()
    print("Keeping server alive for 10s for late connections...")
    time.sleep(10)
    streamer.stop()


if __name__ == "__main__":
    main()
