# experiments/multimodal_bridge.py
"""Multimodal bridge: align physics and text into a shared hidden space.

The bridge wraps TWO frozen modality encoders (``ImagePreprocessor`` for
physics frames, ``TextEncoder`` for text) and adds learnable cross-modal
predictors on top:

    Physics frame  ->  ImagePreprocessor  ->  h_phys   (dim,)
    Text string    ->  TextEncoder        ->  h_text   (dim,)

    Cross-modal predictors (Hebbian, error-driven, no autograd):
        h_text_pred  = W_pt @ h_phys     (physics -> text)
        h_phys_pred  = W_tp @ h_text     (text -> physics)

        error_pt = h_text - h_text_pred
        error_tp = h_phys - h_phys_pred

        # Oja-style update (Hebbian pre×post minus anti-Hebbian normalisation):
        W_pt += lr * (h_phys ⊗ error_pt) - lr * lambda * W_pt * |h_phys|^2
        W_tp += lr * (h_text ⊗ error_tp) - lr * lambda * W_tp * |h_text|^2

The ``fuse()`` method combines both modality hidden states into a single
shared representation:

        h_fused = gate * h_phys + (1 - gate) * h_text
        gate = sigmoid(w_gate . (h_phys || h_text))

When only one modality is available, ``fuse`` falls back to the predicted
counterpart via the cross-modal predictor (asymmetric mode).

Run with:
    # Used as a library by run_crossmodal.py and evaluate_crossmodal.py.
    # Quick standalone sanity check:
    python experiments/multimodal_bridge.py --dim 64 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from image_preprocessor import ImagePreprocessor  # noqa: E402
from text_encoder import TextEncoder  # noqa: E402


# ------------------------------------------------------------------ #
# Defaults
# ------------------------------------------------------------------ #
DEFAULT_DIM = 64           # shared hidden dim (per spec: "建议 64")
DEFAULT_IMG_SIZE = 128
DEFAULT_BLOCK_SIZE = 128
DEFAULT_EMB_DIM = 64
DEFAULT_LR = 0.01
DEFAULT_LAMBDA = 0.001     # weight decay (Oja normalisation)
DEFAULT_SEED = 42


# ------------------------------------------------------------------ #
# MultimodalBridge
# ------------------------------------------------------------------ #
class MultimodalBridge:
    """Shared-space bridge between physics frames and text strings.

    The two modality encoders (ImagePreprocessor, TextEncoder) are FROZEN
    — they are random projections (deterministic given their seeds) that
    reduce raw modality input to a fixed-dim vector. The bridge's LEARNABLE
    parameters are:

      - ``W_pt``: (dim, dim) cross-modal predictor physics → text
      - ``W_tp``: (dim, dim) cross-modal predictor text → physics
      - ``w_gate``: (2*dim,) fusion gate (which modality to trust)

    All updates are error-driven Hebbian (Oja rule); no autograd.
    """

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        img_size: int = DEFAULT_IMG_SIZE,
        block_size: int = DEFAULT_BLOCK_SIZE,
        emb_dim: int = DEFAULT_EMB_DIM,
        lr: float = DEFAULT_LR,
        lam: float = DEFAULT_LAMBDA,
        seed: int = DEFAULT_SEED,
    ):
        self.dim = int(dim)
        self.lr = float(lr)
        self.lam = float(lam)
        self.seed = int(seed)

        # Frozen modality encoders (random projections, fixed seeds).
        self.physics_encoder = ImagePreprocessor(
            output_dim=dim, img_size=img_size, seed=seed,
        )
        self.text_encoder = TextEncoder(
            block_size=block_size, output_dim=dim,
            emb_dim=emb_dim, seed=seed + 1,
        )

        # Learnable cross-modal predictors. Initialised small (Xavier-like).
        rng = np.random.default_rng(seed + 2)
        scale = 1.0 / np.sqrt(dim)
        self.W_pt = rng.normal(0.0, scale, size=(dim, dim)).astype(np.float64)
        self.W_tp = rng.normal(0.0, scale, size=(dim, dim)).astype(np.float64)
        # Fusion gate: sigmoid(w_gate . [h_phys ; h_text]) -> scalar in (0,1).
        self.w_gate = rng.normal(0.0, scale, size=(2 * dim,)).astype(np.float64)

        # Running stats for normalisation (helps Hebbian stability).
        self._h_phys_mean = np.zeros(dim, dtype=np.float64)
        self._h_text_mean = np.zeros(dim, dtype=np.float64)
        self._n_updates = 0

    # ------------------------------------------------------------------ #
    # Encoding (frozen)
    # ------------------------------------------------------------------ #
    def encode_physics(self, frame: np.ndarray) -> np.ndarray:
        """Encode a physics frame to the shared hidden space."""
        h = self.physics_encoder.encode(frame)
        return np.asarray(h, dtype=np.float64).copy()

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text string to the shared hidden space."""
        h = self.text_encoder.encode(text)
        return np.asarray(h, dtype=np.float64).copy()

    # ------------------------------------------------------------------ #
    # Cross-modal prediction
    # ------------------------------------------------------------------ #
    def predict_text_from_physics(self, h_phys: np.ndarray) -> np.ndarray:
        """Predict the text hidden-state from a physics hidden-state."""
        return self.W_pt @ h_phys

    def predict_physics_from_text(self, h_text: np.ndarray) -> np.ndarray:
        """Predict the physics hidden-state from a text hidden-state."""
        return self.W_tp @ h_text

    @staticmethod
    def prediction_error(
        h_target: np.ndarray, h_pred: np.ndarray
    ) -> float:
        """Mean-squared prediction error between two hidden states."""
        diff = h_target - h_pred
        return float(np.dot(diff, diff)) / diff.size

    # ------------------------------------------------------------------ #
    # Hebbian learning (Oja rule, error-driven)
    # ------------------------------------------------------------------ #
    def update(
        self,
        h_phys: np.ndarray,
        h_text: np.ndarray,
        lr: float | None = None,
        lam: float | None = None,
    ) -> dict:
        """One Hebbian update step given a co-occurring (h_phys, h_text) pair.

        Updates W_pt (physics→text) and W_tp (text→physics) using the
        Oja rule: ``ΔW = lr * (pre ⊗ post_error) - lr * λ * W * |pre|^2``.

        Hidden states are L2-normalised before the update to stabilise
        the Hebbian dynamics across modalities with different norms.

        Returns a dict of diagnostics (errors, weight norms).
        """
        lr = self.lr if lr is None else float(lr)
        lam = self.lam if lam is None else float(lam)

        # Normalise inputs to unit norm for stable Hebbian learning.
        h_phys_n = h_phys / (np.linalg.norm(h_phys) + 1e-12)
        h_text_n = h_text / (np.linalg.norm(h_text) + 1e-12)

        # Physics -> text prediction (on normalised inputs).
        h_text_pred = self.predict_text_from_physics(h_phys_n)
        err_pt = h_text_n - h_text_pred            # (dim,)
        # Oja update: pre = h_phys_n, post_error = err_pt.
        delta_W_pt = lr * np.outer(h_phys_n, err_pt) - lam * lr * self.W_pt * float(
            np.dot(h_phys_n, h_phys_n)
        )
        self.W_pt += delta_W_pt

        # Text -> physics prediction (on normalised inputs).
        h_phys_pred = self.predict_physics_from_text(h_text_n)
        err_tp = h_phys_n - h_phys_pred
        delta_W_tp = lr * np.outer(h_text_n, err_tp) - lam * lr * self.W_tp * float(
            np.dot(h_text_n, h_text_n)
        )
        self.W_tp += delta_W_tp

        # Update fusion gate: nudge toward trusting the more predictive modality.
        err_pt_mag = float(np.dot(err_pt, err_pt))
        err_tp_mag = float(np.dot(err_tp, err_tp))
        gate_target = 1.0 if err_pt_mag < err_tp_mag else 0.0
        joint = np.concatenate([h_phys_n, h_text_n])
        gate_val = self._gate_value(h_phys_n, h_text_n)
        gate_err = gate_val - gate_target
        self.w_gate -= lr * gate_err * joint

        # Weight norm clipping — prevents Hebbian divergence. The Oja
        # weight decay is too weak to counteract positive feedback when
        # the same experiences are replayed multiple times (consolidation).
        # Hard cap at MAX_W_NORM keeps the matrices in a useful range.
        MAX_W_NORM = 15.0
        for W in (self.W_pt, self.W_tp):
            n = float(np.linalg.norm(W))
            if n > MAX_W_NORM:
                W *= MAX_W_NORM / n

        # Running means (for normalisation/diagnostics).
        self._n_updates += 1
        n = self._n_updates
        self._h_phys_mean += (h_phys - self._h_phys_mean) / n
        self._h_text_mean += (h_text - self._h_text_mean) / n

        return {
            "err_pt_mse": err_pt_mag / err_pt.size,
            "err_tp_mse": err_tp_mag / err_tp.size,
            "W_pt_norm": float(np.linalg.norm(self.W_pt)),
            "W_tp_norm": float(np.linalg.norm(self.W_tp)),
            "gate_value": float(gate_val),
        }

    # ------------------------------------------------------------------ #
    # Fusion
    # ------------------------------------------------------------------ #
    def _gate_value(self, h_phys: np.ndarray, h_text: np.ndarray) -> float:
        """Compute the sigmoid fusion gate in (0, 1)."""
        joint = np.concatenate([h_phys, h_text])
        return float(1.0 / (1.0 + np.exp(-np.dot(self.w_gate, joint))))

    def fuse(
        self,
        h_phys: np.ndarray | None = None,
        h_text: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fuse physics and text hidden states into one shared vector.

        - Both modalities present: gated average.
        - Only physics: predict text via W_pt, then fuse with gate=0.5.
        - Only text:    predict physics via W_tp, then fuse with gate=0.5.
        - Neither:      return zero vector.
        """
        if h_phys is None and h_text is None:
            return np.zeros(self.dim, dtype=np.float64)

        if h_phys is None:
            # Asymmetric mode: predict physics from text.
            h_phys = self.predict_physics_from_text(h_text)
            gate = 0.5
        elif h_text is None:
            # Asymmetric mode: predict text from physics.
            h_text = self.predict_text_from_physics(h_phys)
            gate = 0.5
        else:
            gate = self._gate_value(h_phys, h_text)

        return gate * h_phys + (1.0 - gate) * h_text

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> str:
        """Save learnable parameters to .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W_pt=self.W_pt,
            W_tp=self.W_tp,
            w_gate=self.w_gate,
            h_phys_mean=self._h_phys_mean,
            h_text_mean=self._h_text_mean,
            n_updates=self._n_updates,
            dim=self.dim,
            lr=self.lr,
            lam=self.lam,
            seed=self.seed,
        )
        return str(path)

    def load(self, path: str | Path) -> "MultimodalBridge":
        """Load learnable parameters from .npz (must match config)."""
        data = np.load(path, allow_pickle=False)
        assert int(data["dim"]) == self.dim, (
            f"dim mismatch: bridge={self.dim}, file={int(data['dim'])}"
        )
        self.W_pt = data["W_pt"]
        self.W_tp = data["W_tp"]
        self.w_gate = data["w_gate"]
        self._h_phys_mean = data["h_phys_mean"]
        self._h_text_mean = data["h_text_mean"]
        self._n_updates = int(data["n_updates"])
        return self


# ------------------------------------------------------------------ #
# CLI: quick sanity check
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multimodal bridge sanity check.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bridge = MultimodalBridge(dim=args.dim, seed=args.seed)

    # Tiny synthetic test: encode a fake frame and a fake string, fuse.
    fake_frame = np.zeros((128, 128), dtype=np.uint8)
    fake_frame[10:20, 10:20] = 200
    fake_text = "The agent moved left."

    h_p = bridge.encode_physics(fake_frame)
    h_t = bridge.encode_text(fake_text)
    h_fused = bridge.fuse(h_p, h_t)

    print("=" * 64)
    print("MultimodalBridge sanity check")
    print("=" * 64)
    print(f"  dim            = {bridge.dim}")
    print(f"  h_phys shape    = {h_p.shape}, norm = {np.linalg.norm(h_p):.4f}")
    print(f"  h_text shape    = {h_t.shape}, norm = {np.linalg.norm(h_t):.4f}")
    print(f"  h_fused norm    = {np.linalg.norm(h_fused):.4f}")

    # One update step.
    diag = bridge.update(h_p, h_t)
    print("\nAfter 1 update step:")
    for k, v in diag.items():
        print(f"  {k:<14} = {v:.6f}")


if __name__ == "__main__":
    main()
