# experiments/image_preprocessor.py
"""Fixed random-projection image preprocessor.

Converts a 128x128 grayscale frame from ``PhysicsSandbox`` into a
fixed-dim observation vector suitable for ``ZeroDataModel.think()``.

Design:
- Zero-pretraining principle: the projection matrix is a FIXED
  Gaussian random matrix (Johnson-Lindenstrauss-style), seeded for
  reproducibility. No learning happens here.
- Normalisation: divide pixel values by 255.0 so inputs are in [0, 1].
- Optional squashing: ``np.tanh`` after projection to keep values in
  (-1, 1), which tends to play nicely with the active-inference
  belief-state update.

Usage:

    from image_preprocessor import ImagePreprocessor
    pp = ImagePreprocessor(output_dim=32, seed=42)
    obs = pp.encode(frame)   # frame: (128, 128) uint8 -> (32,) float64

Run the self-test with:

    python experiments/image_preprocessor.py
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Default output dimensionality — matches ZeroDataModel's default dim.
DEFAULT_OUTPUT_DIM = 32
DEFAULT_IMG_SIZE = 128          # PhysicsSandbox frame side
DEFAULT_SEED = 42


class ImagePreprocessor:
    """Fixed random-projection image encoder.

    The encoder flattens the input image, divides by 255.0, and
    applies a fixed Gaussian random projection matrix
    ``W`` of shape ``(output_dim, n_pixels)``. An optional ``tanh``
    squashing step keeps the output in ``(-1, 1)``.

    Parameters
    ----------
    output_dim : int
        Dimensionality of the returned observation vector. Must
        match ``ZeroDataModel(dim=...)`` for direct feeding into
        ``model.think(obs)``.
    img_size : int
        Side length of the (square) input frame. Defaults to 128 to
        match ``PhysicsSandbox``.
    seed : int
        Seed for the random projection matrix. Same seed + same
        input => same output, across runs and platforms.
    use_tanh : bool
        If True, apply ``np.tanh`` after projection. Recommended.
    """

    def __init__(
        self,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        img_size: int = DEFAULT_IMG_SIZE,
        seed: int = DEFAULT_SEED,
        use_tanh: bool = True,
    ):
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim!r}")
        if img_size < 1:
            raise ValueError(f"img_size must be >= 1, got {img_size!r}")
        self.output_dim = int(output_dim)
        self.img_size = int(img_size)
        self.seed = int(seed)
        self.use_tanh = bool(use_tanh)
        # Fixed random projection matrix. ``default_rng`` is the
        # modern, reproducible NumPy RNG — its stream is deterministic
        # given the seed and does not touch the global RNG state.
        rng = np.random.default_rng(self.seed)
        n_pixels = self.img_size * self.img_size
        # Scale by 1/sqrt(n_pixels) so the projected vector's expected
        # L2 norm is comparable to the input's — standard JL embedding.
        self._W = rng.standard_normal(
            (self.output_dim, n_pixels)
        ) / np.sqrt(n_pixels)

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #
    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Encode a grayscale frame into a fixed-dim observation vector.

        Parameters
        ----------
        frame : np.ndarray
            ``(H, W)`` uint8 array (or any numeric array). If the
            shape is not ``(img_size, img_size)`` it is resized via
            ``np.resize`` — this is a cheap nearest-neighbour fallback
            and is sufficient for the sandbox's deterministic sizes.

        Returns
        -------
        np.ndarray
            ``(output_dim,)`` float64 vector. Values in ``(-1, 1)``
            if ``use_tanh=True`` else unbounded.
        """
        arr = np.asarray(frame)
        if arr.shape != (self.img_size, self.img_size):
            # Cheap resize — works for the sandbox's exact sizes.
            arr = np.resize(arr, (self.img_size, self.img_size))
        # Flatten + normalise to [0, 1].
        x = arr.astype(np.float64).flatten() / 255.0
        # Project.
        y = self._W @ x
        if self.use_tanh:
            y = np.tanh(y)
        return y

    # Convenient alias.
    __call__ = encode

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def projection_matrix(self) -> np.ndarray:
        """Read-only view of the (output_dim, n_pixels) projection matrix."""
        return self._W


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
def _self_test() -> None:
    """Quick smoke test: encode a known frame, verify determinism."""
    # Build a fake 128x128 frame.
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    pp = ImagePreprocessor(output_dim=32, seed=42)
    obs1 = pp.encode(frame)
    obs2 = pp.encode(frame)
    print(f"output shape: {obs1.shape}")
    print(f"output dtype: {obs1.dtype}")
    print(f"output range: [{obs1.min():.4f}, {obs1.max():.4f}]")
    print(f"output norm: {np.linalg.norm(obs1):.4f}")
    # Determinism.
    assert np.array_equal(obs1, obs2), "encode is non-deterministic!"
    print("[OK] determinism: same frame -> same obs")
    # Seed sensitivity.
    pp_other = ImagePreprocessor(output_dim=32, seed=7)
    obs_other = pp_other.encode(frame)
    assert not np.array_equal(obs1, obs_other), \
        "different seeds produced identical projections!"
    print("[OK] seed sensitivity: different seeds -> different obs")
    # Zero-frame sanity (should be ~0 after tanh).
    zero_frame = np.zeros((128, 128), dtype=np.uint8)
    zero_obs = pp.encode(zero_frame)
    print(f"zero-frame obs max abs: {np.abs(zero_obs).max():.6f}")
    assert np.abs(zero_obs).max() < 1e-12, "zero frame produced non-zero obs!"
    print("[OK] zero-frame: produces zero observation")
    # Integration with PhysicsSandbox.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from physics_sandbox import PhysicsSandbox
    sandbox = PhysicsSandbox(num_objects=2, seed=42)
    obs = pp.encode(sandbox.frame)
    print(f"sandbox frame_mean={sandbox.frame.mean():.2f}  "
          f"obs_norm={np.linalg.norm(obs):.4f}")
    print("[OK] integration with PhysicsSandbox")
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    _self_test()
