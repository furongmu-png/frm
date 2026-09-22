# experiments/image_encoder.py
"""Image encoder for the omnimodal phase (Phase K, Task 1.1).

A ZERO-PRETRAIN image encoder:
  1. Resize any image to 128×128 grayscale (using Pillow if available,
     else a pure-numpy nearest-neighbour resample).
  2. Flatten to a (16384,) vector and standardise (zero-mean, unit-norm).
  3. Project through a FIXED random Gaussian matrix (128×16384 → 128×dim)
     seeded deterministically. Same seed → same projection → reproducible.

The encoder is intentionally minimal — there is NO CNN, NO pretrained
backbone, NO learned weights. The "intelligence" lives in the random
projection (Johnson-Lindenstrauss-style dimensionality reduction)
plus the downstream ZeroDataModel's belief updating.

An ``ImageStream`` class loads images from a directory (one at a
time, lazy) and yields them as encoded vectors + the source path.

Determinism: with a fixed ``seed``, the projection matrix is identical
across runs, so the same image → same vector.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Pillow for image loading + resizing (optional — we have a numpy fallback).
try:
    from PIL import Image as _PIL_Image
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False


DEFAULT_IMG_SIZE = 128           # 128×128 grayscale
DEFAULT_OUTPUT_DIM = 32          # match ZeroDataModel.dim
DEFAULT_PROJ_DIM = 256           # internal projection dim
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}


# ------------------------------------------------------------------ #
# ImageEncoder
# ------------------------------------------------------------------ #
class ImageEncoder:
    """Fixed random-projection encoder for grayscale images.

    Same seed → same projection matrix → deterministic encodings.
    """

    def __init__(
        self,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        proj_dim: int = DEFAULT_PROJ_DIM,
        img_size: int = DEFAULT_IMG_SIZE,
        seed: int = 42,
    ):
        self.output_dim = int(output_dim)
        self.proj_dim = int(proj_dim)
        self.img_size = int(img_size)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        # Two-stage projection: 16384 -> proj_dim -> output_dim.
        # The intermediate stage acts as a sparse Johnson-Lindenstrauss
        # reducer. We use a single combined matrix for efficiency.
        flat_dim = self.img_size * self.img_size
        self._proj1 = rng.standard_normal((self.proj_dim, flat_dim)) / np.sqrt(flat_dim)
        self._proj2 = rng.standard_normal((self.output_dim, self.proj_dim)) / np.sqrt(self.proj_dim)
        # Non-linearity (tanh) prevents collapse for very large images.
        # No learned parameters.

    def encode(self, image_or_path) -> np.ndarray:
        """Encode an image (path / ndarray / PIL.Image) -> (output_dim,) vector."""
        arr = self._to_grayscale_array(image_or_path)
        arr = self._resize_to(arr, self.img_size)
        flat = arr.flatten().astype(np.float64)
        # Standardise: zero-mean, unit-norm (so brightness doesn't dominate).
        flat -= flat.mean()
        n = float(np.linalg.norm(flat))
        if n > 1e-12:
            flat /= n
        # Project.
        h = np.tanh(self._proj1 @ flat)
        out = self._proj2 @ h
        return out.astype(np.float64)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _to_grayscale_array(self, image_or_path) -> np.ndarray:
        """Coerce the input to a 2D uint8 ndarray (H, W)."""
        if isinstance(image_or_path, np.ndarray):
            arr = image_or_path
            if arr.ndim == 3:
                # RGB → grayscale via standard luminance.
                arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1]
                       + 0.114 * arr[..., 2])
            return arr.astype(np.uint8)
        if _HAS_PIL and isinstance(image_or_path, _PIL_Image.Image):
            img = image_or_path.convert("L")
            return np.asarray(img, dtype=np.uint8)
        # Otherwise treat as a path.
        path = Path(image_or_path)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        if _HAS_PIL:
            img = _PIL_Image.open(path).convert("L")
            return np.asarray(img, dtype=np.uint8)
        # No PIL fallback — only support PNG via a minimal parser? Skip.
        raise RuntimeError(
            "Pillow not available; cannot load image files. Install pillow."
        )

    def _resize_to(self, arr: np.ndarray, size: int) -> np.ndarray:
        """Resize a 2D grayscale array to (size, size)."""
        if arr.shape == (size, size):
            return arr
        if _HAS_PIL:
            img = _PIL_Image.fromarray(arr.astype(np.uint8), mode="L")
            img = img.resize((size, size), _PIL_Image.BILINEAR)
            return np.asarray(img, dtype=np.uint8)
        # Pure-numpy nearest-neighbour (crude but works).
        h, w = arr.shape
        ys = (np.arange(size) * h / size).astype(int)
        xs = (np.arange(size) * w / size).astype(int)
        return arr[np.ix_(ys, xs)]


# ------------------------------------------------------------------ #
# Synthetic image generator (for tests + smoke runs)
# ------------------------------------------------------------------ #
def make_synthetic_image(
    shape: str = "circle",
    size: int = 128,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a simple synthetic grayscale image (uint8, (size, size)).

    Shapes: ``circle``, ``square``, ``triangle``, ``cross``, ``noise``.
    Used as the default source when no real image directory is provided.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    if shape == "circle":
        r = size // 3
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        img[mask] = 200
    elif shape == "square":
        half = size // 4
        img[(cy - half):(cy + half), (cx - half):(cx + half)] = 200
    elif shape == "triangle":
        # Simple isoceles triangle.
        for y in range(cy - size // 3, cy + size // 3):
            half = int((y - (cy - size // 3)) / 2)
            img[y, (cx - half):(cx + half)] = 200
    elif shape == "cross":
        thickness = size // 8
        img[(cy - thickness):(cy + thickness), :] = 200
        img[:, (cx - thickness):(cx + thickness)] = 200
    elif shape == "noise":
        img = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    else:
        raise ValueError(f"unknown shape {shape!r}")
    # Add a small amount of Gaussian noise for robustness.
    if shape != "noise":
        noise = rng.normal(0, 5, img.shape)
        img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return img


def make_synthetic_image_dir(
    output_dir: Path,
    n_per_shape: int = 5,
    shapes: tuple[str, ...] = ("circle", "square", "triangle", "cross", "noise"),
    size: int = 128,
    seed: int = 42,
) -> list[Path]:
    """Write a synthetic image dataset to ``output_dir``.

    Returns the list of saved PNG paths. Requires Pillow.
    """
    if not _HAS_PIL:
        raise RuntimeError("Pillow not available; cannot save PNG files.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for shape in shapes:
        for i in range(n_per_shape):
            img = make_synthetic_image(
                shape=shape, size=size,
                seed=int(rng.integers(0, 2 ** 31 - 1)),
            )
            path = output_dir / f"{shape}_{i:02d}.png"
            _PIL_Image.fromarray(img, mode="L").save(path)
            paths.append(path)
    return paths


# ------------------------------------------------------------------ #
# ImageStream
# ------------------------------------------------------------------ #
class ImageStream:
    """Stream images from a directory, encoding each on demand.

    Iteration order is deterministic (sorted by filename).
    """

    def __init__(
        self,
        image_dir: str | Path,
        encoder: ImageEncoder,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.image_dir = Path(image_dir)
        self.encoder = encoder
        self._paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()
        )
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self._paths)
        self._idx = 0

    def __len__(self) -> int:
        return len(self._paths)

    def reset(self) -> None:
        self._idx = 0

    def step(self) -> tuple[np.ndarray, str]:
        """Return the next (encoded_vector, path) pair. Wraps around."""
        if not self._paths:
            raise RuntimeError("no images in directory")
        path = self._paths[self._idx % len(self._paths)]
        self._idx += 1
        return self.encoder.encode(path), str(path)

    def random(self) -> tuple[np.ndarray, str]:
        """Return a random (encoded_vector, path) pair."""
        if not self._paths:
            raise RuntimeError("no images in directory")
        rng = np.random.default_rng(None)
        path = self._paths[int(rng.integers(0, len(self._paths)))]
        return self.encoder.encode(path), str(path)


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import tempfile
    enc = ImageEncoder(output_dim=32, seed=42)
    print(f"[self-test] encoder built (proj1={enc._proj1.shape}, proj2={enc._proj2.shape})")
    # Encode a synthetic image.
    img = make_synthetic_image("circle", seed=0)
    v = enc.encode(img)
    print(f"  encoded circle: shape={v.shape}  norm={np.linalg.norm(v):.4f}")
    img2 = make_synthetic_image("square", seed=1)
    v2 = enc.encode(img2)
    cos = float(np.dot(v, v2) / (np.linalg.norm(v) * np.linalg.norm(v2)))
    print(f"  encoded square: shape={v2.shape}  cos(circle, square)={cos:.4f}")
    # Determinism check.
    v_again = enc.encode(make_synthetic_image("circle", seed=0))
    assert np.allclose(v, v_again), "encoder should be deterministic given seed"
    print("[self-test] determinism: OK")
    # Synthetic dir + ImageStream.
    if _HAS_PIL:
        tmp = Path(tempfile.mkdtemp())
        paths = make_synthetic_image_dir(tmp, n_per_shape=2, seed=42)
        print(f"[self-test] wrote {len(paths)} synthetic images to {tmp}")
        stream = ImageStream(tmp, enc, seed=42)
        print(f"  stream len = {len(stream)}")
        v, p = stream.step()
        print(f"  first: {p}  vec norm = {np.linalg.norm(v):.4f}")
    print("[self-test] OK")
