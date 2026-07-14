# src/zero_data_model/capabilities/vision_advanced.py
"""Advanced Computer Vision capabilities for the zero-data cognitive model.

Like the base ``vision`` module, every operator here is a deterministic, rule-based
prior composed with the existing core cognitive modules (math universe, biological
substrate, vision rules). No external CV libraries (no OpenCV / PCL) and no
learned weights are required -- point clouds, video frame analysis and monocular
depth estimation are all derived from geometric / statistical priors.
"""

from __future__ import annotations

import numpy as np

from ..math_universe import MathematicalUniverse
from ..biological import BiologicalSubstrate
from .rules import VisionRules


def _as_2d_float(image: np.ndarray) -> np.ndarray:
    """Coerce any array-like into a 2D float array (matches base vision helper)."""
    img = np.asarray(image, dtype=float)
    if img.ndim == 1:
        img = img.reshape(1, -1)
    elif img.ndim != 2:
        img = img.reshape(-1, img.shape[-1])
    return img


class PointCloudEncoder:
    """Encode a 3D point cloud (Nx3) into a fixed-length L2-normalized vector.

    Features are computed from rule-based geometric statistics: centroid,
    covariance eigenvalues (via ``np.linalg.eigvalsh``), bounding box extents,
    point density and topological features of the flattened coordinates. No
    learned weights are involved.
    """

    def __init__(self, dim: int = 64, math_universe: MathematicalUniverse | None = None):
        self.dim = dim
        self.math_universe = math_universe or MathematicalUniverse(dim=dim)

    def encode(self, points: np.ndarray) -> np.ndarray:
        """Encode ``points`` (Nx3) into a ``dim``-length L2-normalized vector."""
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        n = pts.shape[0]

        if n == 0:
            return np.zeros(self.dim, dtype=float)

        # Centroid (3 features).
        centroid = pts.mean(axis=0)

        # Covariance eigenvalues (3 features); eigvalsh expects a symmetric matrix.
        cov = np.cov(pts.T) if n > 1 else np.zeros((3, 3))
        # Ensure a finite 3-vector of eigenvalues (degenerate inputs may yield NaN).
        try:
            eigvals = np.linalg.eigvalsh(cov) if n > 1 else np.zeros(3)
        except np.linalg.LinAlgError:
            eigvals = np.zeros(3)
        eigvals = np.nan_to_num(np.asarray(eigvals, dtype=float), nan=0.0,
                                posinf=0.0, neginf=0.0)

        # Bounding box extents (3 features).
        bbox_extents = pts.max(axis=0) - pts.min(axis=0)

        # Point density: points per unit bounding-box volume.
        bbox_vol = float(np.prod(bbox_extents + 1e-8))
        density = float(n) / bbox_vol if bbox_vol > 1e-12 else 0.0

        # Topological features of the flattened coords via the math universe.
        flat = pts.flatten()
        topo = np.asarray(
            self.math_universe.topology.topological_features(flat), dtype=float
        )
        topo = np.nan_to_num(topo, nan=0.0, posinf=0.0, neginf=0.0)

        # Combine scalar + vector features into the dim-length vector.
        scalar = np.concatenate([
            centroid.astype(float),
            eigvals.astype(float),
            bbox_extents.astype(float),
            np.array([density, float(n)], dtype=float),
        ])
        combined = np.concatenate([scalar, topo])

        if combined.shape[0] > self.dim:
            combined = combined[: self.dim]
        elif combined.shape[0] < self.dim:
            combined = np.pad(combined, (0, self.dim - combined.shape[0]))

        norm = float(np.linalg.norm(combined))
        if norm > 1e-8:
            combined = combined / norm
        return combined.astype(float)


class VideoFrameAnalyzer:
    """Analyze a sequence of 2D frames with no learned motion model.

    Per-frame motion is the L2 magnitude of the frame-to-frame difference.
    Keyframes are frames whose motion exceeds the mean motion by a rule-based
    threshold. The temporal encoding is produced by seeding the biological
    substrate's cellular automaton with the (binarized) motion series and
    evolving it, then reading out a ``dim``-length vector.
    """

    def __init__(
        self,
        dim: int = 64,
        biological: BiologicalSubstrate | None = None,
        math_universe: MathematicalUniverse | None = None,
    ):
        self.dim = dim
        self.biological = biological or BiologicalSubstrate(dim=dim)
        self.math_universe = math_universe or MathematicalUniverse(dim=dim)
        # Rule-based threshold above the mean for keyframe detection.
        self.keyframe_alpha = 0.5

    def analyze(self, frames) -> dict:
        """Analyze ``frames`` (sequence of 2D images).

        Returns a dict with keys: ``motion_series`` (1D array of length
        ``len(frames) - 1``), ``keyframes`` (list[int] of frame indices where
        motion was anomalously high), ``temporal_encoding`` (``dim`` array),
        ``mean_motion`` (float).
        """
        # Coerce each frame to a 2D float array of common shape.
        imgs = [_as_2d_float(f) for f in frames]
        n = len(imgs)
        if n < 2:
            return {
                "motion_series": np.zeros(max(0, n - 1), dtype=float),
                "keyframes": [],
                "temporal_encoding": np.zeros(self.dim, dtype=float),
                "mean_motion": 0.0,
            }

        # Frame-to-frame motion: mean absolute difference per pixel.
        motion = np.zeros(n - 1, dtype=float)
        for i in range(n - 1):
            a = imgs[i]
            b = imgs[i + 1]
            # Match shapes by truncating to the smaller of the two.
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            diff = b[:h, :w] - a[:h, :w]
            motion[i] = float(np.sqrt(np.mean(diff * diff)))

        mean_motion = float(np.mean(motion))
        std_motion = float(np.std(motion)) + 1e-8

        # Keyframes: motion above mean + alpha * std.
        threshold = mean_motion + self.keyframe_alpha * std_motion
        keyframes = [int(i + 1) for i, m in enumerate(motion) if m > threshold]

        # Temporal encoding: seed the CA with the binarized motion series.
        encoding = self._temporal_encoding(motion)

        return {
            "motion_series": motion,
            "keyframes": keyframes,
            "temporal_encoding": encoding,
            "mean_motion": mean_motion,
        }

    def _temporal_encoding(self, motion: np.ndarray) -> np.ndarray:
        """Encode the motion series into a ``dim``-length vector via the CA.

        The motion series is binarized (above-mean) and seeded into the cellular
        automaton; the resulting evolution is reduced into a ``dim``-vector.
        """
        ca = self.biological.automata
        size = ca.size
        threshold = float(np.mean(motion)) if motion.size > 0 else 0.0
        binary = (motion > threshold).astype(int)
        initial = np.zeros(size, dtype=int)
        m = min(len(binary), size)
        initial[:m] = binary[:m]
        ca.state = initial.copy()
        ca.rule = 30
        n_steps = max(1, min(len(motion), 20))
        history = ca.evolve(n_steps=n_steps)
        flat = history.astype(float).flatten()

        # Reduce the evolution history into a dim-length vector by hashing
        # consecutive chunks of the flattened history into dim bins.
        vec = np.zeros(self.dim, dtype=float)
        for i, v in enumerate(flat):
            vec[i % self.dim] += float(v)
        # Append the per-step motion series summary so the encoding also
        # carries the original motion magnitudes (placed in the trailing bins
        # so it does not collide with the CA-evolution hash).
        if motion.size > 0:
            stats = np.array(
                [float(np.mean(motion)), float(np.std(motion)),
                 float(np.min(motion)), float(np.max(motion))],
                dtype=float,
            )
            base = (self.dim - stats.shape[0]) % self.dim
            for i, s in enumerate(stats):
                vec[(base + i) % self.dim] += s

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec


class DepthEstimator:
    """Monocular depth-cue estimation from a single 2D image (rule-based).

    Combines three deterministic, learning-free depth cues:

      * **Texture gradient** -- local standard deviation varies with row ``y``;
        rows with finer texture are inferred to be farther away.
      * **Linear perspective** -- a Hough-like accumulator on Sobel edges
        detects converging line directions; strong convergence implies depth
        gradient along the convergence axis.
      * **Atmospheric perspective** -- higher rows are hazier / lower-contrast
        and therefore farther (a horizon-style prior).

    The three cues are linearly blended into a single depth map of the same
    shape as the input image.
    """

    def __init__(self, dim: int = 64, math_universe: MathematicalUniverse | None = None):
        self.dim = dim
        self.rules = VisionRules()
        self.math_universe = math_universe or MathematicalUniverse(dim=dim)

    def _texture_gradient(self, img: np.ndarray) -> np.ndarray:
        """Per-row local std as a texture gradient prior; high std = near."""
        h, w = img.shape
        if h < 2 or w < 2:
            return np.zeros_like(img, dtype=float)
        # Local std via a sliding window using the supplied rules.
        # Compute per-row rolling std with a window of 3 (rule-based prior).
        window = 3
        row_stds = np.zeros(h, dtype=float)
        for i in range(h):
            row = img[i]
            if w < window:
                row_stds[i] = float(np.std(row)) if w > 1 else 0.0
            else:
                # Rolling std with a sliding window of `window` pixels.
                csum = np.cumsum(np.insert(row, 0, 0.0))
                csum_sq = np.cumsum(np.insert(row ** 2, 0, 0.0))
                n_w = w - window + 1
                sums = csum[window:] - csum[:-window]
                sums_sq = csum_sq[window:] - csum_sq[:-window]
                means = sums / window
                var = np.maximum(sums_sq / window - means ** 2, 0.0)
                row_stds[i] = float(np.mean(np.sqrt(var)))
        # Map row stds into [0, 1]; high std => near (large depth value).
        rs = row_stds - row_stds.min()
        rs_max = rs.max()
        rs_norm = rs / (rs_max + 1e-8) if rs_max > 1e-8 else np.zeros_like(rs)
        # Broadcast per-row depth across columns.
        return np.tile(rs_norm[:, None], (1, w))

    def _atmospheric_gradient(self, img: np.ndarray) -> np.ndarray:
        """Atmospheric perspective: higher rows (top) are hazier => farther.

        Per-row contrast is computed; lower-contrast rows receive higher depth.
        Returns a depth map in [0, 1].
        """
        h, w = img.shape
        if h < 2:
            return np.zeros_like(img, dtype=float)
        row_contrast = np.zeros(h, dtype=float)
        for i in range(h):
            row = img[i]
            row_contrast[i] = float(np.std(row)) if w > 1 else 0.0
        # Higher contrast => nearer (low depth). Invert and normalize.
        rc = row_contrast.max() - row_contrast
        rc_max = rc.max()
        rc_norm = rc / (rc_max + 1e-8) if rc_max > 1e-8 else np.zeros_like(rc)
        return np.tile(rc_norm[:, None], (1, w))

    def _linear_perspective(self, img: np.ndarray) -> np.ndarray:
        """Hough-like accumulator on Sobel edges to detect converging lines.

        Edge orientations are accumulated into direction bins; the magnitude of
        the dominant orientation gives a per-row perspective strength prior.
        Returns a depth map in [0, 1].
        """
        h, w = img.shape
        gx = self.rules.convolve(img, self.rules.sobel_x)
        gy = self.rules.convolve(img, self.rules.sobel_y)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        # Avoid divide-by-zero in arctan2.
        eps = 1e-8
        orient = np.arctan2(gy, gx + eps)
        # Quantize orientation into 18 bins of 20 degrees over [-pi, pi).
        n_bins = 18
        bin_idx = ((orient + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
        # Per-row dominant-bin strength as perspective prior.
        per_row = np.zeros(h, dtype=float)
        for i in range(h):
            row_mag = mag[i]
            row_bins = bin_idx[i]
            # Strong edges only.
            mask = row_mag > (float(np.mean(row_mag)) + float(np.std(row_mag)))
            if not np.any(mask):
                per_row[i] = 0.0
                continue
            counts = np.bincount(row_bins[mask], minlength=n_bins)
            per_row[i] = float(counts.max()) / float(max(1, mask.sum()))
        # Stronger perspective toward the top of the image (typical horizon
        # convergence): combine per-row strength with row position prior.
        row_pos = np.linspace(0.0, 1.0, h)
        combined = 0.5 * per_row + 0.5 * (1.0 - row_pos)
        c_max = combined.max()
        out = combined / (c_max + 1e-8) if c_max > 1e-8 else np.zeros_like(combined)
        return np.tile(out[:, None], (1, w))

    def estimate(self, image: np.ndarray) -> dict:
        """Estimate a depth map from a single 2D image (rule-based cues).

        Returns ``{'depth_map': ndarray same shape as image, 'depth_stats':
        dict with mean/std/min/max}``.
        """
        img = _as_2d_float(image)
        texture = self._texture_gradient(img)
        atmospheric = self._atmospheric_gradient(img)
        perspective = self._linear_perspective(img)

        # Blend the three cues (equal weights as a flat prior).
        depth = (texture + atmospheric + perspective) / 3.0
        # Normalize to [0, 1] for stability.
        d_min = float(depth.min())
        d_max = float(depth.max())
        if d_max - d_min > 1e-8:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return {
            "depth_map": depth,
            "depth_stats": {
                "mean": float(np.mean(depth)),
                "std": float(np.std(depth)),
                "min": float(np.min(depth)),
                "max": float(np.max(depth)),
            },
        }
