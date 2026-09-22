# experiments/code_encoder.py
"""Code encoder for the omnimodal phase (Phase K, Task 1.3).

A ZERO-PRETRAIN code encoder. It treats source code as a special form
of text but augments the textual bag-of-characters embedding with
SYNTAX-AWARE structural features that are cheap to compute without an
AST parser:

  Structural features (``n_struct = 16``):
    1. mean_indent_depth        : average leading-whitespace width
                                  (normalised by max line length).
    2. max_indent_depth         : peak leading-whitespace width.
    3. n_def                    : count of ``def`` / ``function`` / ``fn``
                                  declarations (per non-empty line).
    4. n_class                  : count of ``class`` / ``struct`` decls.
    5. n_import                 : count of ``import`` / ``from`` / ``#include``.
    6. n_return                 : count of ``return`` keywords.
    7. n_loop                   : count of ``for`` / ``while`` keywords.
    8. n_branch                 : count of ``if`` / ``else`` / ``elif`` /
                                  ``switch`` / ``case`` keywords.
    9. n_assign                 : count of ``=`` (excl. ``==`` / ``>=``).
   10. n_call                   : count of function-call parens ``(``.
   11. n_comment_lines          : count of lines starting with ``#`` or ``//``.
   12. n_blank_lines            : count of blank / whitespace-only lines.
   13. mean_line_length         : average non-empty line length (normalised).
   14. n_distinct_identifiers   : rough count of identifier-like tokens.
   15. brace_balance            : ``{`` count − ``}`` count (sanity).
   16. paren_balance            : ``(`` count − ``)`` count (sanity).

These features are normalised (z-score-style with a fixed prior mean
and std) and concatenated with the text encoder's bag-of-characters
projection. The combined vector is then projected through a FIXED
random matrix to ``output_dim`` to keep the modality interface uniform
with the image and audio encoders.

A ``CodeStream`` class loads ``.py`` (or other code) files from a
directory, optionally splitting them into per-function / per-class
chunks, and yields (encoded_vector, source_path) pairs.

Determinism: with a fixed ``seed``, the projection matrices are
identical across runs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT_DIM = 32
DEFAULT_PROJ_DIM = 64
DEFAULT_N_STRUCT = 16
DEFAULT_BLOCK_CHARS = 1024
SUPPORTED_EXTS = {".py", ".js", ".ts", ".c", ".h", ".cpp", ".hpp", ".cc",
                  ".java", ".rs", ".go", ".rb", ".sh", ".ml"}

# Pre-compiled regexes for syntactic feature extraction.
# Note: these are intentionally simple lexical patterns, NOT a parser.
# They cover Python / C-family / JS / Rust reasonably well and tolerate
# malformed / partial chunks.
KW_DEF    = re.compile(r"\b(def|function|fn|func|sub)\s+\w+")
KW_CLASS  = re.compile(r"\b(class|struct|interface|trait|object)\s+\w+")
KW_IMPORT = re.compile(r"^\s*(import|from|include|require|use)\b", re.MULTILINE)
KW_RETURN = re.compile(r"\breturn\b")
KW_LOOP   = re.compile(r"\b(for|while|foreach|do)\b")
KW_BRANCH = re.compile(r"\b(if|else|elif|switch|case|when|match)\b")
ASSIGN_RE = re.compile(r"(?<![=!<>])[=](?![=])")
CALL_RE   = re.compile(r"\w+\s*\(")
IDENT_RE  = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
LCOMMENT_RE = re.compile(r"^\s*(#|//)", re.MULTILINE)

# Prior mean / std for the 16 structural features. Used to z-score the
# raw counts so a single chunk's features land in roughly the same
# dynamic range as the text encoder's output. These priors are
# heuristic (computed from typical small Python snippets) and are NOT
# learned — they simply normalise magnitudes for the downstream random
# projection.
_STRUCT_PRIOR_MEAN = np.array([
    0.10,   # mean_indent_depth  (fraction of line width)
    0.40,   # max_indent_depth   (fraction)
    0.05,   # n_def per line
    0.01,   # n_class per line
    0.03,   # n_import per line
    0.05,   # n_return per line
    0.10,   # n_loop per line
    0.20,   # n_branch per line
    0.30,   # n_assign per line
    0.50,   # n_call per line
    0.10,   # n_comment_lines per line
    0.20,   # n_blank_lines per line
    0.40,   # mean_line_length (fraction of 200)
    0.50,   # n_distinct_identifiers per line
    0.0,    # brace_balance (often 0 in valid code)
    0.0,    # paren_balance (often 0 in valid code)
], dtype=np.float64)
_STRUCT_PRIOR_STD = np.array([
    0.10, 0.20, 0.05, 0.03, 0.03, 0.05, 0.05, 0.10,
    0.20, 0.30, 0.10, 0.20, 0.20, 0.50, 5.0, 5.0,
], dtype=np.float64)


# ------------------------------------------------------------------ #
# CodeEncoder
# ------------------------------------------------------------------ #
class CodeEncoder:
    """Code encoder: text-bag projection + structural features.

    Parameters
    ----------
    output_dim : int
        Final output dimensionality (matches the model's dim).
    proj_dim : int
        Internal width of the text-only projection. The structural
        features are concatenated AFTER this projection, so the final
        pre-projection vector is ``proj_dim + n_struct`` wide.
    n_struct : int
        Number of structural features. Defaults to 16.
    seed : int
        Seed for all random matrices.
    text_encoder : object, optional
        A pre-built ``TextEncoder`` instance to reuse. If None, a new
        one is created with the same seed.
    """

    def __init__(
        self,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        proj_dim: int = DEFAULT_PROJ_DIM,
        n_struct: int = DEFAULT_N_STRUCT,
        seed: int = 42,
        text_encoder=None,
    ):
        self.output_dim = int(output_dim)
        self.proj_dim = int(proj_dim)
        self.n_struct = int(n_struct)
        self.seed = int(seed)
        # Reuse a TextEncoder if provided (lets callers share the
        # character table across encoders for cross-modal alignment).
        if text_encoder is not None:
            self.text_encoder = text_encoder
            text_proj_out = text_encoder.output_dim
        else:
            # Import locally to avoid hard dependency at module import.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from text_encoder import TextEncoder
            self.text_encoder = TextEncoder(
                block_size=DEFAULT_BLOCK_CHARS,
                output_dim=self.proj_dim,
                seed=seed,
            )
            text_proj_out = self.proj_dim
        # Random projection from (text_proj_out + n_struct) → output_dim.
        rng = np.random.default_rng(seed)
        concat_dim = text_proj_out + self.n_struct
        self._final_proj = rng.standard_normal((self.output_dim, concat_dim)) / np.sqrt(concat_dim)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def encode(self, code: str) -> np.ndarray:
        """Encode a code string -> (output_dim,) vector."""
        if not code:
            return np.zeros(self.output_dim, dtype=np.float64)
        # Text projection: bag-of-characters via the shared TextEncoder.
        text_vec = self.text_encoder.encode(code)  # (proj_dim,)
        # Structural features.
        struct_vec = self._structural_features(code)  # (n_struct,)
        # Concatenate + project.
        concat = np.concatenate([text_vec, struct_vec])
        out = self._final_proj @ concat
        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    # ------------------------------------------------------------------ #
    # Structural features
    # ------------------------------------------------------------------ #
    def _structural_features(self, code: str) -> np.ndarray:
        """Compute the 16 structural features, normalised by the prior."""
        lines = code.splitlines()
        n_lines = max(1, len(lines))
        non_empty = [ln for ln in lines if ln.strip()]
        n_non_empty = max(1, len(non_empty))
        # Indent depth.
        indents = [len(ln) - len(ln.lstrip(" \t")) for ln in non_empty]
        mean_indent = float(np.mean(indents)) / 80.0 if indents else 0.0
        max_indent = float(max(indents)) / 80.0 if indents else 0.0
        # Keyword counts.
        n_def = len(KW_DEF.findall(code))
        n_class = len(KW_CLASS.findall(code))
        n_import = len(KW_IMPORT.findall(code))
        n_return = len(KW_RETURN.findall(code))
        n_loop = len(KW_LOOP.findall(code))
        n_branch = len(KW_BRANCH.findall(code))
        n_assign = len(ASSIGN_RE.findall(code))
        n_call = len(CALL_RE.findall(code))
        n_comment = len(LCOMMENT_RE.findall(code))
        n_blank = sum(1 for ln in lines if not ln.strip())
        # Line length.
        mean_line_len = float(np.mean([len(ln) for ln in non_empty])) if non_empty else 0.0
        mean_line_len_norm = min(mean_line_len / 200.0, 1.0)
        # Identifiers.
        idents = IDENT_RE.findall(code)
        n_distinct = len(set(idents))
        # Brace / paren balance.
        n_open_brace = code.count("{")
        n_close_brace = code.count("}")
        n_open_paren = code.count("(")
        n_close_paren = code.count(")")
        brace_bal = n_open_brace - n_close_brace
        paren_bal = n_open_paren - n_close_paren
        # Per non-empty line normalisation (helps chunk-size invariance).
        feats = np.array([
            mean_indent,
            max_indent,
            n_def / n_non_empty,
            n_class / n_non_empty,
            n_import / n_non_empty,
            n_return / n_non_empty,
            n_loop / n_non_empty,
            n_branch / n_non_empty,
            n_assign / n_non_empty,
            n_call / n_non_empty,
            n_comment / n_non_empty,
            n_blank / n_lines,
            mean_line_len_norm,
            n_distinct / n_non_empty,
            float(brace_bal),
            float(paren_bal),
        ], dtype=np.float64)
        # z-score with the prior.
        feats = (feats - _STRUCT_PRIOR_MEAN) / np.maximum(_STRUCT_PRIOR_STD, 1e-6)
        # Clip extreme outliers (e.g. 100-line paren_balance from broken file).
        feats = np.clip(feats, -10.0, 10.0)
        return feats


# ------------------------------------------------------------------ #
# Synthetic code snippets (for tests + smoke runs)
# ------------------------------------------------------------------ #
_SYNTHETIC_SNIPPETS = {
    "fib": '''def fib(n):
    """Return the n-th Fibonacci number."""
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

# Example
print(fib(10))
''',
    "class_stack": '''class Stack:
    def __init__(self):
        self.items = []

    def push(self, x):
        self.items.append(x)

    def pop(self):
        if not self.items:
            raise IndexError("empty")
        return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0
''',
    "import_heavy": '''import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter

def load_config(path):
    with open(path) as f:
        return json.load(f)

def main():
    cfg = load_config("config.json")
    print(cfg)

if __name__ == "__main__":
    main()
''',
    "loop_algo": '''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

result = bubble_sort([5, 3, 8, 1, 9, 2])
print(result)
''',
}


def make_synthetic_code_snippet(name: str) -> str:
    """Return one of the bundled synthetic code snippets."""
    if name not in _SYNTHETIC_SNIPPETS:
        raise KeyError(f"unknown snippet {name!r}; choose from {list(_SYNTHETIC_SNIPPETS)}")
    return _SYNTHETIC_SNIPPETS[name]


def make_synthetic_code_dir(
    output_dir: Path,
    n_per_snippet: int = 2,
    seed: int = 42,
) -> list[Path]:
    """Write synthetic code files to ``output_dir``.

    Each bundled snippet is written ``n_per_snippet`` times with small
    cosmetic variations (extra comment) so the directory is non-trivial.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, snippet in _SYNTHETIC_SNIPPETS.items():
        for i in range(n_per_snippet):
            text = f"# variant {i}\n" + snippet
            path = output_dir / f"{name}_{i:02d}.py"
            path.write_text(text, encoding="utf-8")
            paths.append(path)
    return paths


# ------------------------------------------------------------------ #
# CodeStream
# ------------------------------------------------------------------ #
class CodeStream:
    """Stream code files from a directory, encoding each on demand.

    Optionally splits files into per-def chunks before encoding, so a
    single long file contributes multiple observations.
    """

    def __init__(
        self,
        code_dir: str | Path,
        encoder: CodeEncoder,
        chunk_mode: str = "whole",
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.code_dir = Path(code_dir)
        self.encoder = encoder
        if chunk_mode not in ("whole", "per_def"):
            raise ValueError(f"unknown chunk_mode {chunk_mode!r}")
        self.chunk_mode = chunk_mode
        self._paths = sorted(
            p for p in self.code_dir.iterdir()
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
        """Return the next (encoded_vector, source_path) pair. Wraps."""
        if not self._paths:
            raise RuntimeError("no code files in directory")
        path = self._paths[self._idx % len(self._paths)]
        self._idx += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        if self.chunk_mode == "per_def":
            chunks = self._split_per_def(text)
            if not chunks:
                chunks = [text]
            # Use the chunk at the current cycle index modulo chunks.
            chunk = chunks[self._idx % len(chunks)]
        else:
            chunk = text
        return self.encoder.encode(chunk), str(path)

    def random(self) -> tuple[np.ndarray, str]:
        if not self._paths:
            raise RuntimeError("no code files in directory")
        rng = np.random.default_rng(None)
        path = self._paths[int(rng.integers(0, len(self._paths)))]
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.encoder.encode(text), str(path)

    @staticmethod
    def _split_per_def(text: str) -> list[str]:
        """Split a Python source file into per-def chunks.

        Each chunk starts at a ``def`` / ``class`` line and ends just
        before the next one (or end of file). Lines before the first
        def form a "preamble" chunk.
        """
        lines = text.splitlines(keepends=True)
        chunks: list[str] = []
        current: list[str] = []
        for ln in lines:
            if re.match(r"^(def|class)\s+\w+", ln):
                if current:
                    chunks.append("".join(current))
                current = [ln]
            else:
                current.append(ln)
        if current:
            chunks.append("".join(current))
        return chunks


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import tempfile

    enc = CodeEncoder(output_dim=32, seed=42)
    print(f"[self-test] encoder built (final_proj={enc._final_proj.shape})")

    # Encode two structurally different snippets.
    fib = make_synthetic_code_snippet("fib")
    stack = make_synthetic_code_snippet("class_stack")
    vf = enc.encode(fib)
    vs = enc.encode(stack)
    print(f"  encoded fib:   shape={vf.shape}  norm={np.linalg.norm(vf):.4f}")
    print(f"  encoded stack: shape={vs.shape}  norm={np.linalg.norm(vs):.4f}")
    cos = float(np.dot(vf, vs) / (np.linalg.norm(vf) * np.linalg.norm(vs) + 1e-12))
    print(f"  cos(fib, stack)={cos:.4f}")

    # Structural features: import_heavy should differ from fib.
    imp = make_synthetic_code_snippet("import_heavy")
    vi = enc.encode(imp)
    cos_i = float(np.dot(vf, vi) / (np.linalg.norm(vf) * np.linalg.norm(vi) + 1e-12))
    print(f"  cos(fib, import_heavy)={cos_i:.4f}")

    # Determinism check.
    vf_again = enc.encode(fib)
    assert np.allclose(vf, vf_again), "encoder should be deterministic given seed"
    print("[self-test] determinism: OK")

    # Synthetic dir + CodeStream.
    tmp = Path(tempfile.mkdtemp())
    paths = make_synthetic_code_dir(tmp, n_per_snippet=2, seed=42)
    print(f"[self-test] wrote {len(paths)} synthetic code files to {tmp}")
    stream = CodeStream(tmp, enc, seed=42)
    print(f"  stream len = {len(stream)}")
    v, p = stream.step()
    print(f"  first: {Path(p).name}  vec norm = {np.linalg.norm(v):.4f}")
    print("[self-test] OK")
