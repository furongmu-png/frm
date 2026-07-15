# src/zero_data_model/persistence.py
"""Model persistence for the zero-data cognitive system.

Saves/loads the full model state to a directory on disk using numpy
``.npz`` for arrays and JSON for configuration metadata. Pickle is
intentionally NOT used so the on-disk format is portable, auditable and
safe to load from untrusted sources.

Security: every save/load path is sandboxed under a module-level
``_PERSISTENCE_ROOT`` directory. Absolute paths, ``..`` traversal and
symlinks pointing outside the root are rejected with ``ValueError``.
Saves are atomic: data is written to a sibling temp directory first,
then renamed into place so a crash never leaves a partial snapshot.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any

import numpy as np

from .model import ZeroDataModel

# --------------------------------------------------------------------------- #
# Persistence root sandbox
# --------------------------------------------------------------------------- #

def _default_persistence_root() -> str:
    """Pick a sensible default root: /app/data in containers, else cwd/data."""
    container_root = "/app/data"
    if os.path.isdir("/app") or os.path.isdir(container_root):
        return container_root
    return os.path.join(os.getcwd(), "data")


# Module-level root. Override at runtime via set_persistence_root() (tests).
_PERSISTENCE_ROOT: str = _default_persistence_root()


def set_persistence_root(path: str) -> None:
    """Override the persistence root at runtime (used by tests).

    The directory is created if missing so callers can point at a fresh
    tmp_path without an extra mkdir.
    """
    global _PERSISTENCE_ROOT
    resolved = os.path.realpath(os.path.abspath(path))
    os.makedirs(resolved, exist_ok=True)
    _PERSISTENCE_ROOT = resolved


def get_persistence_root() -> str:
    """Return the current persistence root (resolved absolute path)."""
    return os.path.realpath(os.path.abspath(_PERSISTENCE_ROOT))


def _validate_path(path: str) -> str:
    """Resolve ``path`` against the persistence root and enforce sandboxing.

    Rejects:
      * absolute paths (caller must supply a relative name)
      * ``..`` segments that escape the root after resolution
      * any component that is a symlink resolving outside the root

    Returns the resolved absolute path inside the root. Raises
    ``ValueError("path outside persistence root")`` on violation.
    """
    if path is None or path == "":
        raise ValueError("path outside persistence root")

    # Reject absolute paths: callers must pass a relative name.
    if os.path.isabs(path):
        raise ValueError("path outside persistence root")

    # Reject explicit ``..`` segments early -- even if realpath would
    # otherwise clamp them, the intent is clearly hostile.
    parts = path.replace("\\", "/").split("/")
    if any(part == ".." for part in parts):
        raise ValueError("path outside persistence root")

    root = get_persistence_root()
    # Join relatively; resolve symlinks where they exist on disk. We use
    # os.path.abspath first (no symlink resolution) then realpath for the
    # final check so a missing target still validates against the root
    # prefix lexically.
    candidate_lexical = os.path.abspath(os.path.join(root, path))

    # If the target already exists, follow its real path to catch symlinks
    # that point outside the root. If it doesn't exist yet (save case),
    # validate the parent that does exist.
    if os.path.islink(candidate_lexical) or os.path.exists(candidate_lexical):
        candidate_real = os.path.realpath(candidate_lexical)
    else:
        # Walk up to the deepest existing ancestor and realpath that, then
        # re-append the missing tail components.
        ancestor = candidate_lexical
        tail: list[str] = []
        while not os.path.exists(ancestor) and ancestor != root:
            ancestor, head = os.path.split(ancestor)
            tail.append(head)
        ancestor_real = os.path.realpath(ancestor)
        candidate_real = os.path.join(ancestor_real, *reversed(tail))

    # Use os.path.commonpath for a robust prefix check (handles trailing
    # slashes and component boundaries correctly).
    if os.path.commonpath([root, candidate_real]) != root:
        raise ValueError("path outside persistence root")
    if os.path.commonpath([root, candidate_lexical]) != root:
        raise ValueError("path outside persistence root")

    return candidate_real


class ModelSerializer:
    """Serialize a :class:`ZeroDataModel` to a directory of npz + json.

    Layout written under ``path``:

    * ``arrays.npz``  -- every numpy weight/state array, keyed by a
      dotted-style identifier flattened to underscores (np.savez requires
      valid Python identifiers as keys).
    * ``config.json`` -- scalar configuration: ``dim``, ``cycle_count``,
      ``quantum_backend_name`` plus the counts needed to walk the npz on
      load.
    """

    # ------------------------------------------------------------------ #
    # save
    # ------------------------------------------------------------------ #
    @staticmethod
    def save(model: ZeroDataModel, path: str) -> None:
        """Save model state to a directory: arrays.npz + config.json.

        ``path`` must be a *relative* name under the persistence root.
        The write is atomic: data lands in a sibling temp dir first,
        then is renamed into place so a crash never produces a partial
        snapshot that load() would later misread.
        """
        target = _validate_path(path)

        # Stage into a temp directory beside the persistence root (same
        # filesystem so os.replace is atomic). tempfile.mkdtemp gives us
        # an exclusive, predictably-named scratch dir.
        root = get_persistence_root()
        staging = tempfile.mkdtemp(prefix=".save-", dir=root)
        try:
            ModelSerializer._write_snapshot(model, staging)

            # Atomic replace of the target directory. If target exists,
            # rename it aside first so os.replace works on a dir.
            backup: str | None = None
            if os.path.exists(target):
                backup = target + ".bak-" + os.path.basename(staging)
                os.replace(target, backup)
            try:
                os.replace(staging, target)
            except OSError:
                # Roll back: restore the backup if rename failed.
                if backup is not None and os.path.exists(backup):
                    os.replace(backup, target)
                raise
            # Success: remove the old backup.
            if backup is not None and os.path.exists(backup):
                shutil.rmtree(backup, ignore_errors=True)
        except Exception:
            # Make sure the staging dir never lingers on failure.
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _write_snapshot(model: ZeroDataModel, path: str) -> None:
        """Write arrays.npz + config.json into ``path`` (already created)."""
        arrays: dict[str, np.ndarray] = {}

        # Consciousness core: per-layer weights and biases.
        for i, layer in enumerate(model.consciousness.layers):
            arrays[f"consciousness_layers_{i}_weights"] = np.asarray(layer.weights)
            arrays[f"consciousness_layers_{i}_bias"] = np.asarray(layer.bias)

        # Active inference: transition / emission / belief_state.
        gm = model.active_inference.generative_model
        arrays["active_inference_transition"] = np.asarray(gm.transition)
        arrays["active_inference_emission"] = np.asarray(gm.emission)
        arrays["active_inference_belief_state"] = np.asarray(gm.belief_state)

        # Category engine: topos classifier + every functor's transforms.
        arrays["category_engine_topos_classifier"] = np.asarray(
            model.category_engine.topos.classifier
        )
        functor_morphism_counts: list[int] = []
        for j, functor in enumerate(model.category_engine.functors):
            # Morphism keys are tuples (unhashable as npz keys); index them.
            morph_items = list(functor.morphism_map.items())
            functor_morphism_counts.append(len(morph_items))
            for k, (_key, transform) in enumerate(morph_items):
                arrays[f"category_engine_functor_{j}_transform_{k}"] = np.asarray(transform)

        # Quantum hybrid: classical_weights + circuit params/entangling +
        # annealer cost_matrix.
        arrays["quantum_hybrid_classical_weights"] = np.asarray(
            model.quantum_hybrid.classical_weights
        )
        arrays["quantum_hybrid_circuit_params"] = np.asarray(
            model.quantum_hybrid.quantum_circuit.params
        )
        arrays["quantum_hybrid_circuit_entangling"] = np.asarray(
            model.quantum_hybrid.quantum_circuit.entangling
        )
        arrays["quantum_hybrid_annealer_cost_matrix"] = np.asarray(
            model.quantum_hybrid.annealer.cost_matrix
        )

        # Biological substrate: morphogenetic grid/morphogens + automata state.
        arrays["biological_morphogenetic_grid"] = np.asarray(
            model.biological.morphogenetic.grid
        )
        for j, morphogen in enumerate(model.biological.morphogenetic.morphogens):
            arrays[f"biological_morphogenetic_morphogens_{j}"] = np.asarray(morphogen)
        arrays["biological_automata_state"] = np.asarray(model.biological.automata.state)

        # Math universe: fractal transforms (scale + offset pairs).
        for j, (scale, offset) in enumerate(model.math_universe.fractal.transforms):
            arrays[f"math_universe_fractal_transforms_{j}_scale"] = np.asarray(scale)
            arrays[f"math_universe_fractal_transforms_{j}_offset"] = np.asarray(offset)

        np.savez(os.path.join(path, "arrays.npz"), **arrays)

        config: dict[str, Any] = {
            "dim": int(model.dim),
            "cycle_count": int(model.cycle_count),
            "quantum_backend_name": str(model.quantum_hybrid.quantum_backend_name),
            "n_consciousness_layers": len(model.consciousness.layers),
            "n_functors": len(model.category_engine.functors),
            "functor_morphism_counts": functor_morphism_counts,
            "n_morphogens": len(model.biological.morphogenetic.morphogens),
            "n_fractal_transforms": len(model.math_universe.fractal.transforms),
        }
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

    # ------------------------------------------------------------------ #
    # load
    # ------------------------------------------------------------------ #
    @staticmethod
    def load(path: str) -> ZeroDataModel:
        """Reconstruct a :class:`ZeroDataModel` from a saved directory.

        ``path`` must be a *relative* name under the persistence root.
        """
        target = _validate_path(path)
        if not os.path.isdir(target):
            raise FileNotFoundError(f"Model directory not found: {path}")

        npz_path = os.path.join(target, "arrays.npz")
        config_path = os.path.join(target, "config.json")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Missing arrays.npz in: {path}")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Missing config.json in: {path}")

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Build a fresh model of the same dim, then overwrite every saved
        # numpy attribute from the npz. Fresh construction handles wiring up
        # the quantum backend (auto-detect on this machine), capability
        # modules, parallel executor, etc.
        model = ZeroDataModel(dim=config["dim"])
        model.cycle_count = int(config.get("cycle_count", 0))

        data = np.load(npz_path)

        # Consciousness core: per-layer weights and biases (overwrites each
        # layer in-place so layer objects keep their identity).
        for i, layer in enumerate(model.consciousness.layers):
            if i < config["n_consciousness_layers"]:
                layer.weights = np.asarray(data[f"consciousness_layers_{i}_weights"])
                layer.bias = np.asarray(data[f"consciousness_layers_{i}_bias"])

        # Active inference generative model arrays.
        gm = model.active_inference.generative_model
        gm.transition = np.asarray(data["active_inference_transition"])
        gm.emission = np.asarray(data["active_inference_emission"])
        gm.belief_state = np.asarray(data["active_inference_belief_state"])

        # Category engine: topos classifier + functor transforms.
        model.category_engine.topos.classifier = np.asarray(
            data["category_engine_topos_classifier"]
        )
        morph_counts = config.get("functor_morphism_counts", [])
        for j, functor in enumerate(model.category_engine.functors):
            if j >= config["n_functors"]:
                break
            n_morphs = morph_counts[j] if j < len(morph_counts) else 0
            morph_items = list(functor.morphism_map.items())
            for k, (key, _value) in enumerate(morph_items):
                if k >= n_morphs:
                    break
                functor.morphism_map[key] = np.asarray(
                    data[f"category_engine_functor_{j}_transform_{k}"]
                )

        # Quantum hybrid arrays.
        model.quantum_hybrid.classical_weights = np.asarray(
            data["quantum_hybrid_classical_weights"]
        )
        model.quantum_hybrid.quantum_circuit.params = np.asarray(
            data["quantum_hybrid_circuit_params"]
        )
        model.quantum_hybrid.quantum_circuit.entangling = np.asarray(
            data["quantum_hybrid_circuit_entangling"]
        )
        model.quantum_hybrid.annealer.cost_matrix = np.asarray(
            data["quantum_hybrid_annealer_cost_matrix"]
        )

        # Biological substrate arrays.
        model.biological.morphogenetic.grid = np.asarray(
            data["biological_morphogenetic_grid"]
        )
        # Replace the morphogen list to preserve length even if the source
        # machine had a different signal count (default is 3).
        loaded_morphogens: list[np.ndarray] = []
        for j in range(config["n_morphogens"]):
            loaded_morphogens.append(
                np.asarray(data[f"biological_morphogenetic_morphogens_{j}"])
            )
        model.biological.morphogenetic.morphogens = loaded_morphogens
        model.biological.automata.state = np.asarray(data["biological_automata_state"])

        # Math universe fractal transforms (scale + offset pairs).
        loaded_transforms: list[tuple[np.ndarray, np.ndarray]] = []
        for j in range(config["n_fractal_transforms"]):
            scale = np.asarray(data[f"math_universe_fractal_transforms_{j}_scale"])
            offset = np.asarray(data[f"math_universe_fractal_transforms_{j}_offset"])
            loaded_transforms.append((scale, offset))
        model.math_universe.fractal.transforms = loaded_transforms

        return model


__all__ = [
    "ModelSerializer",
    "set_persistence_root",
    "get_persistence_root",
]
