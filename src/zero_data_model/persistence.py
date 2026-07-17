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

import io
import json
import os
import shutil
import tempfile
import threading
from typing import Any

import numpy as np

from .model import ZeroDataModel

# Round-5 audit PERSIST5-5: cap config.json size to prevent OOM via a hostile
# multi-MB JSON that would exhaust the parser before any validation runs.
_MAX_CONFIG_BYTES: int = 1 << 20  # 1 MiB — a legitimate config is ~1 KiB.

# Round-5 audit PERSIST5-2: cap per-functor morphism count to prevent OOM via
# inflated counts that each trigger an npz key lookup + array allocation.
_MAX_MORPH_COUNT: int = 4096

# Round-6 audit CONCUR6-2 / CONCUR6-3: module-level lock serialising the
# stage→backup→replace→cleanup sequence in ``save`` and the open-config +
# open-npz sequence in ``load``. Without this, two concurrent saves to the
# same ``path`` (e.g. two ``ZeroDataModel`` instances, or direct callers
# bypassing ``api.py``'s ``model._lock``) race: Save A renames target to
# backup, Save B sees no target and stages fresh, Save A overwrites B's
# snapshot, then ``rmtree(backup)`` deletes the original. Save B's snapshot
# is silently lost. The lock also prevents a concurrent ``load`` from
# reading config.json + arrays.npz from different snapshots (torn read).
_PERSISTENCE_LOCK = threading.Lock()

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

        Round-6 audit CONCUR6-2: the stage→backup→replace→cleanup sequence
        is serialised by ``_PERSISTENCE_LOCK`` so two concurrent saves to
        the same ``path`` cannot lose one snapshot via the rollback path.

        Raises:
            ValueError: if ``path`` escapes the persistence root or the
                config/npz content fails validation.
            OSError: on filesystem errors (disk full, permissions, ...).
        """
        target = _validate_path(path)

        # Stage into a temp directory beside the persistence root (same
        # filesystem so os.replace is atomic). tempfile.mkdtemp gives us
        # an exclusive, predictably-named scratch dir.
        root = get_persistence_root()
        # Write the snapshot OUTSIDE the lock — npz serialisation is the
        # expensive part and does not touch the shared target. Only the
        # stage→backup→replace→cleanup swap needs to be serialised.
        staging = tempfile.mkdtemp(prefix=".save-", dir=root)
        try:
            ModelSerializer._write_snapshot(model, staging)
            with _PERSISTENCE_LOCK:
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

        Round-6 audit CONCUR6-3: the open-config + open-npz sequence is
        serialised by ``_PERSISTENCE_LOCK`` (shared with ``save``) so a
        concurrent ``save``'s ``os.replace(staging, target)`` cannot land
        between the config read and the npz read, which would otherwise
        produce a torn snapshot (config from old, arrays from new).

        Raises:
            FileNotFoundError: if ``path`` or its arrays.npz / config.json
                is missing.
            ValueError: if the config or npz content fails validation
                (sandbox escape, oversized config, missing/extra keys,
                non-numeric dtype, shape mismatch, ...).
            OSError: on filesystem errors.
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

        # Round-5 audit PERSIST5-5: reject an oversized config.json before
        # parsing. A hostile snapshot could stuff a multi-MB JSON that OOMs
        # the parser before any validation runs.
        config_size = os.path.getsize(config_path)
        if config_size > _MAX_CONFIG_BYTES:
            raise ValueError(
                f"config.json too large ({config_size} bytes > "
                f"{_MAX_CONFIG_BYTES}); refusing to load a potentially "
                "hostile snapshot"
            )

        # Round-6 audit CONCUR6-3 + Round-7 audit CONCUR7-1: hold the
        # persistence lock across BOTH the config.json read AND the npz
        # file read so a concurrent ``save``'s ``os.replace(staging, target)``
        # cannot swap the directory between the two reads (torn read:
        # old config + new arrays, or vice versa). The Round-6 fix only
        # locked the config read — the npz read ran unlocked ~130 lines
        # later, so a save could land in that gap. Now we read the npz
        # BYTES under the lock (cheap: sequential disk read) and defer
        # the heavy ``np.load`` + validation + assignment to the unlocked
        # section below, preserving load concurrency.
        with _PERSISTENCE_LOCK:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            with open(npz_path, "rb") as f:
                npz_bytes = f.read()

        # Round-3 audit B-batch: validate the scalar config before acting on
        # any of it. A hostile or corrupt config.json could otherwise request
        # a huge ``dim`` (DoS via OOM) or supply counts that silently mismatch
        # the freshly-constructed model's architecture (silent state
        # corruption: extra saved layers/functors would be dropped, and
        # missing ones would leave fresh-random arrays in place of saved data).
        if "dim" not in config:
            raise ValueError("config.json missing required key: 'dim'")
        # Round-4 audit PERSIST-5: reject bool/float explicitly. ``int(True)``
        # is 1 and ``int(8.5)`` is 8, so the previous ``int(config["dim"])``
        # conversion silently accepted these and then ``ZeroDataModel(dim=1)``
        # constructed a 1-dim model (mismatching the npz's real dim-8 arrays).
        dim_raw = config["dim"]
        if isinstance(dim_raw, bool) or not isinstance(dim_raw, int):
            raise ValueError(
                f"config['dim'] must be a plain int, got "
                f"{type(dim_raw).__name__}: {dim_raw!r}"
            )
        dim = dim_raw
        if dim < 1 or dim > 4096:
            raise ValueError(
                f"refusing to load model with dim={dim!r}: must be in [1, 4096]"
            )

        # Build a fresh model of the same dim, then overwrite every saved
        # numpy attribute from the npz. Fresh construction handles wiring up
        # the quantum backend (auto-detect on this machine), capability
        # modules, parallel executor, etc.
        model = ZeroDataModel(dim=dim)

        # Round-3 audit B-batch: detect architecture mismatches up front. The
        # loop below silently truncates extra layers/functors, which is silent
        # state corruption when the snapshot was saved from a different model
        # topology. Fail loudly instead.
        saved_n_layers = int(config.get("n_consciousness_layers", 0))
        fresh_n_layers = len(model.consciousness.layers)
        if saved_n_layers != fresh_n_layers:
            raise ValueError(
                "snapshot n_consciousness_layers mismatch: config says "
                f"{saved_n_layers} but a fresh dim={dim} model has "
                f"{fresh_n_layers} layers; refusing to silently truncate"
            )
        saved_n_functors = int(config.get("n_functors", 0))
        fresh_n_functors = len(model.category_engine.functors)
        if saved_n_functors != fresh_n_functors:
            raise ValueError(
                "snapshot n_functors mismatch: config says "
                f"{saved_n_functors} but a fresh dim={dim} model has "
                f"{fresh_n_functors} functors; refusing to silently truncate"
            )
        morph_counts = config.get("functor_morphism_counts", [])
        if not isinstance(morph_counts, list) or len(morph_counts) != saved_n_functors:
            raise ValueError(
                "functor_morphism_counts must be a list of length "
                f"n_functors={saved_n_functors}, got {type(morph_counts).__name__} "
                f"of length {len(morph_counts) if isinstance(morph_counts, list) else 'n/a'}"
            )
        # Round-4 audit PERSIST-3: validate each morph count element is a
        # non-negative int (bool rejected — ``isinstance(True, int)`` is True
        # so we explicitly exclude bool). Negative counts would silently
        # discard every transform (``k >= -5`` is always True).
        # Round-5 audit PERSIST5-2: also enforce an upper bound to prevent OOM
        # via inflated counts, and compare against the fresh model's functor
        # morphism_map lengths (same pattern as n_consciousness_layers) so a
        # mismatch is rejected loudly instead of silently truncating/leaving
        # fresh-random transforms in place.
        fresh_morph_counts = [
            len(f.morphism_map) for f in model.category_engine.functors
        ]
        for idx, mc in enumerate(morph_counts):
            if isinstance(mc, bool) or not isinstance(mc, int) or mc < 0:
                raise ValueError(
                    f"functor_morphism_counts[{idx}] must be a non-negative int, "
                    f"got {type(mc).__name__}: {mc!r}"
                )
            if mc > _MAX_MORPH_COUNT:
                raise ValueError(
                    f"functor_morphism_counts[{idx}]={mc} exceeds cap "
                    f"{_MAX_MORPH_COUNT}; refusing to load"
                )
            fresh_mc = fresh_morph_counts[idx] if idx < len(fresh_morph_counts) else 0
            if mc != fresh_mc:
                raise ValueError(
                    f"functor_morphism_counts[{idx}]={mc} but a fresh dim={dim} "
                    f"model has {fresh_mc} morphisms for functor {idx}; "
                    "refusing to silently truncate or leave fresh-random transforms"
                )

        # Round-4 audit PERSIST-1: ``n_morphogens`` and ``n_fractal_transforms``
        # were written by save() but never validated by load(). A missing or
        # malformed value would either KeyError/TypeError deep in the load
        # loop, or silently truncate the morphogen/transform list (silent
        # state corruption — the very bug Round-3 B-batch tried to prevent
        # for layers/functors).
        n_morphogens = config.get("n_morphogens")
        if isinstance(n_morphogens, bool) or not isinstance(n_morphogens, int):
            raise ValueError(
                "config['n_morphogens'] must be a plain int, got "
                f"{type(n_morphogens).__name__}: {n_morphogens!r}"
            )
        if n_morphogens < 0 or n_morphogens > 1024:
            raise ValueError(
                f"refusing to load n_morphogens={n_morphogens}: must be in [0, 1024]"
            )
        n_fractal_transforms = config.get("n_fractal_transforms")
        if isinstance(n_fractal_transforms, bool) or not isinstance(n_fractal_transforms, int):
            raise ValueError(
                "config['n_fractal_transforms'] must be a plain int, got "
                f"{type(n_fractal_transforms).__name__}: {n_fractal_transforms!r}"
            )
        if n_fractal_transforms < 0 or n_fractal_transforms > 1024:
            raise ValueError(
                f"refusing to load n_fractal_transforms={n_fractal_transforms}: "
                "must be in [0, 1024]"
            )

        # Round-4 audit PERSIST-5: ``cycle_count`` must be a non-negative int
        # (bool/float rejected explicitly — ``int(True)==1`` would otherwise
        # silently pass).
        cycle_count_raw = config.get("cycle_count", 0)
        if isinstance(cycle_count_raw, bool) or not isinstance(cycle_count_raw, int):
            raise ValueError(
                "config['cycle_count'] must be a plain int, got "
                f"{type(cycle_count_raw).__name__}: {cycle_count_raw!r}"
            )
        if cycle_count_raw < 0:
            raise ValueError(
                f"config['cycle_count'] must be non-negative, got {cycle_count_raw}"
            )
        # Assign cycle_count now that all scalar config validation is done.
        model.cycle_count = cycle_count_raw

        # Round-7 audit CONCUR7-1: load from the in-memory bytes snapshot
        # read under the lock above, so the npz content is guaranteed to
        # be from the same snapshot as config (no torn read).
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
            # Round-4 audit PERSIST-2: build the set of expected npz keys and
            # verify they all exist before accessing them. Without this, a
            # missing key would raise a bare ``KeyError`` deep in the load
            # loop with no context about which snapshot was being loaded.
            # Round-5 audit PERSIST5-1: the key check MUST run before the
            # dtype loop. The old order iterated ``data.files`` (loading every
            # array to inspect its dtype) before checking keys — so a hostile
            # npz with 10 000 huge junk arrays would all be loaded into memory
            # before the missing-key check rejected the snapshot (OOM DoS).
            # Also reject EXTRA keys: a legitimate snapshot has exactly the
            # expected keys, so extras indicate corruption or tampering.
            expected_keys: set[str] = set()
            expected_keys.update({"active_inference_transition",
                                  "active_inference_emission",
                                  "active_inference_belief_state",
                                  "category_engine_topos_classifier",
                                  "quantum_hybrid_classical_weights",
                                  "quantum_hybrid_circuit_params",
                                  "quantum_hybrid_circuit_entangling",
                                  "quantum_hybrid_annealer_cost_matrix",
                                  "biological_morphogenetic_grid",
                                  "biological_automata_state"})
            for i in range(saved_n_layers):
                expected_keys.add(f"consciousness_layers_{i}_weights")
                expected_keys.add(f"consciousness_layers_{i}_bias")
            for j in range(saved_n_functors):
                for k in range(morph_counts[j]):
                    expected_keys.add(f"category_engine_functor_{j}_transform_{k}")
            for j in range(n_morphogens):
                expected_keys.add(f"biological_morphogenetic_morphogens_{j}")
            for j in range(n_fractal_transforms):
                expected_keys.add(f"math_universe_fractal_transforms_{j}_scale")
                expected_keys.add(f"math_universe_fractal_transforms_{j}_offset")
            actual_keys = set(data.files)
            missing = expected_keys - actual_keys
            if missing:
                raise ValueError(
                    f"arrays.npz missing required keys: {sorted(missing)}"
                )
            extra = actual_keys - expected_keys
            if extra:
                raise ValueError(
                    f"arrays.npz has unexpected keys: {sorted(extra)}"
                )

            # Round-5 audit PERSIST5-4: reject non-numeric dtypes. The old
            # check only rejected kind 'O' (object), letting bytes ('S'),
            # unicode ('U') and void/record ('V') arrays through — all of
            # which could carry unexpected payloads. Only iterate the expected
            # keys (already validated above) so a hostile npz cannot force us
            # to load junk arrays.
            #
            # Round-6 audit PERF6-2: read each expected array ONCE into a
            # local dict. ``NpzFile`` does not cache reads — every ``data[k]``
            # re-decompresses the array from the zip. The previous code read
            # each array twice (once for dtype inspection at line 513, again
            # during the assignment loop), which doubled disk I/O. Loading
            # into a dict once halves the read cost and lets the dtype check
            # inspect the cached array without a second disk read.
            loaded_arrays: dict[str, np.ndarray] = {}
            for k in expected_keys:
                arr = np.asarray(data[k])
                kind = arr.dtype.kind
                if kind in ("O", "S", "U", "V"):
                    raise ValueError(
                        f"refusing non-numeric dtype ({kind}) array: {k}"
                    )
                loaded_arrays[k] = arr

            # Consciousness core: per-layer weights and biases (overwrites each
            # layer in-place so layer objects keep their identity).
            for i, layer in enumerate(model.consciousness.layers):
                if i < config["n_consciousness_layers"]:
                    w = loaded_arrays[f"consciousness_layers_{i}_weights"]
                    b = loaded_arrays[f"consciousness_layers_{i}_bias"]
                    # Round-4 audit PERSIST-2: shape check guards against a
                    # malicious npz that sets dim=1 in config but stuffs
                    # huge arrays into the npz (OOM bypass).
                    if w.shape != (dim, dim):
                        raise ValueError(
                            f"layer {i} weights shape {w.shape} != ({dim},{dim})"
                        )
                    if b.shape != (dim,):
                        raise ValueError(
                            f"layer {i} bias shape {b.shape} != ({dim},)"
                        )
                    layer.weights = w
                    layer.bias = b

            # Active inference generative model arrays.
            gm = model.active_inference.generative_model
            t = loaded_arrays["active_inference_transition"]
            e = loaded_arrays["active_inference_emission"]
            bs = loaded_arrays["active_inference_belief_state"]
            if t.shape != (gm.state_dim, gm.state_dim):
                raise ValueError(
                    f"transition shape {t.shape} != ({gm.state_dim},{gm.state_dim})"
                )
            if e.shape != (gm.state_dim, gm.obs_dim):
                raise ValueError(
                    f"emission shape {e.shape} != ({gm.state_dim},{gm.obs_dim})"
                )
            if bs.shape != (gm.state_dim,):
                raise ValueError(
                    f"belief_state shape {bs.shape} != ({gm.state_dim},)"
                )
            gm.transition = t
            gm.emission = e
            gm.belief_state = bs

            # Category engine: topos classifier + functor transforms.
            # Round-5 audit PERSIST5-3: shape-check every array against the
            # fresh model's corresponding attribute, so a hostile npz cannot
            # stuff huge arrays (OOM bypass) or mismatched shapes (silent
            # broadcasting corruption).
            topos_clf = loaded_arrays["category_engine_topos_classifier"]
            if topos_clf.shape != model.category_engine.topos.classifier.shape:
                raise ValueError(
                    f"topos_classifier shape {topos_clf.shape} != "
                    f"{model.category_engine.topos.classifier.shape}"
                )
            model.category_engine.topos.classifier = topos_clf
            # ``morph_counts`` was validated above against ``n_functors``.
            for j, functor in enumerate(model.category_engine.functors):
                if j >= config["n_functors"]:
                    break
                n_morphs = morph_counts[j] if j < len(morph_counts) else 0
                morph_items = list(functor.morphism_map.items())
                for k, (key, fresh_transform) in enumerate(morph_items):
                    if k >= n_morphs:
                        break
                    loaded = loaded_arrays[
                        f"category_engine_functor_{j}_transform_{k}"
                    ]
                    if loaded.shape != fresh_transform.shape:
                        raise ValueError(
                            f"functor {j} transform {k} shape {loaded.shape} "
                            f"!= {fresh_transform.shape}"
                        )
                    functor.morphism_map[key] = loaded

            # Quantum hybrid arrays.
            qh_cw = loaded_arrays["quantum_hybrid_classical_weights"]
            if qh_cw.shape != model.quantum_hybrid.classical_weights.shape:
                raise ValueError(
                    f"classical_weights shape {qh_cw.shape} != "
                    f"{model.quantum_hybrid.classical_weights.shape}"
                )
            model.quantum_hybrid.classical_weights = qh_cw

            qh_cp = loaded_arrays["quantum_hybrid_circuit_params"]
            if qh_cp.shape != model.quantum_hybrid.quantum_circuit.params.shape:
                raise ValueError(
                    f"circuit_params shape {qh_cp.shape} != "
                    f"{model.quantum_hybrid.quantum_circuit.params.shape}"
                )
            model.quantum_hybrid.quantum_circuit.params = qh_cp

            qh_ce = loaded_arrays["quantum_hybrid_circuit_entangling"]
            if qh_ce.shape != model.quantum_hybrid.quantum_circuit.entangling.shape:
                raise ValueError(
                    f"circuit_entangling shape {qh_ce.shape} != "
                    f"{model.quantum_hybrid.quantum_circuit.entangling.shape}"
                )
            model.quantum_hybrid.quantum_circuit.entangling = qh_ce

            qh_ac = loaded_arrays["quantum_hybrid_annealer_cost_matrix"]
            if qh_ac.shape != model.quantum_hybrid.annealer.cost_matrix.shape:
                raise ValueError(
                    f"annealer_cost_matrix shape {qh_ac.shape} != "
                    f"{model.quantum_hybrid.annealer.cost_matrix.shape}"
                )
            model.quantum_hybrid.annealer.cost_matrix = qh_ac

            # Biological substrate arrays.
            bio_grid = loaded_arrays["biological_morphogenetic_grid"]
            if bio_grid.shape != model.biological.morphogenetic.grid.shape:
                raise ValueError(
                    f"morphogenetic_grid shape {bio_grid.shape} != "
                    f"{model.biological.morphogenetic.grid.shape}"
                )
            model.biological.morphogenetic.grid = bio_grid
            # Replace the morphogen list to preserve length even if the source
            # machine had a different signal count (default is 3).
            fresh_morphogen_shape = (
                model.biological.morphogenetic.morphogens[0].shape
                if model.biological.morphogenetic.morphogens
                else None
            )
            loaded_morphogens: list[np.ndarray] = []
            for j in range(n_morphogens):
                mg = loaded_arrays[f"biological_morphogenetic_morphogens_{j}"]
                if fresh_morphogen_shape is not None and mg.shape != fresh_morphogen_shape:
                    raise ValueError(
                        f"morphogen {j} shape {mg.shape} != {fresh_morphogen_shape}"
                    )
                loaded_morphogens.append(mg)
            model.biological.morphogenetic.morphogens = loaded_morphogens

            bio_auto = loaded_arrays["biological_automata_state"]
            if bio_auto.shape != model.biological.automata.state.shape:
                raise ValueError(
                    f"automata_state shape {bio_auto.shape} != "
                    f"{model.biological.automata.state.shape}"
                )
            model.biological.automata.state = bio_auto

            # Math universe fractal transforms (scale + offset pairs).
            fresh_scale_shape = (
                model.math_universe.fractal.transforms[0][0].shape
                if model.math_universe.fractal.transforms
                else None
            )
            fresh_offset_shape = (
                model.math_universe.fractal.transforms[0][1].shape
                if model.math_universe.fractal.transforms
                else None
            )
            loaded_transforms: list[tuple[np.ndarray, np.ndarray]] = []
            for j in range(n_fractal_transforms):
                scale = loaded_arrays[f"math_universe_fractal_transforms_{j}_scale"]
                offset = loaded_arrays[f"math_universe_fractal_transforms_{j}_offset"]
                if fresh_scale_shape is not None and scale.shape != fresh_scale_shape:
                    raise ValueError(
                        f"fractal transform {j} scale shape {scale.shape} "
                        f"!= {fresh_scale_shape}"
                    )
                if fresh_offset_shape is not None and offset.shape != fresh_offset_shape:
                    raise ValueError(
                        f"fractal transform {j} offset shape {offset.shape} "
                        f"!= {fresh_offset_shape}"
                    )
                loaded_transforms.append((scale, offset))
            model.math_universe.fractal.transforms = loaded_transforms

        return model


__all__ = [
    "ModelSerializer",
    "set_persistence_root",
    "get_persistence_root",
]
