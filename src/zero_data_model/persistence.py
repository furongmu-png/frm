# src/zero_data_model/persistence.py
"""Model persistence for the zero-data cognitive system.

Saves/loads the full model state to a directory on disk using numpy
``.npz`` for arrays and JSON for configuration metadata. Pickle is
intentionally NOT used so the on-disk format is portable, auditable and
safe to load from untrusted sources.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .model import ZeroDataModel


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
        """Save model state to a directory: arrays.npz + config.json."""
        os.makedirs(path, exist_ok=True)

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
        """Reconstruct a :class:`ZeroDataModel` from a saved directory."""
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model directory not found: {path}")

        npz_path = os.path.join(path, "arrays.npz")
        config_path = os.path.join(path, "config.json")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Missing arrays.npz in: {path}")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Missing config.json in: {path}")

        with open(config_path, "r", encoding="utf-8") as f:
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
