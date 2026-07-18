# tests/test_round10_regressions.py
"""Regression tests for the Round-10 audit fixes.

Each test guards one Round-10 finding against silent regression. The
findings came from a "super-military-grade" three-angle audit (numerical
correctness, concurrency/safety, API contract/documentation/test
coverage) that was stricter than Round-9.

  R10-C-001  math_universe.update clips prediction_error to [0, 1e6]
             + clamps step to 0.1 (no gradient ascent, no runaway noise)
  R10-C-002  math_universe.topological_features calls connected_components_1d
             directly (no DeprecationWarning leak on every think() cycle)
  R10-C-003  ShapeAnalyzer._betti0 calls connected_components_1d directly
             (no DeprecationWarning leak)
  R10-C-004  consciousness_core.update clips to [0, 1e6] (no abs() / no
             negative-prediction-error gradient ascent)
  R10-C-005  __init__.pyi stub aliases the class import to
             _ZeroDataMCPServer so the runtime attribute type is
             unambiguous (``type[...] | None``)
  R10-C-006  detect_script docstring lists all 11 supported scripts
             (was stale at 5 scripts since Fix 18)
  R10-C-010  model.think() metadata carries ``self_generated`` flag
  R10-A-001  ModelSerializer.save/load persists the parent + 6 child RNG
             states (rng_state.json) + the seed (config.json) so seeded
             reproducibility survives save→load
  R10-A-004  CategoryTheoryEngine.find_invertible_map rejects extreme
             scale ratios (|scale| outside [1e-6, 1e6])
  R10-A-005  DepthEstimator._texture_gradient uses the two-pass stable
             variance form (no catastrophic cancellation on bright
             backgrounds with small texture variation)
  R10-A-006  BayesianEstimator._update_normal clamps lik_prec to 1e12
             symmetrically with prior_prec (no spurious over-confidence
             from a degenerate known_var like 1e-20)
  R10-A-008  hardware.kernels._betti_numbers returns betti_1=0 for 1D
             point clouds (Vietoris-Rips complex has no 1-cycles in 1D)
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _sandbox_root(tmp_path):
    """Sandbox persistence under tmp_path so save/load tests do not touch
    the real persistence root."""
    from zero_data_model.persistence import set_persistence_root

    root = tmp_path / "persistence"
    set_persistence_root(str(root))
    yield root
    set_persistence_root(str(tmp_path / "_reset"))


# --------------------------------------------------------------------------- #
# R10-C-001: math_universe.update clips to [0, 1e6] + step clamp 0.1
# --------------------------------------------------------------------------- #


def test_r10_c_001_math_universe_update_clips_negative_prediction_error():
    """``math_universe.update`` must clip ``prediction_error`` to ``[0, 1e6]``.

    A negative ``prediction_error`` would invert the noise sign (gradient
    ascent), destroying learned fractal transforms. We verify by feeding
    a large negative error and confirming the resulting transform drift is
    bounded by the same step clamp as a positive error of the same
    magnitude (i.e. the negative sign was clipped to 0, not propagated).
    """
    from zero_data_model.math_universe import MathematicalUniverse

    mu_neg = MathematicalUniverse(dim=8, rng=np.random.default_rng(42))
    mu_pos = MathematicalUniverse(dim=8, rng=np.random.default_rng(42))
    # Both start with identical transforms.
    assert all(
        np.allclose(s1, s2) and np.allclose(o1, o2)
        for (s1, o1), (s2, o2) in zip(
            mu_neg.fractal.transforms, mu_pos.fractal.transforms, strict=True
        )
    )

    mu_neg.update(-1e3)
    mu_pos.update(0.0)  # clipped -1e3 == 0.0 (step == 0, no-op)

    for (s1, _o1), (s2, _o2) in zip(
        mu_neg.fractal.transforms, mu_pos.fractal.transforms, strict=True
    ):
        assert np.allclose(s1, s2), (
            "math_universe.update(-1e3) drifted transforms differently "
            "than update(0.0); the negative error was not clipped to 0"
        )


def test_r10_c_001_math_universe_update_clamps_step_to_0_1():
    """A ``prediction_error`` near the 1e6 ceiling must not inject more
    than 0.1 std-dev noise per step (``1e6 * 0.001 = 1000`` was the old
    runaway behavior)."""
    from zero_data_model.math_universe import MathematicalUniverse

    mu_before = MathematicalUniverse(dim=8, rng=np.random.default_rng(7))
    before_scales = [s.copy() for s, _ in mu_before.fractal.transforms]
    # Run an extreme update that, unclamped, would inject 1000 std-dev
    # noise per element (1e6 * 0.001 = 1000). With the clamp, it should
    # inject at most 0.1 * 4 (4-sigma draw) per element.
    mu_before.update(1e6)
    max_delta = 0.0
    for (s_after, _), s_before in zip(
        mu_before.fractal.transforms, before_scales, strict=True
    ):
        delta = float(np.max(np.abs(s_after - s_before)))
        max_delta = max(max_delta, delta)
    # 0.1 std-dev noise per element should not move any element by more
    # than ~0.5 (5 sigma). The unclamped path would move elements by
    # 1000+ sigma = thousands.
    assert max_delta < 1.0, (
        f"step clamp failed: max transform delta = {max_delta} (should be "
        f"< 1.0 with a 0.1 step clamp)"
    )


# --------------------------------------------------------------------------- #
# R10-C-002: math_universe.topological_features emits no DeprecationWarning
# --------------------------------------------------------------------------- #


def test_r10_c_002_math_universe_topological_features_no_deprecation_warning():
    """``topological_features`` must NOT emit a ``DeprecationWarning``.

    The previous implementation called the deprecated
    ``compute_betti_numbers`` wrapper, which forwarded to
    ``connected_components_1d`` but emitted a warning each call. Since
    ``topological_features`` is on the ``think()`` hot path, every cycle
    polluted the warning stream (and crashed callers that escalate
    DeprecationWarning to an error).
    """
    from zero_data_model.math_universe import MathematicalUniverse

    mu = MathematicalUniverse(dim=8, rng=np.random.default_rng(0))
    data = np.array([0.1, 0.2, 0.9, 1.5, 5.0, 5.1, 5.2, 7.0])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=DeprecationWarning)
        # Must not raise. (``topological_features`` lives on the
        # TopologicalAnalyzer instance, not on MathematicalUniverse directly.)
        features = mu.topology.topological_features(data)
    assert features.shape == (8,)
    assert np.isfinite(features).all()


# --------------------------------------------------------------------------- #
# R10-C-003: ShapeAnalyzer._betti0 emits no DeprecationWarning
# --------------------------------------------------------------------------- #


def test_r10_c_003_shape_analyzer_betti0_no_deprecation_warning():
    """``ShapeAnalyzer._betti0`` must NOT emit a ``DeprecationWarning``.

    Same R10-C-002 pattern: the previous implementation called the
    deprecated ``compute_betti_numbers`` wrapper, leaking a warning on
    every ``analyze`` call.
    """
    from zero_data_model.capabilities.vision import ShapeAnalyzer

    analyzer = ShapeAnalyzer(dim=8)
    img = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=DeprecationWarning)
        # analyze runs _betti0 internally; must not raise.
        result = analyzer.analyze(img)
    assert "beti0" in result
    assert np.isfinite(result["beti0"])


# --------------------------------------------------------------------------- #
# R10-C-004: consciousness_core.update clips to [0, 1e6] (no abs())
# --------------------------------------------------------------------------- #


def test_r10_c_004_consciousness_core_update_clips_negative():
    """``consciousness_core.update`` must clip ``prediction_error`` to
    ``[0, 1e6]``. The previous ``[-1e6, 1e6]`` clip + ``abs()`` was dead
    code (MSE is non-negative) but signalled an inconsistent contract.

    We verify by passing a large negative error and confirming the
    layer weights drift identically to ``update(0.0)`` (i.e. the negative
    was clipped to 0, producing a no-op since step == 0.0).
    """
    from zero_data_model.consciousness_core import ConsciousnessCore

    cc_neg = ConsciousnessCore(dim=8, rng=np.random.default_rng(11))
    cc_zero = ConsciousnessCore(dim=8, rng=np.random.default_rng(11))
    # Identical starting weights.
    for l1, l2 in zip(cc_neg.layers, cc_zero.layers, strict=True):
        assert np.allclose(l1.weights, l2.weights)

    cc_neg.update(-1e3)
    cc_zero.update(0.0)
    for l1, l2 in zip(cc_neg.layers, cc_zero.layers, strict=True):
        assert np.allclose(l1.weights, l2.weights), (
            "consciousness_core.update(-1e3) drifted weights differently "
            "than update(0.0); negative error was not clipped to 0"
        )


# --------------------------------------------------------------------------- #
# R10-C-005: __init__.pyi stub aliases the class import
# --------------------------------------------------------------------------- #


def test_r10_c_005_init_stub_aliases_class_import():
    """The ``__init__.pyi`` stub must alias the class import to
    ``_ZeroDataMCPServer`` so the runtime attribute ``ZeroDataMCPServer``
    has an unambiguous type annotation (``type[_ZeroDataMCPServer] | None``)
    distinct from the class itself.

    We verify by parsing the stub and checking:
      * the import line aliases to ``_ZeroDataMCPServer``
      * the annotation references ``_ZeroDataMCPServer`` (not the
        ambiguous same-name binding that previously existed)
    """
    import pathlib

    stub_path = (
        pathlib.Path(__import__("zero_data_model").__file__).parent
        / "__init__.pyi"
    )
    stub_text = stub_path.read_text(encoding="utf-8")
    # The class is imported under an alias, not bound to its original name.
    assert "from .mcp_server import ZeroDataMCPServer as _ZeroDataMCPServer" in stub_text, (
        "stub should alias the class import to _ZeroDataMCPServer"
    )
    # The annotation references the alias, not the original name.
    assert "ZeroDataMCPServer: type[_ZeroDataMCPServer] | None" in stub_text, (
        "stub annotation should reference _ZeroDataMCPServer (the alias)"
    )


# --------------------------------------------------------------------------- #
# R10-C-006: detect_script docstring lists all 11 supported scripts
# --------------------------------------------------------------------------- #


def test_r10_c_006_detect_script_docstring_lists_11_scripts():
    """The ``detect_script`` docstring must list all 11 supported scripts
    (latin, cyrillic, greek, hebrew, arabic, devanagari, thai, hiragana,
    katakana, cjk, hangul) so callers know what to expect. The previous
    docstring listed only 5 scripts (latin/cyrillic/cjk/arabic/mixed).

    We verify in all three places that carry the docstring:
      * model.ZeroDataModel.detect_script
      * mcp_server.ZeroDataMCPServer.detect_script
      * model.pyi stub
    """
    import pathlib
    import re

    expected_scripts = {
        "latin", "cyrillic", "greek", "hebrew", "arabic",
        "devanagari", "thai", "hiragana", "katakana", "cjk", "hangul",
    }

    # 1. Runtime model.detect_script docstring.
    from zero_data_model.model import ZeroDataModel
    doc = ZeroDataModel.detect_script.__doc__ or ""
    # Extract the parenthesized comma-separated list of scripts.
    match = re.search(r"\(([^)]+)\)", doc)
    assert match is not None, f"docstring has no script list: {doc!r}"
    listed = {s.strip().lower() for s in match.group(1).split(",")}
    assert expected_scripts.issubset(listed), (
        f"model.detect_script docstring missing scripts: "
        f"{expected_scripts - listed}"
    )

    # 2. MCP server detect_script docstring.
    from zero_data_model.mcp_server import ZeroDataMCPServer
    doc = ZeroDataMCPServer.detect_script.__doc__ or ""
    match = re.search(r"\(([^)]+)\)", doc)
    assert match is not None, f"mcp docstring has no script list: {doc!r}"
    listed = {s.strip().lower() for s in match.group(1).split(",")}
    assert expected_scripts.issubset(listed), (
        f"mcp_server.detect_script docstring missing scripts: "
        f"{expected_scripts - listed}"
    )

    # 3. Stub file docstring.
    stub_path = (
        pathlib.Path(__import__("zero_data_model").__file__).parent
        / "model.pyi"
    )
    stub_text = stub_path.read_text(encoding="utf-8")
    # Find the detect_script docstring in the stub.
    match = re.search(
        r"def detect_script\(self, text: str\) -> str:\s*\"\"\"(.+?)\"\"\"",
        stub_text, re.DOTALL,
    )
    assert match is not None, "stub has no detect_script docstring"
    script_list_match = re.search(r"\(([^)]+)\)", match.group(1))
    assert script_list_match is not None, (
        f"stub docstring has no script list: {match.group(1)!r}"
    )
    listed = {s.strip().lower() for s in script_list_match.group(1).split(",")}
    assert expected_scripts.issubset(listed), (
        f"stub detect_script docstring missing scripts: "
        f"{expected_scripts - listed}"
    )


# --------------------------------------------------------------------------- #
# R10-C-010: model.think() metadata carries self_generated
# --------------------------------------------------------------------------- #


def test_r10_c_010_think_metadata_has_self_generated_true_when_no_input():
    """``ZeroDataModel.think()`` with no input must set
    ``metadata["self_generated"] = True`` so downstream callers can tell
    the cycle ran on internally-generated content.
    """
    from zero_data_model.model import ZeroDataModel

    model = ZeroDataModel(dim=8, seed=1)
    out = model.think()  # no input -> self-generated
    assert "self_generated" in out.metadata, (
        "think() metadata missing 'self_generated' key"
    )
    assert out.metadata["self_generated"] is True


def test_r10_c_010_think_metadata_has_self_generated_false_with_input():
    """``ZeroDataModel.think(signal)`` with real input must set
    ``metadata["self_generated"] = False``.
    """
    from zero_data_model.model import ZeroDataModel

    model = ZeroDataModel(dim=8, seed=1)
    signal = np.ones(8) * 0.3
    out = model.think(signal)
    assert "self_generated" in out.metadata, (
        "think() metadata missing 'self_generated' key"
    )
    assert out.metadata["self_generated"] is False


# --------------------------------------------------------------------------- #
# R10-A-001: ModelSerializer persists RNG state across save→load
# --------------------------------------------------------------------------- #


def test_r10_a_001_save_load_preserves_rng_state_for_seeded_model():
    """``ModelSerializer.save`` then ``load`` must restore the parent +
    child RNG states so a seeded model resumes from the exact RNG
    position. Without this, ``think()`` after load diverges from
    ``think()`` on the saved model — breaking seeded reproducibility.

    We verify by:
      1. Constructing a seeded model and advancing its RNG with one
         ``think()`` cycle.
      2. Saving the model and loading it back.
      3. Comparing the parent + 6 child ``bit_generator.state`` dicts
         between the original and the loaded model — they must be
         exactly equal (state was preserved, not re-seeded from scratch).
    """
    from zero_data_model.model import ZeroDataModel
    from zero_data_model.persistence import ModelSerializer

    model_a = ZeroDataModel(dim=8, seed=42)
    # Advance the RNG by drawing from it (think() draws stochastic noise).
    sig = np.ones(8) * 0.4
    model_a.think(sig)

    # Save A and load it back.
    ModelSerializer.save(model_a, "snap_rng")
    model_b = ModelSerializer.load("snap_rng")

    # Parent RNG state must match exactly.
    state_a = model_a._rng.bit_generator.state
    state_b = model_b._rng.bit_generator.state
    assert state_a == state_b, (
        "parent RNG state differs across save→load; seeded reproducibility broken"
    )
    # Each of the 6 child RNG states must match exactly.
    assert len(model_a.modules) == len(model_b.modules) == 6
    for i, (mod_a, mod_b) in enumerate(
        zip(model_a.modules, model_b.modules, strict=True)
    ):
        sa = mod_a._rng.bit_generator.state
        sb = mod_b._rng.bit_generator.state
        assert sa == sb, (
            f"child RNG state {i} ({type(mod_a).__name__}) differs across "
            f"save→load; seeded reproducibility broken"
        )


def test_r10_a_001_save_writes_rng_state_json_and_seed_in_config():
    """The save directory must contain a ``rng_state.json`` file and the
    ``config.json`` must include the ``seed`` field."""
    import pathlib

    from zero_data_model.model import ZeroDataModel
    from zero_data_model.persistence import ModelSerializer, get_persistence_root

    model = ZeroDataModel(dim=8, seed=99)
    ModelSerializer.save(model, "snap_files")
    snap_dir = pathlib.Path(get_persistence_root()) / "snap_files"
    # rng_state.json must exist.
    assert (snap_dir / "rng_state.json").is_file(), (
        "save() did not write rng_state.json"
    )
    # config.json must include the seed.
    with open(snap_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    assert "seed" in cfg, "config.json missing 'seed' key"
    assert cfg["seed"] == 99, f"config['seed'] = {cfg['seed']!r}, expected 99"
    # rng_state.json must have parent_state + child_states.
    with open(snap_dir / "rng_state.json", encoding="utf-8") as f:
        rng_state = json.load(f)
    assert "parent_state" in rng_state
    assert "child_states" in rng_state
    assert isinstance(rng_state["child_states"], list)
    assert len(rng_state["child_states"]) == 6, (
        f"expected 6 child RNG states, got {len(rng_state['child_states'])}"
    )


def test_r10_a_001_load_validates_seed_type():
    """``load`` must reject a malformed ``seed`` (bool/float/string) with
    a ``ValueError`` rather than silently coercing it."""
    import pathlib

    from zero_data_model.model import ZeroDataModel
    from zero_data_model.persistence import ModelSerializer, get_persistence_root

    model = ZeroDataModel(dim=8, seed=1)
    ModelSerializer.save(model, "snap_bad_seed")
    snap_dir = pathlib.Path(get_persistence_root()) / "snap_bad_seed"
    cfg_path = snap_dir / "config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["seed"] = True  # bool — must be rejected
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    with pytest.raises(ValueError, match="seed"):
        ModelSerializer.load("snap_bad_seed")


# --------------------------------------------------------------------------- #
# R10-A-004: find_invertible_map rejects extreme scale ratios
# --------------------------------------------------------------------------- #


def test_r10_a_004_find_invertible_map_rejects_extreme_scale():
    """``find_invertible_map`` must return ``None`` when the source/target
    norm ratio is outside ``[1e-6, 1e6]``. A scale of 1e7 produces a
    condition number of ~1e14 (scale²) which amplifies float64 round-off
    into the significant digits of any downstream transfer_solution call.
    """
    from zero_data_model.category_engine import CategoryTheoryEngine

    engine = CategoryTheoryEngine(dim=4, rng=np.random.default_rng(0))
    source = np.array([1.0, 0.0, 0.0, 0.0])
    # target norm is 1e7 * source norm -> scale = 1e7 (rejected)
    target_huge = source * 1e7
    result_huge = engine.find_invertible_map(source, target_huge)
    assert result_huge is None, (
        "find_invertible_map accepted scale=1e7 (should reject > 1e6)"
    )
    # target norm is 1e-7 * source norm -> scale = 1e-7 (rejected)
    target_tiny = source * 1e-7
    result_tiny = engine.find_invertible_map(source, target_tiny)
    assert result_tiny is None, (
        "find_invertible_map accepted scale=1e-7 (should reject < 1e-6)"
    )


def test_r10_a_004_find_invertible_map_accepts_moderate_scale():
    """Sanity check: moderate scales (1e-3 to 1e3) must still succeed."""
    from zero_data_model.category_engine import CategoryTheoryEngine

    engine = CategoryTheoryEngine(dim=4, rng=np.random.default_rng(0))
    source = np.array([1.0, 0.5, 0.0, 0.0])
    for scale in (1e-3, 1e-2, 1.0, 1e2, 1e3):
        target = source * scale
        result = engine.find_invertible_map(source, target)
        assert result is not None, (
            f"find_invertible_map rejected scale={scale} (should accept)"
        )
        # Verify the map actually maps source -> target.
        mapped = result @ source
        assert np.allclose(mapped, target, atol=1e-9), (
            f"scale={scale}: T @ source = {mapped}, expected {target}"
        )


# --------------------------------------------------------------------------- #
# R10-A-005: _texture_gradient stable on bright backgrounds
# --------------------------------------------------------------------------- #


def test_r10_a_005_texture_gradient_stable_for_large_mean():
    """``DepthEstimator._texture_gradient`` must not suffer catastrophic
    cancellation when row values have a large mean relative to their std
    (e.g. a bright constant background plus small texture variation).

    The old ``E[X²] - (E[X])²`` form lost 8+ digits of precision on
    bright backgrounds, sometimes going negative before the clamp.
    The new two-pass ``mean((x - mean)²)`` form is exact to float64.
    """
    from zero_data_model.capabilities.vision_advanced import DepthEstimator

    de = DepthEstimator(dim=8)
    # Bright background (mean = 1e6) with tiny texture (std = 1e-3).
    # Old form: 1e12 - 1e12 + epsilon -> catastrophic cancellation.
    rng = np.random.default_rng(0)
    background = 1e6
    texture = rng.standard_normal((4, 8)) * 1e-3
    img = background + texture
    grad = de._texture_gradient(img)
    assert np.isfinite(grad).all(), (
        "_texture_gradient produced non-finite values on bright background"
    )
    assert (grad >= 0.0).all(), (
        "_texture_gradient produced negative values (catastrophic cancellation)"
    )
    # The texture std (~1e-3) is small but non-zero, so the gradient
    # should NOT be identically zero (which would indicate the variance
    # computation collapsed to zero via cancellation).
    assert grad.std() > 0.0, (
        "_texture_gradient collapsed to constant zero (variance lost to "
        "catastrophic cancellation)"
    )


def test_r10_a_005_texture_gradient_matches_reference_two_pass():
    """``_texture_gradient`` must match a reference two-pass variance
    computation to float64 precision. This guards against silent
    regression back to the unstable ``E[X²] - (E[X])²`` form."""
    from zero_data_model.capabilities.vision_advanced import DepthEstimator

    de = DepthEstimator(dim=8)
    rng = np.random.default_rng(123)
    img = rng.standard_normal((5, 6)) * 0.1 + 0.5
    got = de._texture_gradient(img)

    # Reference: two-pass per-row rolling std, normalized to [0, 1].
    # MUST include the ``+ 1e-8`` epsilon used by the implementation's
    # denominator (otherwise we'd see a ~1e-7 absolute discrepancy on the
    # final normalized output, which is not a precision bug — it's the
    # documented epsilon).
    window = 3
    h, w = img.shape
    ref_stds = np.zeros(h)
    for i in range(h):
        row = img[i]
        windows = np.lib.stride_tricks.sliding_window_view(row, window)
        means = windows.mean(axis=1, keepdims=True)
        var = ((windows - means) ** 2).mean(axis=1)
        ref_stds[i] = float(np.mean(np.sqrt(np.maximum(var, 0.0))))
    rs = ref_stds - ref_stds.min()
    rs_max = rs.max()
    if rs_max > 1e-8:
        # Match the implementation's epsilon exactly.
        rs_norm = rs / (rs_max + 1e-8)
        ref = np.tile(rs_norm[:, None], (1, w))
    else:
        ref = np.zeros((h, w))
    np.testing.assert_allclose(got, ref, atol=1e-12)


# --------------------------------------------------------------------------- #
# R10-A-006: BayesianEstimator lik_prec clamp
# --------------------------------------------------------------------------- #


def test_r10_a_006_bayesian_estimator_lik_prec_clamped_for_degenerate_known_var():
    """``BayesianEstimator._update_normal`` must clamp ``lik_prec`` to
    1e12 symmetrically with ``prior_prec``. A degenerate ``known_var``
    (e.g. 1e-20) would otherwise make a single observation override any
    prior strength AND collapse the posterior variance to spurious
    over-confidence (``1 / (1e12 + 1e20) ≈ 1e-20``).
    """
    from zero_data_model.capabilities.analytics_advanced import BayesianEstimator

    est = BayesianEstimator(dim=4)
    # Set a degenerate known_var that would make lik_prec = 1e20 unclamped.
    est.known_var = 1e-20
    est.reset(prior_mean=0.0, prior_var=1.0)

    # Prior is N(0, 1). Observe x=5 (5 sigma away). With lik_prec clamped
    # to 1e12, posterior_var should be approximately 1 / (1 + 1e12) ≈ 1e-12,
    # NOT 1 / (1 + 1e20) ≈ 1e-20 (the unclamped over-confident value).
    est._update_normal(5.0)
    # The clamp keeps posterior_var well above 1e-20 (the spurious
    # over-confidence floor). 1e-12 is the symmetric clamp; we accept
    # anything above 1e-15 as "not over-confident".
    assert est.normal_var > 1e-15, (
        f"posterior_var = {est.normal_var} collapsed to spurious "
        f"over-confidence (lik_prec not clamped)"
    )


def test_r10_a_006_bayesian_estimator_lik_prec_zero_known_var_does_not_crash():
    """A ``known_var`` of 0 must not raise ``ZeroDivisionError``. The
    clamp guards against this by max-ing known_var with 1e-12 before
    taking the reciprocal."""
    from zero_data_model.capabilities.analytics_advanced import BayesianEstimator

    est = BayesianEstimator(dim=4)
    est.known_var = 0.0
    est.reset(prior_mean=0.0, prior_var=1.0)
    # Must not raise.
    est._update_normal(1.0)
    assert np.isfinite(est.normal_mean)
    assert np.isfinite(est.normal_var)


# --------------------------------------------------------------------------- #
# R10-A-008: _betti_numbers returns betti_1 = 0 for 1D point clouds
# --------------------------------------------------------------------------- #


def test_r10_a_008_betti_numbers_returns_zero_betti1():
    """``hardware.kernels._betti_numbers`` must return ``betti_1 = 0``
    for 1D point clouds. A Vietoris-Rips complex on a 1D point cloud
    has no 2-simplices, so its first homology group ``H_1`` is trivially
    zero. The previous ``betti_1 = n_points - betti_0`` formula was
    unrelated to the actual first Betti number.
    """
    from zero_data_model.hardware import kernels

    rng = np.random.default_rng(0)
    for _ in range(5):
        n = rng.integers(2, 50)
        data = np.sort(rng.standard_normal(int(n)))
        max_radius = float(rng.uniform(0.5, 2.0))
        b0, b1 = kernels._betti_numbers(
            np.ascontiguousarray(data, dtype=float),
            max_radius,
        )
        assert b1 == 0, (
            f"_betti_numbers returned betti_1={b1} (should be 0 for 1D "
            f"point clouds); n_points={n}, max_radius={max_radius}, b0={b0}"
        )
        # Sanity: b0 must be a positive integer <= n_points.
        assert 1 <= b0 <= n, f"b0={b0} out of valid range [1, {n}]"


def test_r10_a_008_betti_numbers_b0_unchanged_by_betti1_fix():
    """The R10-A-008 fix only zeroed out ``betti_1``; ``betti_0`` (the
    count of connected components) must be unaffected. We verify by
    comparing against a pure-python reference loop."""
    from zero_data_model.hardware import kernels

    rng = np.random.default_rng(7)
    for _ in range(5):
        n = int(rng.integers(2, 30))
        data = np.sort(rng.standard_normal(n))
        max_radius = float(rng.uniform(0.3, 1.5))
        # Reference: gap-detection loop (R9-001 contract).
        ref_b0 = 1
        for i in range(1, n):
            if data[i] - data[i - 1] > max_radius:
                ref_b0 += 1
        got_b0, _ = kernels._betti_numbers(
            np.ascontiguousarray(data, dtype=float),
            max_radius,
        )
        assert got_b0 == ref_b0, (
            f"_betti_numbers b0 changed: got={got_b0}, expected={ref_b0} "
            f"(n={n}, max_radius={max_radius})"
        )
