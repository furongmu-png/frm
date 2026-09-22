# experiments/text_experience_buffer.py
"""Text-domain experience replay buffer (Phase I, Task 1.1).

This module is a THIN SHIM over the existing
``experiments/experience_buffer.py``. The original buffer is already
modality-agnostic — it stores ``(obs_vec, action_idx, next_obs_vec,
prediction_error, info_gain)`` tuples, which exactly matches what the
text-reading loop produces (the obs_vec is the TextEncoder output, the
action_idx is the navigation action 0..4).

We re-export the same classes here so:

  1. The text-loop code can import from a single domain-specific
     module (``from text_experience_buffer import TextExperienceBuffer``)
     rather than the generic ``experience_buffer`` — the import path
     documents INTENT.
  2. If text-specific extensions are later needed (e.g. storing the
     raw text block alongside the obs vector, or co-occurrence-based
     priority), they can be added to ``TextExperienceBuffer`` without
     touching the sandbox buffer.

Why no copy-paste?
  The original buffer is ~300 lines, well-tested
  (``experiments/test_experience_buffer.py`` — 11 tests covering all 3
  sampling modes, FE reduction, pickle round-trip), and the storage
  contract is identical. Reusing it avoids divergence.

Usage:
    from text_experience_buffer import TextExperienceBuffer, TextConsolidator

    buffer = TextExperienceBuffer(capacity=5000, seed=42)
    buffer.add(obs=obs_vec, action=nav_action_idx,
               next_obs=next_obs_vec,
               error=free_energy, info_gain=ig, step=step)
    batch = buffer.sample(32, mode="priority")
"""

from __future__ import annotations

# Re-export the generic buffer + consolidator unchanged.
from experience_buffer import (  # noqa: F401
    ConsolidationLog,
    ConsolidationResult,
    Experience,
    ExperienceBuffer,
    OfflineConsolidator,
)


class TextExperienceBuffer(ExperienceBuffer):
    """Text-domain alias for ``ExperienceBuffer``.

    No behavioural changes — the field order
    ``(obs, action, next_obs, error, info_gain, step)`` matches the
    text loop's natural output. The class exists to make imports
    self-documenting and to provide a hook for future text-specific
    extensions (e.g. storing the raw text block for word-boundary
    analysis in the consolidation phase).
    """

    pass


class TextConsolidator(OfflineConsolidator):
    """Text-domain alias for ``OfflineConsolidator``.

    The consolidation algorithm (sample batch → update_belief → update)
    is identical to the sandbox path: both feed the same
    ``ActiveInferenceEngine`` API. The alias is for import clarity.
    """

    pass


__all__ = [
    "Experience",
    "ExperienceBuffer",
    "TextExperienceBuffer",
    "OfflineConsolidator",
    "TextConsolidator",
    "ConsolidationResult",
    "ConsolidationLog",
]
