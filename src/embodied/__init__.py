"""Embodied active perception loop (Stage-1 cognitive upgrade).

Provides:
  - EmbodiedBody             : first-person 3D-ish body with gaze/touch/force
  - SensorimotorPredictor    : predicts sensory consequences of perceptual actions
  - ProprioceptiveEncoder    : encodes joint/postural state into latent space

All components follow the FEP / active-inference style of the rest of the
codebase: thread-safe (RLock), local Hebbian updates, NaN guards, numpy-only.
"""
from .body_env import EmbodiedBody, EmbodiedObservation, BodyAction
from .sensorimotor_predictor import SensorimotorPredictor, SensorimotorPrediction
from .proprioception import ProprioceptiveEncoder, ProprioceptiveState

__all__ = [
    "EmbodiedBody",
    "EmbodiedObservation",
    "BodyAction",
    "SensorimotorPredictor",
    "SensorimotorPrediction",
    "ProprioceptiveEncoder",
    "ProprioceptiveState",
]
