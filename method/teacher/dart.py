"""Episode-local action drift, applied outside the clean teacher controller."""

from __future__ import annotations

import numpy as np

THETA = 0.15
GUARD_STEPS = 15


def validate_scale(scale: float) -> float:
    scale = float(scale)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("dart_noise must be finite and nonnegative")
    return scale


class DartNoise:
    """Discrete OU drift with the requested stationary marginal standard deviation.

    x[t+1] = (1-theta)*x[t] + std*sqrt(theta*(2-theta))*N(0,1).
    Start at zero; advance on every physical step, including protected steps.
    Guards mask the applied drift without modifying the teacher's clean action.
    """

    def __init__(self, seed: int, scale: float):
        self.scale = validate_scale(scale)
        self.rng = np.random.default_rng(seed + 10_000_019)
        self.value = np.zeros(20)
        self.std = np.zeros(20)
        self.std[:2] = 0.10 * self.scale
        self.std[4:11] = 0.02 * self.scale
        self.previous_closed = np.zeros(2, dtype=bool)
        self.last_transition = None
        self.transitions = None
        self.step = 0

    def execute(self, clean, teacher, demo_step: int) -> np.ndarray:
        """Call after teacher.act, with its demo clock from BEFORE that call."""
        closed = clean[18:20] > 0
        if np.any(closed != self.previous_closed):
            self.last_transition = self.step
        self.previous_closed = closed.copy()
        if self.transitions is None:
            reference = teacher.actions[teacher.chosen_demo]
            self.transitions = (
                np.flatnonzero(
                    np.any((reference[1:, 18:20] > 0) != (reference[:-1, 18:20] > 0), axis=1)
                )
                + 1
            )
            # A reference starting closed transitions on its first command.
            if np.any(reference[0, 18:20] > 0):
                self.transitions = np.r_[0, self.transitions]
            # The controller opens both fingers at the end even if a reference
            # were to finish closed. Guard that possible synthetic transition.
            if np.any(reference[-1, 18:20] > 0):
                self.transitions = np.r_[self.transitions, len(reference)]
        before_transition = np.any(
            (self.transitions >= demo_step) & (self.transitions - demo_step <= GUARD_STEPS)
        )
        after_transition = (
            self.last_transition is not None and self.step - self.last_transition <= GUARD_STEPS
        )
        retreat = teacher.release_physical_step is not None
        self.step += 1
        if self.scale == 0:
            return clean.copy()
        self.value = (1 - THETA) * self.value + (
            self.std * np.sqrt(THETA * (2 - THETA)) * self.rng.standard_normal(20)
        )
        if before_transition or after_transition or retreat:
            return clean.copy()
        drift = self.value.copy()
        if getattr(teacher, "alignment_done", False):
            drift[:2] = 0  # Preserve the frozen base throughout lowering.
        return np.clip(clean + drift, -1, 1).astype(np.float32)
