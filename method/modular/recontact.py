"""Observation features for diagnosing a plate trapped between nominally open fingers."""

import numpy as np


def mechanical_features(state):
    q = state[..., 7:9] * 1000
    v = state[..., 29:31] * 1000
    return np.concatenate(
        [
            q,
            v,
            np.abs(v),
            np.abs(q[..., :1] - q[..., 1:]),
            np.max(q, axis=-1, keepdims=True),
            state[..., 44:45],
        ],
        axis=-1,
    )


def features(state, vision, plate, site, use_vision):
    x = mechanical_features(state)
    if use_vision:
        return np.concatenate([x, vision, plate, plate - site], axis=-1)
    return x


def score(weights, x):
    return (
        np.clip((x - weights["mean"]) / weights["scale"], -10, 10) @ weights["coef"]
        + weights["bias"]
    )


class OpenFingerContact:
    """Estimated contact with a plate during a commanded-open settle window.

    Call once per controller step after issuing the open command. A hit needs
    an above-threshold measurement now and six steps ago, matching the spacing
    of the rendered training observations. Reset when leaving this window.
    """

    def __init__(self):
        import json
        from collections import deque
        from pathlib import Path

        saved = json.loads(Path(__file__).with_name("open_finger_contact.json").read_text())
        self.weights = {k: np.asarray(saved[k]) for k in ("mean", "scale", "coef", "bias")}
        self.threshold = float(saved["threshold"])
        self._recent = deque(maxlen=7)

    def reset(self):
        self._recent.clear()

    def update(self, state):
        if state[44] > 0.08:
            self.reset()
            return False
        hit = bool(score(self.weights, mechanical_features(state)) >= self.threshold)
        self._recent.append(hit)
        return len(self._recent) == 7 and self._recent[0] and hit
