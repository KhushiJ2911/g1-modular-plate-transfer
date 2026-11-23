"""Analysis only: the real bce1 policy, unchanged, plus a side log of its gripper-open probability.

Actions are exactly what submission.policy returns. ADV_PROBE_OUT = jsonl path.
One line per policy call: t (first control step of the chunk), p (open prob per executed step), a (executed gripper cmd).
"""
import json
import os

import numpy as np

from submission.model import GRIPPER_INDEX
from submission.policy import make_policy as _make


class Probe:
    def __init__(self):
        self.inner = _make()
        self.out = open(os.environ["ADV_PROBE_OUT"], "a")
        self.raw = None
        forward = self.inner.model.forward

        def record(*args, **kwargs):
            self.raw = forward(*args, **kwargs)
            return self.raw

        self.inner.model.forward = record
        self.t = 0

    def reset(self):
        self.inner.reset()
        self.t = 0

    def __call__(self, obs):
        actions = self.inner(obs)
        n = len(actions)
        logit = self.raw[0, :n, GRIPPER_INDEX].float().cpu().numpy()
        p = 1 / (1 + np.exp(-logit))
        self.out.write(json.dumps({"t": self.t, "p": [round(float(x), 4) for x in p],
                                   "a": [float(x) for x in actions[:, 18]]}) + "\n")
        self.out.flush()
        self.t += n
        return actions


def make_policy():
    return Probe()
