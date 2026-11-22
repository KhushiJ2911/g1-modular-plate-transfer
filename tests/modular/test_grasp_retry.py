from types import SimpleNamespace

import numpy as np

from method.modular import layout
from method.modular.fallback_seating import FallbackSeatingController
from method.modular.phases import TIMEOUTS, Phase
from submission.policy_grasp_retry import RetainedPinchController


def test_retry_only_preserves_a_settled_bilateral_pinch(monkeypatch):
    monkeypatch.setattr(FallbackSeatingController, "_pickup_task", lambda self, state, task: task)
    ctl = object.__new__(RetainedPinchController)
    ctl.machine = SimpleNamespace(phase=Phase.GRASP, elapsed=TIMEOUTS[Phase.GRASP] - 5)
    ctl.closing = True
    ctl.source = np.zeros(3)
    state = np.zeros(50)
    state[layout.GRIPPERS] = [0.9, 0]
    state[layout.LEFT_FINGERS] = -0.016
    task = {"held": False, "plate_xyz": ctl.source.copy(), "contacts": {"source_rack": True}}
    assert ctl._pickup_task(state, task) is task
    ctl.machine.elapsed += 1
    assert ctl._pickup_task(state, task)["held"]
    state[layout.LEFT_FINGERS.start] = -0.020
    assert ctl._pickup_task(state, task) is task

    ctl.machine.phase = Phase.LIFT
    ctl.lost_streak = 9
    ctl.fk_rise = 0.1
    ctl.lift_origin = np.zeros(3)
    task["plate_xyz"][2] = 0.09
    state[layout.LEFT_FINGERS] = -0.016
    assert ctl._pickup_task(state, task) is task
    task["contacts"]["source_rack"] = False
    assert ctl._pickup_task(state, task)["held"]
    ctl.lost_streak = 8
    assert ctl._pickup_task(state, task) is task
