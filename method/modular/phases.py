"""Transitions and timeout recovery for the geometric plate-transfer controller."""

from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    APPROACH = "APPROACH"
    PREGRASP = "PREGRASP"
    GRASP = "GRASP"
    LIFT = "LIFT"
    CARRY = "CARRY"
    ALIGN = "ALIGN"
    LOWER = "LOWER"
    RELEASE = "RELEASE"
    SETTLE = "SETTLE"
    RETREAT = "RETREAT"
    HOLD = "HOLD"


NEXT = {
    Phase.APPROACH: Phase.PREGRASP,
    Phase.PREGRASP: Phase.GRASP,
    Phase.GRASP: Phase.LIFT,
    Phase.LIFT: Phase.CARRY,
    Phase.CARRY: Phase.ALIGN,
    Phase.ALIGN: Phase.LOWER,
    Phase.LOWER: Phase.RELEASE,
    Phase.RELEASE: Phase.SETTLE,
    Phase.SETTLE: Phase.RETREAT,
    Phase.RETREAT: Phase.HOLD,
}

TIMEOUTS = {
    Phase.APPROACH: 260,
    Phase.PREGRASP: 120,
    Phase.GRASP: 120,
    Phase.LIFT: 165,
    Phase.CARRY: 350,
    Phase.ALIGN: 160,
    Phase.LOWER: 160,
    Phase.RELEASE: 80,
    Phase.SETTLE: 140,
    Phase.RETREAT: 90,
}

GRASP_PHASES = (Phase.PREGRASP, Phase.GRASP, Phase.LIFT)
PLACE_PHASES = (Phase.ALIGN, Phase.LOWER)
SETTLE_STEPS = 75


class PhaseMachine:
    """Keep commanded-action clocks and bound retries across an episode."""

    def __init__(self):
        self.phase = Phase.APPROACH
        self.elapsed = 0
        self.step = 0
        self.grasp_retries = 0
        self.place_retries = 0
        self.transitions = []

    def _move(self, dest: Phase, reason: str):
        self.transitions.append(
            {"from": self.phase.value, "to": dest.value, "step": self.step, "reason": reason}
        )
        self.phase = dest
        self.elapsed = 0

    def advance(self, ready: bool, *, lost: bool = False, unsafe_retreat: bool = False):
        phase = self.phase
        if phase == Phase.HOLD:
            return
        if unsafe_retreat:
            self._move(Phase.HOLD, "retreat_guard")
        elif ready and not lost:
            self._move(NEXT[phase], "exit_check")
        elif lost or self.elapsed >= TIMEOUTS[phase]:
            if phase in GRASP_PHASES and self.grasp_retries < 2:
                self.grasp_retries += 1
                self._move(Phase.PREGRASP, "retry_grasp")
            elif phase in PLACE_PHASES and not lost and self.place_retries < 1:
                self.place_retries += 1
                self._move(Phase.ALIGN, "retry_placement")
            else:
                self._move(Phase.HOLD, "lost_grasp" if lost else "phase_timeout")

    def tick(self):
        self.elapsed += 1
        self.step += 1
