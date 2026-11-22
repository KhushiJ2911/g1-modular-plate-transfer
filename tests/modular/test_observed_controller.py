import numpy as np

from method.modular.estimated import SITE_ROTATION
from method.modular.observed_controller import SITE_FRAMES, LockedSiteContacts


def test_seating_probe_keeps_the_same_site_after_lock():
    state = np.zeros(50)
    filt = LockedSiteContacts()
    site = np.array([0.55, -0.14, 0.85])
    task = {
        "site_xyz": site.copy(),
        "plate_xyz": site + [0, 0, -0.040],
        "plate_rot": SITE_ROTATION,
        "contacts": {"target_rack": False, "source_rack": False, "table": False, "floor": False},
    }
    for _ in range(SITE_FRAMES):
        filt.update(state, task)
    later = {**task, "site_xyz": site + [0, -0.084, 0]}
    corrected = filt.update(state, later)
    np.testing.assert_allclose(corrected["site_xyz"], site)
    assert corrected["contacts"]["target_rack"]
    assert not corrected["contacts"]["table"]
    assert not task["contacts"]["target_rack"]


def test_seating_probe_clears_stale_site_and_rejects_unseated_plate():
    state = np.zeros(50)
    filt = LockedSiteContacts()
    site = np.array([0.60, -0.20, 0.85])
    task = {
        "site_xyz": site,
        "plate_xyz": site + [0, 0, -0.005],
        "plate_rot": SITE_ROTATION,
        "contacts": {"target_rack": True, "source_rack": False, "table": False, "floor": False},
    }
    filt.update(state, {**task, "site_xyz": site + 0.1})
    filt.reset()
    result = filt.update(state, task)
    np.testing.assert_allclose(result["site_xyz"], site)
    assert not result["contacts"]["target_rack"]


def test_source_filter_holds_walking_anchor_and_keeps_lift_measurement_live():
    from method.modular.observed_controller import StableSourcePlate
    from method.modular.phases import Phase

    state = np.zeros(50)
    filt = StableSourcePlate()
    pos = np.array([0.56, 0.24, 0.817])
    task = {"plate_xyz": pos}
    for _ in range(SITE_FRAMES):
        filt.update(state, task, Phase.APPROACH, 0)
    noisy = {"plate_xyz": pos + [-0.06, -0.04, -0.02]}
    for step in range(SITE_FRAMES + 1):
        cmd = filt.update(state, noisy, Phase.APPROACH, step)
    np.testing.assert_array_equal(cmd["plate_xyz"], pos)
    closing = filt.update(state, noisy, Phase.GRASP, 30, closing=True)
    np.testing.assert_allclose(closing["plate_xyz"][2], pos[2] - 0.005)
    assert filt.update(state, noisy, Phase.LIFT, 0, closing=True) is noisy
