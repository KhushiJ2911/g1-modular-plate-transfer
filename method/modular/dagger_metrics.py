"""Paired perception errors along the new candidate's own independent closed-loop trajectories."""

import json
from pathlib import Path

import numpy as np


def summarize(folder):
    rows = []
    for s in range(3111000, 3111030):
        with np.load(folder / f"{s}.npz") as f:
            truth, pred, raw = f["truth"], f["baseline"], f["raw"]
            assert len(raw) == len(truth) and json.loads(str(f["meta"]))["seed"] == s
            rows.append((s, f["phase"], truth[:, [0, 1, 2, 7, 15]], raw, pred[:, [0, 1, 2, 7, 15]]))
    result = {}
    for arm, i in (("v1b", 3), ("heads", 4)):
        result[arm] = {}
        for phase in sorted(set(np.concatenate([r[1] for r in rows]))):
            errors = []
            guesses = []
            labels = []
            for r in rows:
                m = r[1] == phase
                if m.any():
                    errors.append(100 * (r[i][m, :3] - r[2][m, :3]))
                    guesses.extend(r[i][m, 3:] >= 0.5)
                    labels.extend(r[2][m, 3:] > 0)
            e = np.concatenate(errors)
            g, t = np.array(guesses), np.array(labels)
            entry = dict(
                episodes=len(errors),
                frames=len(e),
                bias_cm=float(np.mean([np.linalg.norm(v.mean(0)) for v in errors])),
                noise_cm=float(np.mean([np.linalg.norm(v.std(0)) for v in errors])),
                bias_xyz_cm=e.mean(0).tolist(),
                xy_mae_cm=float(np.linalg.norm(e[:, :2], axis=1).mean()),
                z_mae_cm=float(abs(e[:, 2]).mean()),
                p95_cm=float(np.percentile(np.linalg.norm(e, axis=1), 95)),
            )
            for j, key in enumerate(("held", "target_rack")):
                entry[key] = dict(
                    fp=int(np.sum(g[:, j] & ~t[:, j])),
                    fn=int(np.sum(~g[:, j] & t[:, j])),
                    positive=int(t[:, j].sum()),
                    negative=int((~t[:, j]).sum()),
                )
            result[arm][phase] = entry
    return result


if __name__ == "__main__":
    import sys

    print(json.dumps(summarize(Path(sys.argv[1]))))
