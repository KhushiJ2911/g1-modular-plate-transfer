"""Calibrate the held plate center from public estimates in the measured left-hand frame."""

import json
from pathlib import Path

import modal

from ops_modular.astra_v2_train import image

app = modal.App("modular-astra-center-calibration")
records = modal.Volume.from_name("modular-runs")


@app.function(image=image, cpu=0.25, memory=512, volumes={"/runs": records}, timeout=300)
def fit(plan, paths):
    import numpy as np

    if len(plan) != len(paths) or len(paths) != 18:
        raise ValueError("Expected the original 12-fit/six-check grasp calibration")
    rows = []
    for seed, _, split in plan:
        path = Path(paths[str(seed)])
        if not 3056500 <= seed < 3056600 or not path.is_relative_to("/runs/grasp-validation"):
            raise ValueError("Expected a generation handoff trace")
        with np.load(path, allow_pickle=False) as f:
            trace = json.loads(str(f["rows"]))
            summary = json.loads(str(f["summary"]))
        if seed != summary["seed"] or not summary["secure"]:
            raise ValueError("Incomplete physical grasp in generation calibration")
        lift = [r for r in trace if r["phase"] == "LIFT"]
        last = [r for r in lift if summary["carry"] - 9 <= r["step"] <= summary["carry"]]
        if len(last) != 10 or not all(r["held"] for r in last):
            raise ValueError("Missing held label in the original calibration window")
        # _enter(LIFT) latches the measured pose on the final GRASP action.
        pinch = next(
            r for r in reversed(trace) if r["phase"] == "GRASP" and r["step"] < last[0]["step"]
        )

        def in_hand(frame, key):
            return np.array(frame["rotation"]).T @ (np.array(frame[key]) - frame["hand"])

        rows.append(
            dict(
                seed=seed,
                split=split,
                truth=np.median([in_hand(r, "truth") for r in last], axis=0),
                visual=np.median([in_hand(r, "visual") for r in last], axis=0),
                pinch=in_hand(pinch, "used"),
            )
        )

    def models(train, test):
        nominal = np.median([z["truth"] for z in train], axis=0)
        out = {"prior": nominal, "visual": test["visual"], "pinch": test["pinch"]}
        for key in ("visual", "pinch"):
            bias = np.median([z[key] - z["truth"] for z in train], axis=0)
            out[key + "_calibrated"] = test[key] - bias
        out["consensus"] = np.median(
            np.array([nominal, out["visual_calibrated"], out["pinch_calibrated"]]), axis=0
        )
        return out

    train = [z for z in rows if z["split"] == "fit"]
    check = [z for z in rows if z["split"] == "check"]
    if (len(train), len(check)) != (12, 6):
        raise ValueError("Changed calibration split")
    all_errors = {}
    predictions = []
    for split, group in (("leave_one_out", train), ("check", check)):
        errors = {}
        for z in group:
            other = [x for x in train if x["seed"] != z["seed"]]
            pred = models(other, z)
            errors_per = {k: float(100 * np.linalg.norm(v - z["truth"])) for k, v in pred.items()}
            predictions.append(
                dict(seed=z["seed"], split=split, truth=z["truth"].tolist(), cm=errors_per)
            )
            for k, v in errors_per.items():
                errors.setdefault(k, []).append(v)
        all_errors[split] = {
            k: dict(
                mean=float(np.mean(v)), p50=float(np.median(v)), p95=float(np.percentile(v, 95))
            )
            for k, v in errors.items()
        }
    winner = min(all_errors["leave_one_out"], key=lambda k: all_errors["leave_one_out"][k]["mean"])
    out = dict(
        errors_cm=all_errors,
        training_choice=winner,
        prior=np.median([z["truth"] for z in train], axis=0).tolist(),
        biases={
            k: np.median([z[k] - z["truth"] for z in train], axis=0).tolist()
            for k in ("visual", "pinch")
        },
        episodes=predictions,
    )
    dest = Path("/runs/astra-integration/center_calibration.json")
    dest.write_text(json.dumps(out))
    records.commit()
    return out
