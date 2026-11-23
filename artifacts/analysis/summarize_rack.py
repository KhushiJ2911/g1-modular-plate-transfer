"""Summarize Brief 21 replay traces. Usage: summarize_rack.py EVAL_ROOT TRACE_DIR."""

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np


def read_trace(path):
    with gzip.open(path, "rt") as f:
        rows = list(csv.DictReader(f))
    trace = {}
    for k in rows[0]:
        if k in {"phase", "reason"}:
            trace[k] = np.array([r[k] for r in rows])
        elif k in {"target_rack", "table", "floor"}:
            trace[k] = np.array([r[k] == "True" for r in rows])
        else:
            trace[k] = np.array([float(r[k]) if r[k] else np.nan for r in rows])
    return trace


def summary(trace, actions):
    s = trace
    t, rack, d, tilt, ph = s["t"], s["target_rack"], 100 * s["distance"], s["tilt"], s["phase"]
    contact = np.flatnonzero(rack)
    opens = np.flatnonzero(ph == "RELEASE")
    tail = actions[-200:, :18]
    active = tail[:, [0, 1, *range(4, 11)]]
    # Match the pre-action states to the final 200 emitted commands.
    tail_states = slice(-len(tail) - 1, -1)
    result = dict(
        steps=int(t[-1]),
        replay_success=bool(s["replay_success"][-1]),
        ever_held=bool(s["held"].any()),
        end_phase=ph[-1],
        end_rack=rack[-1],
        end_held=int(s["held"][-1]),
        end_cm=d[-1],
        end_deg=tilt[-1],
        end_xy_cm=100 * np.hypot(s["dx"][-1], s["dy"][-1]),
        end_dz_cm=100 * s["dz"][-1],
        end_dx_cm=100 * s["dx"][-1],
        end_dy_cm=100 * s["dy"][-1],
        tail_action_diff_rms=np.sqrt(np.mean(np.diff(tail, axis=0) ** 2)),
        tail_action_max_range=np.max(np.ptp(tail, axis=0)),
        tail_active_diff_rms=np.sqrt(np.mean(np.diff(active, axis=0) ** 2)),
        tail_median_xy_cm=100 * np.median(np.hypot(s["dx"][tail_states], s["dy"][tail_states])),
        tail_median_dz_cm=100 * np.median(s["dz"][tail_states]),
        tail_dist_range_cm=np.ptp(d[tail_states]),
        probe_steps=np.sum((actions[:, 18] > 0) & np.all(actions[:, [0, 1, 3]] == 0, axis=1)),
    )
    # Longest uninterrupted held, closed, unsupported interval within 10 cm.
    hover = ~rack & ~s["table"] & ~s["floor"] & (s["held"] > 0) & (s["grip"] > 0) & (d <= 10)
    edges = np.flatnonzero(np.diff(np.r_[False, hover, False]))
    if len(edges):
        start, stop = max(zip(edges[::2], edges[1::2], strict=True), key=lambda v: v[1] - v[0])
        result.update(hover_start=start, hover_steps=stop - start)
        ix = slice(max(start, stop - 200), stop)
        span = actions[ix, :18]
        if len(span) > 1:
            result.update(
                hover_dx_cm=100 * np.median(s["dx"][ix]),
                hover_dy_cm=100 * np.median(s["dy"][ix]),
                hover_xy_cm=100 * np.median(np.hypot(s["dx"][ix], s["dy"][ix])),
                hover_dz_cm=100 * np.median(s["dz"][ix]),
                hover_range_cm=np.ptp(d[ix]),
                hover_action_diff_rms=np.sqrt(np.mean(np.diff(span, axis=0) ** 2)),
                hover_action_max_range=np.ptp(span, axis=0).max(),
            )
    if len(contact):
        c = contact[0]
        above = np.flatnonzero((t >= c) & (tilt > 20))
        if len(above):
            result["first_above_20_after_contact"] = above[0]
        result.update(
            contact_t=c,
            contact_deg=tilt[c],
            contact_cm=d[c],
            contact_plus_10_deg=tilt[min(c + 10, len(t) - 1)],
            contact_plus_25_deg=tilt[min(c + 25, len(t) - 1)],
            contact_plus_50_deg=tilt[min(c + 50, len(t) - 1)],
            min_deg_in_contact=tilt[rack].min(),
            in_contact_under_20=np.sum(rack & (tilt < 20)),
            closed_contact_under_20=np.sum(rack & (s["grip"] > 0) & (tilt < 20)),
            feasible_closed_steps=np.sum(
                rack & ~s["table"] & (s["grip"] > 0) & (tilt <= 20) & (d <= 5)
            ),
        )
    if len(opens):
        i = opens[0]
        result.update(
            open_t=i,
            open_cm=d[i],
            open_deg=tilt[i],
            open_rack=rack[i],
            open_dx_cm=100 * s["dx"][i],
            open_dy_cm=100 * s["dy"][i],
            open_dz_cm=100 * s["dz"][i],
            logged_open_cm=100 * s["logged_d"][i],
            logged_open_deg=s["logged_tilt"][i],
            rendered_open_cm=100 * s["est_d"][i],
            rendered_open_deg=s["est_tilt"][i],
            rendered_open_z_cm=100 * s["est_dz"][i] if "est_dz" in s else np.nan,
        )
    return result


def errors(traces):
    for label, selector in (
        ("regular <=8cm", lambda s: (s["t"] % 200 == 0) & (s["distance"] <= 0.08)),
        (
            "regular <=8cm, closed",
            lambda s: (s["t"] % 200 == 0) & (s["distance"] <= 0.08) & (s["grip"] > 0),
        ),
        ("open decisions, logged", lambda s: (s["reason"] != "") & (s["phase"] == "RELEASE")),
        (
            "re-render vs logged, PLACE/RELEASE",
            lambda s: (
                (s["reason"] != "")
                & np.isin(s["phase"], ["PLACE", "RELEASE"])
                & np.isfinite(s["logged_d"])
            ),
        ),
    ):
        arrays = []
        for s in traces:
            mask = selector(s)
            if label.startswith("open decisions"):
                ed, et = s["logged_d"] - s["distance"], s["logged_tilt"] - s["tilt"]
            elif label.startswith("re-render"):
                ed, et = s["est_d"] - s["logged_d"], s["est_tilt"] - s["logged_tilt"]
            else:
                ed, et = s["est_d"] - s["distance"], s["est_tilt"] - s["tilt"]
            mask &= np.isfinite(ed) & np.isfinite(et)
            if mask.any():
                arrays.append(np.column_stack([100 * ed[mask], et[mask]]))
        if arrays:
            x = np.concatenate(arrays)
            print(
                label,
                "episodes",
                len(arrays),
                "samples",
                len(x),
                "[cm, deg] bias",
                x.mean(0).round(2),
                "MAE",
                abs(x).mean(0).round(2),
                "p90 abs",
                np.percentile(abs(x), 90, axis=0).round(2),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("traces", type=Path)
    args = parser.parse_args()
    records = []
    baseline_failures = []
    other_failures = []
    for p in sorted(args.traces.glob("*.csv.gz")):
        name, seed = p.name.removesuffix(".csv.gz").rsplit("_", 1)
        with (args.root / name / "episodes.csv").open() as f:
            row = next(r for r in csv.DictReader(f) if r["seed"] == seed)
        with np.load(args.root / name / f"episode_{int(row['episode']):03}.actions.npz") as a:
            actions = a["actions"]
        s = read_trace(p)
        if row["success"] == "False":
            (baseline_failures if "s2_st10" in name else other_failures).append(s)
        records.append(
            dict(
                run=name,
                seed=seed,
                original=row["success"],
                original_steps=row["steps"],
                **summary(s, actions),
            )
        )
    if not records:
        raise ValueError("No complete trace files")
    with (args.traces / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dict.fromkeys(k for r in records for k in r))
        writer.writeheader()
        writer.writerows(records)
    for name, group in (
        ("baseline failures", baseline_failures),
        ("tilt-fix failures", other_failures),
    ):
        print(name, len(group))
        errors(group)


if __name__ == "__main__":
    main()
