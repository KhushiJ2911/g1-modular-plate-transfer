"""Collect detached results and publish checkpoint provenance plus the closed-loop phase table."""

import hashlib
import json
import os
from pathlib import Path

import modal

RUNS = Path("/home/keyu/advaitic/plans/modular_controller/runs")
DIR = RUNS / "astra_dagger"


def report():
    if os.environ.get("MODAL_CONFIG_PATH") != "/home/keyu/.modal_modular.toml":
        raise ValueError("Select the dedicated account")
    volume = modal.Volume.from_name("modular-data")

    def read(name):
        return b"".join(volume.read_file("astra-dagger/" + name))

    # Committed volume records survive restarts and expiry of the producer's call.
    expected = {"generation.json", "heads.json", "closed_loop.json"}
    available = {Path(entry.path).name for entry in volume.listdir("astra-dagger")}
    if missing := expected - available:
        print("Awaiting " + ", ".join(sorted(missing)), flush=True)
        return
    generation, fit, audit = [
        json.loads(read(name + ".json")) for name in ("generation", "heads", "closed_loop")
    ]
    for name, record in (("generation", generation), ("heads", fit), ("closed_loop", audit)):
        (DIR / (name + ".json")).write_text(json.dumps(record, indent=2))
    digest = fit["sha256"]
    checkpoint = read("heads.npz")
    if digest != audit["checkpoint"] or hashlib.sha256(checkpoint).hexdigest() != digest:
        raise ValueError("Mismatched estimator digest")
    dest = Path("checkpoints/astra") / (digest + ".npz")
    dest.write_bytes(checkpoint)
    p = audit["per_phase"]
    lines = [
        f"DAgger sidecar SHA-256 `{digest}` is at modular-data:/astra-dagger/heads.npz. Use ADV_HEADS to select it with submission.policy_modular_heads:make_policy.",
        f"Whole-episode split: {len(fit['train_seeds'])} fresh training, {len(fit['validation_seeds'])} selection, {len(fit['test_seeds'])} offline test. The fit included {fit['teacher_count']} historical teacher episodes. One T4 trained for {fit['seconds']:.1f} s including I/O.",
        f"Generation finished {len(generation['episodes'])} verified CPU episodes using {sum(x['seconds'] for x in generation['episodes']) / 3600:.2f} worker-hours (100 container limit). The 3111000-series audit used {sum(x['seconds'] for x in audit['episodes']) / 3600:.2f} worker-hours with four CPU workers after smoke. It recorded {sum(x['success'] for x in audit['episodes'])}/{len(audit['episodes'])} successes and {sum(x['fell'] for x in audit['episodes'])} falls.",
        "Each cell gives raw v1b → new heads on the same observation-driven trajectories. Bias is the mean magnitude of each episode's mean plate XYZ error. Noise is the mean norm of within-episode XYZ standard deviations.",
        "| Phase | Episodes / frames | Bias cm | Noise cm | XY MAE cm | Z MAE cm | XYZ p95 cm | Held FP/FN | Rack contact FP/FN |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in (
        "APPROACH",
        "PREGRASP",
        "GRASP",
        "LIFT",
        "CARRY",
        "ALIGN",
        "LOWER",
        "RELEASE",
        "SETTLE",
        "RETREAT",
        "HOLD",
    ):
        if phase not in p["heads"]:
            lines[-1] += f"\n| {phase} | 0 / 0 | — | — | — | — | — | — | — |"
            continue
        old, new = p["v1b"][phase], p["heads"][phase]
        fields = [
            f"{old[k]:.2f} → {new[k]:.2f}" for k in ("bias_cm", "noise_cm", "xy_mae_cm", "z_mae_cm", "p95_cm")
        ]
        fields += [
            f"{old[k]['fp']}/{old[k]['fn']} → {new[k]['fp']}/{new[k]['fn']}"
            for k in ("held", "target_rack")
        ]
        lines[-1] += (
            f"\n| {phase} | {new['episodes']} / {new['frames']} | " + " | ".join(fields) + " |"
        )
    body = "\n\n".join(lines) + "\n"
    (DIR / "report.md").write_text(body)
    heading = "### Astra · completed DAgger closed-loop audit " + digest[:12]
    for path in (RUNS / "astra_findings.md", Path("EXPERIMENTS.md")):
        if heading not in path.read_text():
            with path.open("a") as f:
                f.write("\n" + heading + "\n\n" + body)
    print(body, flush=True)


if __name__ == "__main__":
    report()
