"""Independent CPU candidate audit and detached end-to-end training supervisor."""

import json
import time
from pathlib import Path

import modal

from ops_modular.astra_dagger import data, image, run_episode, volume

app = modal.App("modular-astra-dagger-audit")


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=5400,
    max_containers=30,
    volumes={"/runs": volume, "/data": data},
    scaledown_window=5,
)
def audit(seed, digest):
    if not 3111000 <= seed < 3111030:
        raise ValueError("Closed-loop audit seed range required")
    return run_episode(seed, "heads", digest)


@app.function(image=image, cpu=0.125, memory=512, timeout=86400, volumes={"/data": data})
def supervise():
    import subprocess

    folder = Path("/data/astra-dagger")

    def wait_for(count):
        while True:
            data.reload()
            if all(
                (folder / "sol" / f"{s}.npz").is_file() for s in range(3110000, 3110000 + count)
            ):
                return
            time.sleep(45)

    wait_for(1)
    trainer = modal.Function.from_name("modular-astra-dagger-fit", "fit")
    trainer.spawn(True).get()
    wait_for(300)
    trained = trainer.spawn(False).get()
    digest = trained["sha256"]
    smoke = audit.spawn(3111000, digest).get()
    assert smoke["frames"] > 10 and smoke["seed"] == 3111000
    calls = [audit.spawn(s, digest) for s in range(3111001, 3111030)]
    rows = [smoke] + [c.get() for c in calls]
    data.reload()
    cmd = ["/root/policy/bin/python", "-m", "method.modular.dagger_metrics", str(folder / digest)]
    metrics = json.loads(subprocess.check_output(cmd, cwd="/root/code", text=True))
    result = dict(checkpoint=digest, episodes=rows, per_phase=metrics)
    (folder / "closed_loop.json").write_text(json.dumps(result, indent=2))
    data.commit()
    return result
