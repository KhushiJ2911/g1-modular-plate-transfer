"""Simulation jobs on Modal CPU containers: DAgger collection (and later evals).

Usage (from ~/advaitic):
  modal run ops/modal_sim.py::check                       # build image, run advaitic-check
  modal run ops/modal_sim.py::dagger --ckpt student_v1b_bce1.pt --name dagger3 \
      --start 2500000 --count 120 --per-shard 6

Image: the wt-vla worktree (evaluator src + uv.lock identical to the main checkout,
verified 11-21) installed with `uv sync --frozen`; BiGym assets via bigym-download;
`advaitic-check` (official integrity check) must pass at build time.
Policy runs in a separate CPU-only torch env (/root/mlenv), as on the laptop.
Rendering: Mesa software EGL (same as the laptop teacher-data generation).
"""
import os
import subprocess
import time

import modal

CODE = "/root/code"
# Code snapshot for the sim image; ADV_SIM_CODE selects another worktree (e.g. wt-harness).
SIM_CODE = os.environ.get("ADV_SIM_CODE", "/home/keyu/advaitic/wt-vla")
GL_ENV = {
    "MUJOCO_GL": "egl",
    "LIBGL_ALWAYS_SOFTWARE": "1",
    "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    "EGL_PLATFORM": "surfaceless",
    "ORT_DISABLE_TELEMETRY": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libegl1", "libgl1", "libgles2", "libegl-mesa0",
                 "libgl1-mesa-dri", "libosmesa6", "libglib2.0-0")
    .pip_install("uv==0.9.5")
    .env(GL_ENV)
    .add_local_dir(SIM_CODE, CODE, copy=True,
                   ignore=["**/__pycache__", "**/checkpoints", ".git", ".venv", "results"])
    .run_commands(
        f"cd {CODE} && uv sync --frozen --no-dev",
        f"cd {CODE} && uv run bigym-download --task move_plate",
        f"cd {CODE} && uv run advaitic-check",
        "uv venv /root/mlenv --python 3.12",
        "uv pip install --python /root/mlenv/bin/python torch==2.5.1 torchvision==0.20.1 "
        "--index-url https://download.pytorch.org/whl/cpu",
        "uv pip install --python /root/mlenv/bin/python numpy==2.5.5 pyyaml==6.0.3 "
        "pillow==13.0.0 safetensors==0.8.0",
    )
)

ckpts = modal.Volume.from_name("advaitic-ckpts", create_if_missing=True)
runs = modal.Volume.from_name("advaitic-runs", create_if_missing=True)
app = modal.App(os.environ.get("ADV_SIM_APP", "advaitic-sim"), image=image)


@app.function(timeout=900)
def check():
    out = subprocess.run(f"cd {CODE} && uv run advaitic-check", shell=True,
                         capture_output=True, text=True)
    return out.returncode, out.stdout[-1500:] + out.stderr[-1500:]


@app.function(volumes={"/ckpts": ckpts, "/runs": runs}, cpu=4, memory=8192, timeout=3 * 3600)
def collect_shard(ckpt: str, name: str, start: int, stop: int, extra: str = ""):
    # The collector takes fcntl locks in its --out dir; on a Modal Volume that hung
    # (11-21: idle container, empty log). Collect on local disk, copy at the end.
    out = f"/tmp/out/shard_{start}"
    log = f"/tmp/out/shard_{start}.log"
    cmd = (f"mkdir -p /tmp/out && cd {CODE} && {CODE}/.venv/bin/python -m method.dagger.collect "
           f"--checkpoint /ckpts/{ckpt} --seeds {start}:{stop} --out {out} "
           f"--policy-python /root/mlenv/bin/python {extra} > {log} 2>&1")
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True).returncode
    subprocess.run(f"mkdir -p /runs/{name} && cp -r {out} {log} /runs/{name}/", shell=True)
    runs.commit()
    return start, rc, round(time.time() - t0)


# GPU variant: same job, but MuJoCo renders through the NVIDIA EGL driver instead of
# Mesa software rendering (CPU rendering measured ~10x slower than the laptop, 11-21).
GPU_GL = ("env -u LIBGL_ALWAYS_SOFTWARE -u __EGL_VENDOR_LIBRARY_FILENAMES -u EGL_PLATFORM "
          "MUJOCO_GL=egl ")


@app.function(volumes={"/ckpts": ckpts, "/runs": runs}, gpu="T4", cpu=4, memory=8192,
              timeout=3 * 3600)
def collect_shard_gpu(ckpt: str, name: str, start: int, stop: int, extra: str = ""):
    out = f"/tmp/out/shard_{start}"
    log = f"/tmp/out/shard_{start}.log"
    subprocess.run("mkdir -p /tmp/out; nvidia-smi -L > /tmp/out/gpu.txt 2>&1; "
                   "ls /usr/share/glvnd/egl_vendor.d/ >> /tmp/out/gpu.txt 2>&1", shell=True)
    cmd = (f"cd {CODE} && {GPU_GL}{CODE}/.venv/bin/python -m method.dagger.collect "
           f"--checkpoint /ckpts/{ckpt} --seeds {start}:{stop} --out {out} "
           f"--policy-python /root/mlenv/bin/python {extra} > {log} 2>&1")
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True).returncode
    subprocess.run(f"mkdir -p /runs/{name} && cp -r {out} {log} /tmp/out/gpu.txt /runs/{name}/",
                   shell=True)
    runs.commit()
    return start, rc, round(time.time() - t0)


@app.local_entrypoint()
def dagger(ckpt: str, name: str, start: int, count: int, per_shard: int = 6, extra: str = "",
           gpu: bool = False):
    shards = [(ckpt, name, s, min(s + per_shard, start + count), extra)
              for s in range(start, start + count, per_shard)]
    t0 = time.time()
    fn = collect_shard_gpu if gpu else collect_shard
    for s, rc, dt in fn.starmap(shards):
        print(f"shard {s}: exit={rc} {dt}s", flush=True)
    print(f"all shards done in {time.time() - t0:.0f}s")

@app.function(volumes={"/ckpts": ckpts, "/runs": runs}, gpu="T4", cpu=4, memory=8192,
              timeout=2 * 3600)
def eval_run(ckpt: str, name: str, seed_start: int, episodes: int = 30,
             policy: str = "submission.policy:make_policy", env: str = "", extra: str = ""):
    """Official evaluator, unchanged. Rendering on the T4 (NVIDIA EGL, pixel-identical to the
    laptop); the policy runs on CPU in /root/mlenv. Results copied to /runs/evals/<name>."""
    out = f"/tmp/eval/{name}"
    cmd = (f"cd {CODE} && {GPU_GL}ADV_CKPT=/ckpts/{ckpt} {env} {CODE}/.venv/bin/advaitic-eval "
           f"--policy {policy} --policy-python /root/mlenv/bin/python --episodes {episodes} "
           f"--seed-start {seed_start} --method {name} --checkpoint {ckpt} --out {out} {extra} "
           f"> /tmp/eval/{name}.log 2>&1")
    subprocess.run("mkdir -p /tmp/eval", shell=True)
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True).returncode
    subprocess.run(f"mkdir -p /runs/evals && cp -r {out} /tmp/eval/{name}.log /runs/evals/",
                   shell=True)
    runs.commit()
    return name, rc, round(time.time() - t0)


@app.local_entrypoint()
def evals(spec: str):
    """spec: ';'-separated 'ckpt,name,seed_start[,policy]' entries, all run in parallel.
    policy defaults to the BC policy (submission.policy:make_policy)."""
    jobs = [x.split(",") for x in spec.split(";") if x.strip()]
    args = [(j[0], j[1], int(j[2]), 30, j[3] if len(j) > 3 and j[3] else "submission.policy:make_policy",
             j[4] if len(j) > 4 else "")
            for j in jobs]
    t0 = time.time()
    for name, rc, dt in eval_run.starmap(args):
        print(f"{name}: exit={rc} {dt}s", flush=True)
    print(f"all evals done in {time.time() - t0:.0f}s")

# --- Pretrained-VLA evals: same simulator image + a LeRobot 0.7.0 env for the policy process ---
lr_image = image.run_commands(
    "uv venv /root/lrenv --python 3.12",
    "uv pip install --python /root/lrenv/bin/python "
    "'lerobot[pi,smolvla,molmoact2,dataset,training]==0.7.0'",
).env({"HF_HOME": "/lr/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
lr_vol = modal.Volume.from_name("advaitic-lerobot", create_if_missing=True)


@app.function(image=lr_image, volumes={"/lr": lr_vol, "/runs": runs}, gpu="L4", cpu=4,
              memory=32768, timeout=3 * 3600)
def eval_run_lr(ckpt: str, name: str, seed_start: int, episodes: int = 30, extra: str = ""):
    """Official evaluator, unchanged; policy = submission.policy_lerobot in the LeRobot env on the
    L4 GPU; ckpt is a path under /lr/runs (a LeRobot pretrained_model dir)."""
    out = f"/tmp/eval/{name}"
    cmd = (f"cd {CODE} && {GPU_GL}ADV_CKPT=/lr/runs/{ckpt} {CODE}/.venv/bin/advaitic-eval "
           f"--policy submission.policy_lerobot:make_policy --policy-python /root/lrenv/bin/python "
           f"--episodes {episodes} --seed-start {seed_start} --method {name} --checkpoint {ckpt} "
           f"--out {out} {extra} > /tmp/eval/{name}.log 2>&1")
    subprocess.run("mkdir -p /tmp/eval", shell=True)
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True).returncode
    subprocess.run(f"mkdir -p /runs/evals && cp -r {out} /tmp/eval/{name}.log /runs/evals/",
                   shell=True)
    runs.commit()
    return name, rc, round(time.time() - t0)


@app.local_entrypoint()
def evals_lr(spec: str):
    """spec: ';'-separated 'ckpt_dir_under_/lr/runs,name,seed_start' entries, in parallel."""
    jobs = [x.split(",") for x in spec.split(";") if x.strip()]
    t0 = time.time()
    for name, rc, dt in eval_run_lr.starmap([(j[0], j[1], int(j[2]), 30, j[3] if len(j) > 3 else "") for j in jobs]):
        print(f"{name}: exit={rc} {dt}s", flush=True)
    print(f"all evals done in {time.time() - t0:.0f}s")


@app.function(image=lr_image.add_local_file("/home/keyu/advaitic/ops/lr_check.py", "/root/lr_check.py"),
              volumes={"/lr": lr_vol}, gpu="L4", cpu=4, memory=32768, timeout=1800)
def lr_check_job(ckpts: str):
    out = subprocess.run(f"cd {CODE} && /root/lrenv/bin/python /root/lr_check.py {ckpts}", shell=True,
                         capture_output=True, text=True)
    return out.stdout[-6000:] + out.stderr[-3000:]


@app.local_entrypoint()
def lr_check(ckpts: str):
    print(lr_check_job.remote(" ".join(f"/lr/runs/{c}" for c in ckpts.split())))



@app.function(image=image.add_local_file("/home/keyu/advaitic/ops/policy_smoke.py", "/root/policy_smoke.py"),
              volumes={"/ckpts": ckpts, "/runs": runs}, gpu="T4", cpu=4, memory=8192, timeout=1200)
def smoke_run(ckpt: str, policy: str = "submission.policy:make_policy", env: str = "", calls: int = 20):
    """Fast pre-flight (one reset + `calls` policy calls via the evaluator's PolicyProcess). Not a score."""
    cmd = (f"cd {CODE} && {GPU_GL}ADV_CKPT=/ckpts/{ckpt} {env} {CODE}/.venv/bin/python /root/policy_smoke.py "
           f"{policy} /root/mlenv/bin/python {calls}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("SMOKE ")]
    return ckpt, (line[-1][6:] if line else "NO OUTPUT rc=%d %s" % (r.returncode, (r.stdout + r.stderr)[-1500:]))


@app.local_entrypoint()
def smoke(spec: str):
    """spec: ';'-separated 'ckpt[,policy[,env]]'. Run before any full eval batch."""
    jobs = [x.split(",") for x in spec.split(";") if x.strip()]
    for ckpt, res in smoke_run.starmap([(j[0], j[1] if len(j) > 1 and j[1] else
                                         "submission.policy:make_policy", j[2] if len(j) > 2 else "")
                                        for j in jobs]):
        print(ckpt, res, flush=True)
