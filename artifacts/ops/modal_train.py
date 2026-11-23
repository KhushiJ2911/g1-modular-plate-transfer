"""Run training jobs for both tracks (microVLA, BC student) on Modal GPUs.

Usage (from ~/advaitic):
  modal run ops/modal_train.py::build_cache --module method.vla.train
  modal run ops/modal_train.py::bench --gpu L4 --module method.vla.train --steps 300
  modal run --detach ops/modal_train.py::run --gpu L4 --module method.vla.train --run-name X --args "..."

Data:  volume advaitic-data   -> /data   (raw episodes, uploaded once)
Cache: volume advaitic-cache  -> /cache  (uint8 image cache, built on Modal)
Runs:  volume advaitic-runs   -> /runs   (run dirs, checkpoints, logs)
Code is the local wt-vla worktree, snapshotted at launch; the git SHA is logged.
"""
import os
import shlex
import subprocess
import time

import modal

CODE = "/root/code"
DATA_DIRS = "/data/human60 /data/teacher_v1 /data/teacher_v1_dart0 /data/dagger /data/dagger2"
# Code snapshot to run; ADV_CODE selects e.g. an exact `git archive` export.
LOCAL_CODE = os.environ.get("ADV_CODE", "/home/keyu/advaitic/wt-vla")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy==2.5.5", "transformers==4.57.6", "tokenizers==0.22.2",
        "huggingface-hub==0.36.2", "safetensors==0.8.0", "pyyaml==6.0.3", "pillow==13.0.0",
    )
    # SigLIP teacher for distillation, baked into the image (public weights).
    .run_commands(
        # The trainer only loads the vision tower (method/vla/train.py:228).
        "python -c \"from transformers import SiglipVisionModel;"
        "SiglipVisionModel.from_pretrained('google/siglip-base-patch16-224')\""
    )
    .env({"PYTHONPATH": CODE, "PYTHONDONTWRITEBYTECODE": "1"})
    .add_local_dir(LOCAL_CODE, CODE, ignore=["**/__pycache__", "**/checkpoints", ".git", ".venv"])
)

data = modal.Volume.from_name("advaitic-data")
cache = modal.Volume.from_name("advaitic-cache", create_if_missing=True)
runs = modal.Volume.from_name("advaitic-runs", create_if_missing=True)
ckpts = modal.Volume.from_name("advaitic-ckpts", create_if_missing=True)
VOLS = {"/data": data, "/cache": cache, "/runs": runs, "/ckpts": ckpts}
app = modal.App("advaitic-train", image=image)


def _sh(cmd, log=None):
    print("+", cmd, flush=True)
    t0 = time.time()
    with open(log, "a") if log else open("/dev/null", "w") as f:
        p = subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
        p.wait()
    return p.returncode, time.time() - t0


@app.function(volumes=VOLS, cpu=8, memory=32768, timeout=3600)
def build_cache(module: str = "method.vla.train", data: str = ""):
    """Build the image cache once (CPU only) by running the dataset constructor.
    data: space-separated dataset names under /data (default: the microVLA v1 mix)."""
    dirs = [f"/data/{d}" for d in data.split()] if data else DATA_DIRS.split()
    cfg = "method/vla/microvla_v1.yaml"
    dims = "0,1,2,3,4,5,6,7,8,18,19,20,21,40,41,42,43,44,45"
    code = (
        "import yaml; from method.bc.dataset import EpisodeStore\n"
        f"c=yaml.safe_load(open('{cfg}'))\n"
        f"s=EpisodeStore({dirs!r}, '/cache', val_episodes=c['val_episodes'], "
        f"proprio_dims=[{dims}])\n"
        "print('cache digest', s.digest)\n"
    )
    rc, dt = _sh(f"python -c {shlex.quote(code)}")
    cache.commit()
    return rc, round(dt)


def _train(module, run_name, args, gpu, data_dirs=DATA_DIRS):
    runs.reload()
    run_dir = f"/runs/{run_name}"
    log = f"/runs/{run_name}.log"
    sha = subprocess.run("cat .git_sha 2>/dev/null", shell=True, cwd=CODE,
                         capture_output=True, text=True).stdout.strip()
    with open(log, "a") as f:
        f.write(f"# gpu={gpu} code_sha={sha} module={module} args={args}\n")
    _sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", log)
    cmd = (f"python -m {module} --device cuda --cache-dir /cache --run-dir {run_dir} "
           f"--data-dirs {data_dirs} {args}")
    rc, dt = _sh(cmd, log)
    with open(log, "a") as f:
        f.write(f"# exit={rc} wall_seconds={dt:.0f}\n")
    runs.commit()
    return rc, round(dt)


@app.function(volumes=VOLS, gpu="L4", cpu=8, memory=32768, timeout=6 * 3600)
def train_job(module: str, run_name: str, args: str, gpu: str = "L4", data_dirs: str = DATA_DIRS):
    return _train(module, run_name, args, gpu, data_dirs)


@app.local_entrypoint()
def bench(gpu: str = "L4", module: str = "method.vla.train", steps: int = 300,
          extra: str = "--microbatch-size 64 --num-workers 4"):
    name = f"bench_{module.split('.')[1]}_{gpu}_{int(time.time())}"
    args = f"--steps {steps} --val-every 100000 --sample-every 100000 --save-every 100000 {extra}"
    print(train_job.with_options(gpu=gpu, timeout=3600).remote(module, name, args, gpu))


@app.local_entrypoint()
def run(gpu: str, module: str, run_name: str, args: str = "", data: str = ""):
    """data: space-separated dataset names under /data (default: the microVLA mix)."""
    dirs = " ".join(f"/data/{d}" for d in data.split()) if data else DATA_DIRS
    print(train_job.with_options(gpu=gpu).remote(module, run_name, args, gpu, dirs))
