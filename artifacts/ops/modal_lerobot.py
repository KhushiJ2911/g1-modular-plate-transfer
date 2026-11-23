"""Pretrained-VLA benchmark on Modal: dataset conversion and fine-tuning with LeRobot 0.7.0.

  modal run ops/modal_lerobot.py::convert [--limit 3]
  modal run --detach ops/modal_lerobot.py::train --gpu A100-40GB --name X --args "..."
Volumes: advaitic-data (/data, read), advaitic-lerobot (/lr: datasets/, runs/, hf/).
"""
import subprocess
import time

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("lerobot[pi,smolvla,molmoact2,dataset,training]==0.7.0")
    .env({"HF_HOME": "/lr/hf", "HF_LEROBOT_HOME": "/lr/datasets", "PYTHONDONTWRITEBYTECODE": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file("/home/keyu/advaitic/ops/lerobot_convert.py", "/root/lerobot_convert.py")
    .add_local_file("/home/keyu/advaitic/ops/lerobot_zeroshot.py", "/root/lerobot_zeroshot.py")
    .add_local_file("/home/keyu/advaitic/ops/lerobot_verify_same.py", "/root/lerobot_verify_same.py")
)
data = modal.Volume.from_name("advaitic-data")
lr = modal.Volume.from_name("advaitic-lerobot", create_if_missing=True)
app = modal.App("advaitic-lerobot", image=image)
DS = "/lr/datasets/advaitic/plate_transfer"


def _sh(cmd, log):
    print("+", cmd, flush=True)
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"# {time.strftime('%F %T')} {cmd}\n")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
        p.wait()
        f.write(f"# exit={p.returncode} wall_seconds={time.time() - t0:.0f}\n")
    return p.returncode, round(time.time() - t0)


SRC = "/data/human60 /data/teacher_v1 /data/teacher_v1_dart0"


@app.function(volumes={"/data": data, "/lr": lr}, cpu=4, memory=8192, timeout=3 * 3600)
def convert_job(limit: int = 0, out: str = DS, shard: str = "0/1", src: str = SRC):
    # write to local disk first, then copy (Volume is not a good target for many small writes)
    tmp = "/tmp/ds"
    subprocess.run(f"rm -rf {tmp} {out}", shell=True)
    rc = _sh(f"python /root/lerobot_convert.py {tmp} {src} --limit {limit} --shard {shard} "
             f"--repo-id advaitic/{out.rsplit('/', 1)[-1]}", f"/tmp/convert.log")
    subprocess.run(f"mkdir -p {out.rsplit('/', 1)[0]} && cp -r {tmp} {out} && "
                   f"cp /tmp/convert.log {out}.convert.log", shell=True)
    lr.commit()
    return shard, rc


@app.function(volumes={"/lr": lr}, cpu=8, memory=32768, timeout=2 * 3600)
def merge_job(n: int, ids: str = "", new: str = "plate_transfer"):
    """Merge datasets under /lr/datasets/advaitic. ids: space-separated names (default the n shards)."""
    lr.reload()
    ids = [f"advaitic/{x}" for x in ids.split()] if ids else [f"advaitic/plate_transfer_s{i}" for i in range(n)]
    subprocess.run(f"rm -rf /lr/datasets/advaitic/{new}", shell=True)
    import json as _json
    rc = _sh(f"lerobot-edit-dataset --repo_id {ids[0]} "
             f"--new_repo_id advaitic/{new} --operation.type merge "
             f"--operation.repo_ids '{_json.dumps(ids)}'",
             f"/lr/merge_{new}.log")
    lr.commit()
    return rc


@app.function(volumes={"/lr": lr}, gpu="A100-40GB", cpu=8, memory=65536, timeout=4 * 3600)
def train_job(name: str, args: str, gpu: str, env: str = "", exe: str = "lerobot-train", ds: str = DS):
    lr.reload()
    log = f"/lr/runs/{name}.log"
    subprocess.run("mkdir -p /lr/runs", shell=True)
    _sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; pip show lerobot | head -2", log)
    rc = _sh(f"{env} {exe} --dataset.repo_id=advaitic/{ds.rsplit('/', 1)[-1]} --dataset.root={ds} "
             f"--output_dir=/lr/runs/{name} --job_name={name} --policy.device=cuda "
             f"--policy.push_to_hub=false {args}", log)
    lr.commit()
    return rc


@app.local_entrypoint()
def convert(limit: int = 0, shards: int = 12, src: str = SRC, prefix: str = "plate_transfer_s"):
    if limit:
        print(convert_job.remote(limit, DS + f"_test{limit}", "0/1", src))
        return
    base = DS.rsplit("/", 1)[0]
    jobs = [(0, f"{base}/{prefix}{i}", f"{i}/{shards}", src) for i in range(shards)]
    for shard, rc in convert_job.starmap(jobs):
        print("shard", shard, rc, flush=True)


@app.local_entrypoint()
def merge(shards: int = 12, ids: str = "", new: str = "plate_transfer"):
    print(merge_job.remote(shards, ids, new))


@app.local_entrypoint()
def train(gpu: str, name: str, args: str = "", env: str = "", timeout: int = 4 * 3600,
          exe: str = "lerobot-train", ds: str = DS):
    print(train_job.with_options(gpu=gpu, timeout=timeout).remote(name, args, gpu, env, exe, ds))


@app.function(volumes={"/lr": lr}, cpu=4, memory=32768, timeout=1800)
def verify_job(saved: str, ref: str):
    lr.reload()
    out = subprocess.run(f"python /root/lerobot_verify_same.py {saved} {ref}", shell=True,
                         capture_output=True, text=True)
    return out.stdout + out.stderr[-2000:]


@app.local_entrypoint()
def verify(saved: str, ref: str):
    print(verify_job.remote(saved, ref))


@app.function(volumes={"/lr": lr}, cpu=4, memory=16384, timeout=2 * 3600)
def prefetch_job(repos: str):
    """Download public base checkpoints into the volume's HF cache (CPU, no GPU time)."""
    out = []
    for repo in repos.split():
        rc = _sh(f"python -c \"from huggingface_hub import snapshot_download as s; print(s('{repo}'))\"",
                 "/lr/prefetch.log")
        out.append((repo, rc))
    lr.commit()
    return out


@app.local_entrypoint()
def prefetch(repos: str = "lerobot/smolvla_base lerobot/pi05_base allenai/MolmoAct2"):
    print(prefetch_job.remote(repos))


@app.function(volumes={"/lr": lr}, cpu=2, memory=8192, timeout=1800)
def localize_pi05():
    """Copy lerobot/pi05_base to /lr/models/pi05_base (dereferenced) and point its saved
    preprocessor's tokenizer at /lr/tokenizers/paligemma (the gated google/paligemma tokenizer
    files from the user's own local HF cache). Only that one field changes; diff is printed."""
    import glob
    import json
    import shutil
    from huggingface_hub import snapshot_download
    src = snapshot_download("lerobot/pi05_base")
    dst = "/lr/models/pi05_base"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, symlinks=False)
    changes = []
    for f in glob.glob(f"{dst}/*processor*.json"):
        d = json.load(open(f))
        for step in d.get("steps", []):
            cfg = step.get("config", {})
            if cfg.get("tokenizer_name") == "google/paligemma-3b-pt-224":
                cfg["tokenizer_name"] = "/lr/tokenizers/paligemma"
                changes.append((f, step.get("registry_name")))
        json.dump(d, open(f, "w"), indent=2)
    lr.commit()
    return sorted(glob.glob(f"{dst}/*")), changes


@app.local_entrypoint()
def localize():
    print(localize_pi05.remote())
