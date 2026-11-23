import modal

app = modal.App("advaitic-hello")


@app.function(gpu="T4", timeout=300)
def gpu_check():
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return out.stdout.strip() or out.stderr.strip()


@app.local_entrypoint()
def main():
    print("GPU:", gpu_check.remote())
