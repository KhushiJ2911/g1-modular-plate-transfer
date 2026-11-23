"""Check fine-tuned LeRobot checkpoints: adapter output, normaliser stats source, trainable params."""
import glob, json, sys, time
import numpy as np, torch
sys.path.insert(0, "/root/code")
from submission.policy_lerobot import LeRobotPolicy
ds_stats = json.load(open("/lr/datasets/advaitic/plate_transfer/meta/stats.json"))
for ck in sys.argv[1:]:
    print("=====", ck, flush=True)
    t = time.time()
    p = LeRobotPolicy(ck)
    print("load s", round(time.time() - t, 1))
    pol = p.policy
    n_all = sum(x.numel() for x in pol.parameters())
    cfg = json.load(open(f"{ck}/config.json"))
    print("type", cfg.get("type"), "params %.1fM" % (n_all / 1e6),
          {k: cfg.get(k) for k in ("train_expert_only", "freeze_vision_encoder", "train_mode_vlm",
                                   "chunk_size", "n_action_steps") if k in cfg})
    # trainable params as configured by the policy's own init (from the training code path)
    try:
        g = pol.get_optim_params()
        g = g if isinstance(g, (list, tuple)) else [g]
        n_tr = 0
        for item in g:
            ps = item["params"] if isinstance(item, dict) else [item]
            n_tr += sum(x.numel() for x in ps if getattr(x, "requires_grad", False))
        print("trainable (get_optim_params, requires_grad) %.1fM" % (n_tr / 1e6))
    except Exception as e:
        print("get_optim_params failed", repr(e)[:200])
    # normaliser stats: find action mean/std inside the saved preprocessor state
    for f in glob.glob(f"{ck}/*.safetensors"):
        if "processor" in f:
            from safetensors.torch import load_file
            st = load_file(f)
            for k, v in st.items():
                if "action" in k and ("mean" in k or "std" in k):
                    ref = np.asarray(ds_stats["action"]["mean" if "mean" in k else "std"])
                    print(" ", f.rsplit("/", 1)[-1], k, "max|diff vs dataset|=%.2e" % np.abs(v.numpy() - ref).max())
    obs = {"rgb": np.random.randint(0, 255, (3, 3, 84, 84), dtype=np.uint8),
           "state": np.zeros(50, dtype=np.float32)}
    p.reset(); t = time.time(); a = p(obs); dt = time.time() - t
    t = time.time(); a = p(obs); dt2 = time.time() - t
    print("action", a.shape, a.dtype, "range", float(a.min()), float(a.max()),
          "gripper", a[:, 18].tolist()[:3], "call s %.2f / %.2f" % (dt, dt2), flush=True)
