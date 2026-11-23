Implemented microVLA training, masked flow loss, resumable checkpoints, cached inference and optional SigLIP distillation. Vendor source remains unchanged.

- CPU smoke and Ruff checks pass.
- Resume matches uninterrupted training bit for bit.
- 11,198,026 parameters; 482 tokens/sample.
- Real training/evaluation remain unverified; the matching five-source cache is missing.

[Report with design choices, normalization stats and commands](/home/keyu/advaitic/wt-vla/method/vla/10_microvla.md). Saved inside the worktree because `/home/keyu/advaitic/reports/` is outside the writable sandbox.