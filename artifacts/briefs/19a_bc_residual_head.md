# Brief #19a (FAST, ~30 min): residual action head for the BC policy (more gradient pathways)
Worktree /home/keyu/advaitic/wt-bcres (branch bcres, at 5233dc7 = exact bce1 code). Leave files uncommitted (git
metadata is read-only in your sandbox; I commit). No evaluator/physics changes; ADV_THROTTLE=1 for anything
heavier than unit tests; tiny CPU checks only.
Change `submission/model.py` BCModel: add config option `head_type` in {"mlp" (default, EXACTLY the current
head, so old checkpoints load and behave bit-identically), "resmlp"}. "resmlp": Linear(in → 512) then N (config
`head_blocks`, default 3) pre-norm residual blocks x = x + Dropout(Linear(GELU(Linear(LayerNorm(x)))))
(width 512, hidden 1024, dropout 0.1), final LayerNorm + Linear → horizon*len(ACTIVE_DIMS); initialise the last
Linear of each block to zero so the network starts as a shallow model and gradients flow through the skips.
Plumb `head_type`/`head_blocks` through the yaml/CLI (method/bc/train.py) into the checkpoint config and
submission/policy.py loading, exactly like other model options (e.g. how `gripper_loss` travels).
Also add an option `fuse_skip` (default False): concatenate the proprio embedding to the head input of every
residual block's output path... ONLY if trivial; otherwise skip it.
Tests: (1) default head produces identical outputs for the existing bce1 checkpoint
`/home/keyu/advaitic/ckpts/student_v1b_bce1.pt` before/after your change (load + forward on random input,
torch.equal); (2) resmlp forward/backward shapes; (3) a 2-step CPU train with `--head-type resmlp` on 2 episodes
from /home/keyu/advaitic/data/dagger3 into a /tmp run dir (use `--data-weights 1`), and a load of that
checkpoint through submission/policy.py. Report the exact CLI flags.
