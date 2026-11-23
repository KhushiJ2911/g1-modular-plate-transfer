Implemented ResFormer-style value residuals in both microVLA stacks. Each prefix and action expert layer after the first learns its own sigmoid weight, initialized to `0.5`. Cached inference stores the mixed prefix values and recomputes the action stack’s first-layer values at each denoising step. [Zhou et al., *Value Residual Learning*](https://aclanthology.org/2025.acl-long.1375/)

The flag defaults to `false`. Enable it with this exact override.

```bash
ADV_THROTTLE=1 uv run --active --frozen --no-sync python -B -m method.vla.train \
  --config method/vla/microvla_v1.yaml --value-residual \
  --run-dir method/vla/checkpoints/microvla_v1_value_residual
```

Both YAMLs accept `value_residual: true`. Use `--no-value-residual` to override it.

- **Off:** A fixed-seed CPU comparison against `28a180d` matched bit-for-bit for initial weights, forward outputs, gradients, prefix K/V, and cached and uncached samples.
- **On:** Cached and uncached samples agreed. All four logits in the tiny model received finite, nonzero gradients and updated during a two-step synthetic CPU Muon run. The policy wrapper loaded and used the resulting checkpoint. [Tests](/home/keyu/advaitic/wt-muon/method/vla/test_value_residual.py)

Ruff reports two pre-existing findings in the vendored file. All changes are uncommitted.