"""CPU large-config forward/backward and analytical memory accounting.

No real data, cache, teacher weights, GPU, or simulator is accessed.
"""

import json
import os
import time
from pathlib import Path

import torch
from torch import nn

from method.bc.train import read_config
from method.vla.model import (
    MicroVLA,
    flow_loss_terms,
    instruction_tokens,
    model_config,
    parameter_counts,
    token_counts,
)
from method.vla.muon import build_optimizer, optimizer_counts
from method.vla.train import Config, validate_config


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for CPU-only checks")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    values = read_config(Path(__file__).with_name("microvla_large.yaml"))
    config = Config(**{**values, "device": "cpu"})
    validate_config(config)
    model = MicroVLA(model_config(config.horizon, **config.model_overrides))
    counts, tokens = parameter_counts(model), token_counts(model.config)
    assert 35_000_000 <= counts["total"] <= 45_000_000
    assert tokens["total"] == 482
    images = torch.randint(0, 256, (1, 3, 3, 96, 96), dtype=torch.uint8)
    state = torch.zeros(1, 19)
    actions = torch.zeros(1, config.horizon, 10)
    parameter_storage = {p.untyped_storage().data_ptr() for p in model.parameters()}
    saved_storage = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        if pointer not in parameter_storage:
            saved_storage[pointer] = storage.nbytes()
        return tensor

    started = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        total, n = flow_loss_terms(
            model,
            images,
            state,
            instruction_tokens(1, "cpu"),
            actions,
            torch.ones_like(actions, dtype=torch.bool),
        )
    (total / n).backward()
    elapsed = time.perf_counter() - started
    assert any(p.grad is not None for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer = build_optimizer(model, optimizer="muon", lr=config.lr)
    # Exercise the real large matrices, not just the tiny unit-test geometry.
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    results = dict(
        params=counts,
        tokens=tokens,
        flow_optimizer_groups=optimizer_counts(optimizer),
        distill_optimizer_groups=optimizer_counts(
            build_optimizer(
                model, optimizer="muon", distill_head=nn.Linear(model.config.width, 768)
            )
        ),
        cpu_microbatch_1_forward_backward_seconds=elapsed,
        flow_loss=float(total.detach() / n),
        # The final prefix finish result is unused by the vendor action path.
        unused_gradient_parameters=[name for name, p in model.named_parameters() if p.grad is None],
        cpu_saved_nonparameter_storage_bytes_microbatch_1=sum(saved_storage.values()),
    )
    muon_params = sum(
        row["parameters"] for row in results["flow_optimizer_groups"] if row["optimizer"] == "muon"
    )
    adam_params = counts["total"] - muon_params
    # fp32 model, EMA, grads (12 B/param), one Muon / two AdamW moment buffers.
    results["static_training_gib"] = (
        12 * counts["total"] + 4 * muon_params + 8 * adam_params
    ) / 2**30
    # Approximation only: CPU kernels and CUDA bf16 save different intermediates.
    results["saved_activation_bf16_batch64_proxy_gib"] = sum(saved_storage.values()) * 32 / 2**30
    results["one_dense_fp32_joint_score_all_layers_batch64_gib"] = (
        64 * model.config.heads * tokens["total"] ** 2 * model.config.depth * 4 / 2**30
    )
    v1 = MicroVLA(model_config())
    results["v1"] = dict(
        params=parameter_counts(v1),
        flow_optimizer_groups=optimizer_counts(build_optimizer(v1, optimizer="muon")),
        distill_optimizer_groups=optimizer_counts(
            build_optimizer(v1, optimizer="muon", distill_head=nn.Linear(v1.config.width, 768))
        ),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
