"""Hybrid Muon/AdamW for microVLA, without requiring torch.optim.Muon.

Credit: Keller Jordan's Muon (https://github.com/KellerJordan/Muon) for
Nesterov + quintic Newton--Schulz; Moonlight (https://arxiv.org/abs/2502.16982)
for the 0.2 * sqrt(max(rows, cols)) update-RMS matching rule.
"""

import math

import torch


@torch.no_grad()
def newton_schulz(update, steps=5):
    """Approximate the polar factor; fp32 CPU, bf16 CUDA, rows <= columns.

    The deliberately nonconvergent quintic gives approximate unit singular
    values, not an exact SVD. Near-zero singular values may need more steps.
    """
    if update.ndim != 2 or steps < 1:
        raise ValueError("Newton--Schulz requires a matrix and positive steps")
    with torch.autocast(device_type=update.device.type, enabled=False):
        x = update.to(torch.bfloat16 if update.is_cuda else torch.float32)
        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.T
        x = x / (x.norm() + 1e-7)
        for _ in range(steps):
            gram = x @ x.T
            polynomial = -4.7750 * gram + 2.0315 * (gram @ gram)
            x = 3.4445 * x + polynomial @ x
        return x.T if transposed else x


@torch.no_grad()
def muon_update(update, steps=5, split_sizes=None):
    """Orthogonalise/scale each projection independently, including fused QKV."""
    pieces = (update,) if split_sizes is None else update.split(split_sizes, dim=0)
    result = [newton_schulz(p, steps) * (0.2 * math.sqrt(max(p.shape))) for p in pieces]
    return result[0] if split_sizes is None else torch.cat(result, dim=0)


class HybridMuon(torch.optim.AdamW):
    """One serialisable optimizer: Muon matrices and ordinary AdamW fallback.

    Reuse PyTorch's AdamW implementation and state format for fallback groups.
    Muon momentum buffers live in the same state dict, so ordinary optimizer
    checkpoint loading restores both algorithms and their group learning rates.
    """

    def __init__(self, groups, *, lr, weight_decay, momentum=0.95, ns_steps=5):
        if not 0 <= momentum < 1 or not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError("Invalid Muon momentum or Newton--Schulz steps")
        super().__init__(groups, lr=lr, weight_decay=weight_decay)
        for group in self.param_groups:
            group.setdefault("use_muon", False)
            group.setdefault("momentum", momentum)
            group.setdefault("ns_steps", ns_steps)
            if group["use_muon"] and any(p.ndim != 2 for p in group["params"]):
                raise ValueError("Muon groups must contain only matrices")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = self.param_groups
        for group in groups:
            if not group["use_muon"]:
                continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    # AdamW's checkpoint loader expects a scalar step in every
                    # nonempty state; it is also useful for resume inspection.
                    state["step"] = torch.tensor(0.0)
                state["step"] += 1
                buffer = state["momentum_buffer"]
                buffer.lerp_(grad, 1 - group["momentum"])
                # Nesterov lookahead, without modifying the accumulated gradient.
                nesterov = grad.lerp(buffer, group["momentum"])
                update = muon_update(nesterov, group["ns_steps"], group.get("split_sizes"))
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"])
        try:
            self.param_groups = [g for g in groups if not g["use_muon"]]
            super().step()
        finally:
            self.param_groups = groups
        return loss


def hybrid_groups(model, *, distill_head=None):
    """Explicit structural whitelist; every other trainable tensor uses AdamW.

    Modulation stays on AdamW: zero-initialised scale/shift/gate projections
    control conditioning rather than transformer feature mixing.
    """
    candidates = {}

    def matrix(p, component, split_sizes=None):
        if p.ndim != 2 or id(p) in candidates:
            raise ValueError("Invalid or duplicate Muon matrix")
        candidates[id(p)] = (component, split_sizes)

    for block in model.vision.blocks:
        d = block.self_attn.embed_dim
        matrix(block.self_attn.in_proj_weight, "vision", (d, d, d))
        for p in (block.self_attn.out_proj.weight, block.linear1.weight, block.linear2.weight):
            matrix(p, "vision")
    if distill_head is None:
        for block in model.layers:
            for component, expert in (("prefix", block.prefix), ("expert", block.action)):
                h, d = expert.heads, expert.head_dim
                matrix(expert.qkv.weight, component, (h * d, d, d))
                for p in (expert.out.weight, expert.ffn.up.weight, expert.ffn.down.weight):
                    matrix(p, component)

    named = (
        [("vision." + n, p) for n, p in model.vision.named_parameters()]
        + [("distill_head." + n, p) for n, p in distill_head.named_parameters()]
        if distill_head is not None
        else list(model.named_parameters())
    )
    groups, assigned = {}, set()
    for name, p in named:
        if not p.requires_grad:
            continue
        if id(p) in assigned:
            raise ValueError(f"Duplicate parameter: {name}")
        assigned.add(id(p))
        if id(p) in candidates:
            component, split = candidates[id(p)]
            kind = "qkv" if split else "matrices"
            key = (True, component, kind)
        else:
            component = (
                "vision"
                if name.startswith("vision.")
                else "prefix"
                if name.startswith("layers.") and ".prefix." in name
                else "expert"
                if name.startswith("layers.") and ".action." in name
                else "distill_head"
                if name.startswith("distill_head.")
                else "embeddings_heads"
            )
            split, kind = None, "fallback"
            key = (False, component, kind)
        if key not in groups:
            groups[key] = dict(
                params=[], use_muon=key[0], group_name=f"{component}/{kind}", split_sizes=split
            )
        groups[key]["params"].append(p)
    if not assigned or not set(candidates).issubset(assigned):
        raise ValueError("Missing trainable Muon parameters")
    return list(groups.values())


def build_optimizer(
    model,
    *,
    optimizer="adamw",
    lr=3e-4,
    weight_decay=0.01,
    muon_lr=None,
    momentum=0.95,
    ns_steps=5,
    distill_head=None,
):
    if optimizer == "adamw":
        # Preserve legacy parameter order, one group, and the AdamW state format.
        parameters = (
            [*model.vision.parameters(), *distill_head.parameters()]
            if distill_head is not None
            else list(model.parameters())
        )
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer != "muon":
        raise ValueError(f"Unknown optimizer: {optimizer}")
    groups = hybrid_groups(model, distill_head=distill_head)
    for group in groups:
        group["lr"] = muon_lr if group["use_muon"] and muon_lr is not None else lr
        group["base_lr"] = group["lr"]
    return HybridMuon(
        groups, lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps
    )


def optimizer_counts(optimizer):
    return [
        dict(
            optimizer="muon" if g.get("use_muon", False) else "adamw",
            group=g.get("group_name", "all"),
            tensors=len(g["params"]),
            parameters=sum(p.numel() for p in g["params"]),
        )
        for g in optimizer.param_groups
    ]


def print_optimizer_table(optimizer, phase):
    rows = optimizer_counts(optimizer)
    print(f"Optimizer parameters ({phase})", flush=True)
    print(f"{'optimizer':<10} {'group':<30} {'tensors':>8} {'parameters':>14}", flush=True)
    for row in rows:
        print(
            f"{row['optimizer']:<10} {row['group']:<30} {row['tensors']:>8} "
            f"{row['parameters']:>14,}",
            flush=True,
        )
    for name in ("muon", "adamw"):
        count = sum(row["parameters"] for row in rows if row["optimizer"] == name)
        print(f"{name:<10} {'TOTAL':<30} {'':>8} {count:>14,}", flush=True)
