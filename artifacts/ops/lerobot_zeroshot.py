"""lerobot-train with the weight update disabled -> a step-0 ("zero-shot") checkpoint.

Runs the exact lerobot 0.7.0 training entry point (same feature mapping, dataset normalisation
stats, processors and checkpoint format as our fine-tunes), but replaces update_policy with a
no-op: no forward, no backward, no optimizer step. Use with --steps=1 --save_freq=1.
The saved weights are therefore the released base weights (verified separately).
"""
import sys

import lerobot.scripts.lerobot_train as T


def no_update(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator,
              lr_scheduler=None, lock=None, sample_weighter=None, ema=None):
    train_metrics.loss = 0.0
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = 0.0
    return train_metrics, {}


if __name__ == "__main__":  # dataloader workers re-import this module; only the parent trains
    T.update_policy = no_update
    sys.exit(T.main())
