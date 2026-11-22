#!/usr/bin/env bash
# Run from the worktree with PyTorch in PYTHONPATH. Optional arguments: an RGB/state NPZ
# and --full-prior to exercise 1,700 public policy calls before the reset.
set -eu
root=$(pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export PYTHONPATH="$root:$root/src${PYTHONPATH:+:$PYTHONPATH}"
export ADV_ESTIMATOR="$root/checkpoints/est_v1b/last.pt"
export ADV_HEADS="$root/checkpoints/astra/64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff.npz"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for case in fresh reused; do
    uv run --locked python tests/modular/test_process_state.py "$case" "$tmp/$case" "$@"
done
cmp "$tmp/fresh" "$tmp/reused"
sha256sum "$tmp/fresh" | cut -d' ' -f1
