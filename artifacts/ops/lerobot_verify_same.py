"""Compare a saved LeRobot checkpoint's tensors with a reference safetensors.
Keys are matched after stripping a leading 'model.' from either side. Reports identical tensors after
casting the reference to the saved dtype, and lists keys present on one side only."""
import sys

import torch
from safetensors.torch import load_file


def norm(d):
    d = {k[6:] if k.startswith("model.") else k: v for k, v in d.items()}
    return {k.replace("vision_tower.vision_model.", "vision_tower."): v for k, v in d.items()}


a, b = norm(load_file(sys.argv[1])), norm(load_file(sys.argv[2]))
shared = sorted(set(a) & set(b))
diff = [k for k in shared if not torch.equal(a[k], b[k].to(a[k].dtype))]
maxdiff = max(((a[k].float() - b[k].float()).abs().max().item() for k in shared), default=0)
print(f"saved={len(a)} ref={len(b)} shared={len(shared)} identical={len(shared) - len(diff)} "
      f"max_abs_diff={maxdiff:.3g}")
print("differ", diff[:10])
print("only_in_saved", [k for k in a if k not in b][:10])
print("only_in_ref", [k for k in b if k not in a][:10])
