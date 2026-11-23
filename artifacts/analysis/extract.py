import json, numpy as np
from pathlib import Path
from bigym.loco import make_gym
D=Path("/home/keyu/.cache/huggingface/hub/datasets--SWIRL-Lab--bigym-g1-native60/snapshots/4c3e4414aaf74540899de0b9e82239ef927c9710/move_plate")
st=json.loads((D/"meta/episode_init_states.json").read_text())["episodes"]
g=make_gym("move_plate"); demos=g.get_demos(60, decode_images=False)
out={}
for i,d in enumerate(demos):
    e=st[str(i)]
    out[f"a{i}"]=np.asarray(d['action'],np.float32); out[f"s{i}"]=np.asarray(d['obs']['state'],np.float32)
    out[f"seed{i}"]=np.int64(np.asarray(e["arrays"]["seed"]["data"]).ravel()[0])
np.savez_compressed("demos_lowdim.npz",**out); print("ok")
