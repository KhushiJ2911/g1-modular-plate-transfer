import sys, json, numpy as np
from pathlib import Path
from advaitic_eval.environment import make_environment, observed_source_slot
D=Path("/home/keyu/.cache/huggingface/hub/datasets--SWIRL-Lab--bigym-g1-native60/snapshots/4c3e4414aaf74540899de0b9e82239ef927c9710/move_plate")
st=json.loads((D/"meta/episode_init_states.json").read_text())["episodes"]
idx=[int(i) for i in sys.argv[1].split(',')]
Z=np.load("/home/keyu/advaitic/analysis/demos_lowdim.npz"); demos=[{"action":Z[f"a{i}"],"obs":{"state":Z[f"s{i}"]}} for i in range(60)]
env=make_environment()
out=open(sys.argv[2],'a')
for i in idx:
    e=st[str(i)]; seed=int(np.asarray(e["arrays"]["seed"]["data"]).ravel()[0])
    pre=e["arrays"].get("pre_engage_steps",{}).get("data")
    ts=env.reset(seed=seed); slot=observed_source_slot(env)
    s0=np.asarray(ts.low_dim_obs); dstate=float(np.abs(s0-demos[i]['obs']['state'][0]).max())
    A=demos[i]['action']; n=0; first_succ=None
    for k in range(1700):
        a=A[min(k,len(A)-1)].copy()
        ts=env.step(a); n+=1
        if first_succ is None and env.episode_succeeded(): first_succ=k
        if ts.last(): break
    traj_err=None
    r=dict(demo=i,seed=seed,slot=slot+1,pre_engage=pre,init_state_maxdiff=round(dstate,4),
           success=bool(env.episode_succeeded() and not env.episode_fell()),fell=bool(env.episode_fell()),
           term=env.episode_termination(),steps=n,first_success_step=first_succ,demo_len=len(A))
    print(json.dumps(r),file=out,flush=True); print(r,flush=True)
